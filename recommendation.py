import os
import asyncio
import logging
import json
import re
from typing import List, Dict, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from sentence_transformers import CrossEncoder
from semanticrag import semantic_rag
from graphrag import graphrag_response
import hashlib
from langchain_core.tools import tool
import redis
import time

# ==== Logging Setup ====
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==== Environment Variables ====
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REDIS_URL = os.getenv("REDIS_URL")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set in environment variables")

# ==== Initialize Clients ====
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=0.5)
cross_encoder = CrossEncoder("BAAI/bge-reranker-base")

# Initialize Redis client
redis_client = None
if REDIS_URL:
    try:
        redis_client = redis.from_url(REDIS_URL)
        redis_client.ping()
        logger.info(f"Connected to Redis: {REDIS_URL}")
    except redis.exceptions.ConnectionError as e:
        logger.error(f"Error connecting to Redis: {e}")
        redis_client = None
else:
    logger.info("REDIS_URL not set. Caching disabled")

# ==== State Definition ====
class AgentState(BaseModel):
    query: str
    emotion: str = ""
    recommendations: List[Dict] = Field(default_factory=list)
    final_answer: Optional[List[str]] = None
    cache_key: Optional[str] = None

# ==== Query Validation ====
def validate_query(query: str) -> bool:
    """
    Validate if the query contains a recognizable category or price.
    """
    KNOWN_CATEGORIES = [
        'tshirt', 'jacket', 'shirt', 'jeans', 'dress', 'shoes', 'coat', 'hat', 'bag', 'trousers', 'shorts',
        'denim', 'pants', 'bottoms', 'sweater', 'bodysuit'  # Added bottoms and others
    ]
    query_lower = query.lower()
    for cat in KNOWN_CATEGORIES:
        if re.search(r'\b' + re.escape(cat) + r's?\b', query_lower):
            return True
    if re.search(r'₹?\d+|under|below|less than|max|budget', query_lower):
        return True
    return False

# ==== Emotion Analysis ====
async def analyze_emotion(query: str) -> str:
    prompt = """
    Analyze the emotional tone of the query and classify it as one of: positive, negative, neutral, excited, frustrated, tired, or curious.
    Return only the emotion label.
    Query: {query}
    """
    try:
        response = await llm.ainvoke(prompt.format(query=query))
        return response.content.strip()
    except Exception as e:
        logger.warning(f"Error in emotion analysis: {e}")
        return "neutral"

# ==== Query Enhancement ====
async def enhance_query(query: str, emotion: str) -> str:
    prompt = """
    Rewrite the query to be specific and aligned with the user's emotional tone, keeping category and price constraints intact.
    Map 'jeans' to 'Bottoms' if mentioned, as the dataset uses 'Bottoms' for jeans-like products.
    Emotion: {emotion}
    Query: {query}
    Return the rewritten query as a single string.
    """
    try:
        response = await llm.ainvoke(prompt.format(emotion=emotion, query=query))
        return response.content.strip()
    except Exception as e:
        logger.warning(f"Error in query enhancement: {e}")
        return query

# ==== Cache Layer ====
def cached_results(cache_key: str) -> Optional[List[Dict]]:
    if not redis_client:
        logger.info("Skipping cache check: Redis unavailable")
        return None
    try:
        cached_data = redis_client.get(cache_key)
        if cached_data:
            logger.info(f"Cache hit: {cache_key}")
            return json.loads(cached_data)
        logger.info(f"Cache miss: {cache_key}")
        return None
    except Exception as e:
        logger.warning(f"Error accessing Redis cache: {e}")
        return None

def save_to_cache(cache_key: str, results: List[Dict], ex: int = 3600):
    if not redis_client:
        logger.info("Skipping cache save: Redis unavailable")
        return
    try:
        redis_client.set(cache_key, json.dumps(results), ex=ex)
        logger.info(f"Saved to cache: {cache_key}")
    except Exception as e:
        logger.warning(f"Error saving to Redis cache: {e}")

# ==== Parse Results ====
def parse_results(results: str) -> List[Dict]:
    try:
        parsed = []
        if not results or results in ["I'm not sure based on the current data.", "😕 No products found matching your query."]:
            return parsed
        lines = results.split("\n")
        current = {}
        for line in lines:
            line = line.strip()
            if line.startswith("**") and "₹" in line:
                if current:
                    parsed.append(current)
                name_price = line.split("(₹")
                current = {"name": name_price[0].replace("**", "").strip()}
                try:
                    current["price"] = float(name_price[1].split(")")[0])
                except:
                    price_match = re.search(r"Price: ₹(\d+\.?\d*)", results)
                    if price_match:
                        current["price"] = float(price_match.group(1))
                    else:
                        logger.warning(f"Failed to parse price for product: {line}")
                        current["price"] = 0.0
            elif line.startswith("- Category:"):
                parts = line.split(", ")
                try:
                    current["category"] = parts[0].replace("- Category: ", "").strip()
                    current["fabric"] = parts[1].replace("Fabric: ", "").strip()
                except IndexError:
                    logger.warning(f"Failed to parse category/fabric: {line}")
                    current["category"] = ""
                    current["fabric"] = ""
            elif line.startswith("- [Product Link]("):
                try:
                    current["link"] = line.split("(")[1].split(")")[0]
                except IndexError:
                    logger.warning(f"Failed to parse link: {line}")
                    current["link"] = ""
            elif line.startswith("- ") and not line.startswith("- [Product Link]("):
                current["description"] = line[2:].strip()
        if current and current.get("name"):
            parsed.append(current)
        if not parsed:
            logger.warning(f"No products parsed from results: {results[:100]}...")
        return parsed
    except Exception as e:
        logger.error(f"Error parsing results: {e}")
        return []

# ==== Sanitize Dictionary for JSON Serialization ====
def sanitize_dict(d: Dict) -> Dict:
    """Convert non-serializable types (e.g., float32) to JSON-serializable types."""
    import numpy as np
    result = {}
    for key, value in d.items():
        if isinstance(value, np.floating):
            result[key] = float(value)
        elif isinstance(value, dict):
            result[key] = sanitize_dict(value)
        elif isinstance(value, list):
            result[key] = [sanitize_dict(item) if isinstance(item, dict) else item for item in value]
        else:
            result[key] = value
    return result

# ==== Rerank Results ====
def rerank_results(query: str, semantic: str, graph: str) -> List[Dict]:
    try:
        semantic_products = parse_results(semantic) if semantic and semantic.strip() and semantic != "I'm not sure based on the current data." else []
        graph_products = parse_results(graph) if graph and graph.strip() and graph != "😕 No products found matching your query." else []
        
        logger.info(f"Semantic RAG products: {len(semantic_products)} retrieved")
        logger.debug(f"Semantic products: {semantic_products}")
        logger.info(f"Graph RAG products: {len(graph_products)} retrieved")
        logger.debug(f"Graph products: {graph_products}")
        
        products = semantic_products + graph_products
        unique = {}
        for p in products:
            key = p.get("name", "") + p.get("link", "")
            if key and key not in unique:
                unique[key] = p
        product_list = list(unique.values())
        
        if not product_list:
            logger.warning("No products retrieved from either semantic or graph RAG")
            return []
        
        pairs = [(query, p.get("description", p.get("name", ""))) for p in product_list]
        scores = cross_encoder.predict(pairs)
        ranked = sorted(
            [{**p, "score": s} for p, s in zip(product_list, scores)],
            key=lambda x: x["score"],
            reverse=True
        )
        return ranked[:3]
    except Exception as e:
        logger.error(f"Error in reranking (semantic: {bool(semantic)}, graph: {bool(graph)}): {e}")
        return []

# ==== Retryable LLM Call ====
async def invoke_with_retry(prompt: str, max_retries: int = 3, delay: float = 2.0) -> Optional[str]:
    """Invoke LLM with retries on failure."""
    for attempt in range(max_retries):
        try:
            response = await llm.ainvoke(prompt)
            content = response.content.strip()
            if content.startswith("```json") and content.endswith("```"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()
            return content
        except Exception as e:
            logger.warning(f"LLM invoke failed (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                await asyncio.sleep(delay)
            continue
    logger.error(f"LLM invoke failed after {max_retries} attempts")
    return None

# ==== Evaluate Results ====
async def evaluate_results(query: str, results: List[Dict]) -> List[Dict]:
    if not results:
        return []
    prompt = """
    Evaluate each product for relevance to the query. Return a JSON array of objects, each with:
    - name: Product name (string)
    - relevance_score: Integer from 0 to 10
    - reason: Brief explanation of the score (string)
    Do not include Markdown, code blocks, or any text outside the JSON array. Return only the JSON array.
    Query: {query}
    Products: {products}
    Example: [
        {"name": "Product A", "relevance_score": 8, "reason": "Matches query well"},
        {"name": "Product B", "relevance_score": 5, "reason": "Partially relevant"}
    ]
    """
    try:
        sanitized_results = [sanitize_dict(r) for r in results]
        product_str = json.dumps(sanitized_results, indent=2)
        logger.debug(f"Evaluation input products: {product_str}")
        response = await invoke_with_retry(prompt.format(query=query, products=product_str))
        logger.debug(f"Raw evaluation response: {response}")
        if not response:
            logger.warning("Empty response from LLM in evaluate_results")
            return results
        try:
            evaluated = json.loads(response)
            if not isinstance(evaluated, list):
                logger.error(f"Invalid evaluation response: Expected list, got {type(evaluated)}")
                return results
            for r in evaluated:
                if not isinstance(r, dict) or not all(k in r for k in ["name", "relevance_score", "reason"]):
                    logger.error(f"Invalid evaluation item: {r}")
                    return results
            filtered = []
            for r in evaluated:
                if isinstance(r.get("relevance_score"), (int, float)) and r.get("relevance_score", 0) >= 6:
                    original = next((p for p in results if p.get("name") == r["name"]), None)
                    if original:
                        filtered.append({
                            **original,
                            "relevance_score": r["relevance_score"],
                            "relevance_reason": r["reason"]
                        })
            return filtered if filtered else results
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in evaluation response: {response} Error: {e}")
            return results
    except Exception as e:
        logger.error(f"Error in evaluation: {e}")
        return results

# ==== Generate Final Answer ====
async def generate_final_answer(query: str, emotion: str, results: List[Dict]) -> List[str]:
    if not results:
        max_price_match = re.search(r'under\s*₹?(\d+)|budget.*₹?(\d+)', query.lower())
        max_price = float(max_price_match.group(1) or max_price_match.group(2)) if max_price_match else None
        category_match = re.search(r'\b(jeans|jacket|shirt|tshirt|dress|shoes|coat|hat|bag|trousers|shorts|denim|pants|bottoms|sweater|bodysuit)\b', query.lower())
        category = category_match.group(1).capitalize() if category_match else "items"
        if category.lower() == "jeans":
            category = "Bottoms"  # Map jeans to Bottoms
        message = f"No {category} found in our collection"
        if max_price:
            message += f" under ₹{int(max_price)}"
        message += ". Try a different category or budget!"
        return [message]

    prompt = """
    Create a list of fashion recommendations based on the user's query and emotion, using a trendy, professional Gen Z tone with slang (slay, vibes, drip). Return a JSON array of strings, each describing one product with:
    - Product name and price (₹)
    - Why it's a vibe (description + style tips)
    - Category and fabric details
    - Link to purchase
    Do not include Markdown, code blocks, or any text outside the JSON array. Return only the JSON array.
    Query: {query}
    Emotion: {emotion}
    Products: {products}
    Example: [
        "Yo, the Summer Dress (₹500) is straight-up serving looks! This flowy cotton baddie screams beachy vibes and pairs perf with strappy sandals. Category: Dress, Fabric: Cotton. Cop it here: http://example.com/dress"
    ]
    """
    try:
        sanitized_results = [sanitize_dict(r) for r in results]
        product_str = json.dumps(sanitized_results, indent=2)
        logger.debug(f"Final answer input products: {product_str}")
        response = await invoke_with_retry(prompt.format(query=query, emotion=emotion, products=product_str))
        logger.debug(f"Raw final answer response: {response}")
        if not response:
            logger.warning("Empty response from LLM in generate_final_answer")
            return [f"No {sanitized_results[0]['category']} found matching your vibe. Try a different style or budget!"]
        try:
            final_answer = json.loads(response)
            if not isinstance(final_answer, list) or not all(isinstance(item, str) for item in final_answer):
                logger.error(f"Invalid final answer response: Expected list of strings, got {final_answer}")
                return [
                    f"Yo, {r['name']} (₹{r['price']}) is giving vibes! {r.get('description', 'Great pick!')} Style it up for that drip. Category: {r.get('category', 'Unknown')}, Fabric: {r.get('fabric', 'Unknown')}. Cop it here: {r.get('link', 'No link available')}"
                    for r in sanitized_results[:3]
                ]
            return final_answer
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in final answer response: {response} Error: {e}")
            return [
                f"Yo, {r['name']} (₹{r['price']}) is giving vibes! {r.get('description', 'Great pick!')} Style it up for that drip. Category: {r.get('category', 'Unknown')}, Fabric: {r.get('fabric', 'Unknown')}. Cop it here: {r.get('link', 'No link available')}"
                for r in sanitized_results[:3]
            ]
    except Exception as e:
        logger.error(f"Error in final answer generation: {e}")
        return [
            f"Yo, {r['name']} (₹{r['price']}) is giving vibes! {r.get('description', 'Great pick!')} Style it up for that drip. Category: {r.get('category', 'Unknown')}, Fabric: {r.get('fabric', 'Unknown')}. Cop it here: {r.get('link', 'No link available')}"
            for r in sanitized_results[:3]
        ]

# ==== Main Tool ====
@tool
async def recommendation_agent(query: str) -> List[str]:
    """
    Fashion Recommendation Agent: Given a user query, this agent enhances the query, retrieves and reranks product data using semantic and graph RAG, and returns a list of high-quality product recommendations in a Gen Z, professional fashion expert tone.
    """
    if not validate_query(query):
        return ["Yo, fam, we need more deets! Try mentioning a product like 'bottoms' or a budget like 'under ₹3000' for that fire drip!"]

    state = AgentState(query=query)

    # ==== Emotion detection ====
    state.emotion = await analyze_emotion(query)

    # ==== Enhanced query ====
    enhanced_query = await enhance_query(state.query, state.emotion)

    # ==== Generate cache key ====
    cache_key = hashlib.sha256((state.query + enhanced_query).encode()).hexdigest()
    state.cache_key = cache_key

    cached = cached_results(cache_key)
    if cached:
        state.recommendations = cached
    else:
        # ==== RAG Results ====
        semantic_result = await semantic_rag(enhanced_query)
        graph_results = await graphrag_response(enhanced_query)

        # ==== Rerank ====
        reranked = rerank_results(enhanced_query, semantic_result, graph_results)

        # ==== Evaluate ====
        evaluated = await evaluate_results(enhanced_query, reranked)

        state.recommendations = evaluated
        save_to_cache(cache_key, evaluated)

    # ==== Final Answer ====
    state.final_answer = await generate_final_answer(state.query, state.emotion, state.recommendations)
    return state.final_answer

# ==== Core Recommendation Function ====
async def get_recommendations(query: str) -> List[str]:
    """
    Core recommendation function that can be called directly without the @tool decorator.
    """
    if not validate_query(query):
        return ["Yo, fam, we need more deets! Try mentioning a product like 'bottoms' or a budget like 'under ₹3000' for that fire drip!"]

    state = AgentState(query=query)

    # ==== Emotion detection ====
    state.emotion = await analyze_emotion(query)

    # ==== Enhanced query ====
    enhanced_query = await enhance_query(state.query, state.emotion)

    # ==== Generate cache key ====
    cache_key = hashlib.sha256((state.query + enhanced_query).encode()).hexdigest()
    state.cache_key = cache_key

    cached = cached_results(cache_key)
    if cached:
        state.recommendations = cached
    else:
        # ==== RAG Results ====
        semantic_result = await semantic_rag(enhanced_query)
        graph_results = await graphrag_response(enhanced_query)

        # ==== Rerank ====
        reranked = rerank_results(enhanced_query, semantic_result, graph_results)

        # ==== Evaluate ====
        evaluated = await evaluate_results(enhanced_query, reranked)

        state.recommendations = evaluated
        save_to_cache(cache_key, evaluated)

    # ==== Final Answer ====
    state.final_answer = await generate_final_answer(state.query, state.emotion, state.recommendations)
    return state.final_answer

# ==== CLI Test Function ====
async def main():
    """
    CLI test function to simulate queries and test the recommendation agent.
    """
    print("Fashion Recommendation Agent CLI")
    print("Type 'exit' or 'quit' to stop.")
    while True:
        user_input = input("\nEnter your fashion query (e.g., 'jeans under 3000'): ").strip()
        if user_input.lower() in ('exit', 'quit'):
            print("Exiting CLI...")
            break
        print(f"\nProcessing query: {user_input}")
        try:
            recommendations = await get_recommendations(user_input)
            if recommendations:
                print("\nRecommendations:")
                for rec in recommendations:
                    print(f"- {rec}")
            else:
                print("No recommendations found. Try a more specific query!")
        except Exception as e:
            logger.error(f"Error processing query '{user_input}': {e}")
            print(f"Oops, something broke! Error: {e}")
        print("\n" + "="*50)

if __name__ == "__main__":
    asyncio.run(main())