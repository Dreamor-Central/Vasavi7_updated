import os
import logging
import asyncio
import json
import re
from typing import List, Dict, Optional
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pinecone import Pinecone
from sentence_transformers import CrossEncoder
from fuzzywuzzy import process, fuzz
import argparse

# ==== Setup Logging ====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("semanticrag.log"), logging.StreamHandler()]
)
logger = logging.getLogger("VasaviRAG")

# ==== Load Environment Variables ====
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not OPENAI_API_KEY or not PINECONE_API_KEY:
    logger.error("Missing required environment variables: OPENAI_API_KEY or PINECONE_API_KEY")
    raise ValueError("Environment variables not set")

# ==== Clients ====
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
index = pinecone_client.Index("vasavi3")

# ==== Cross Encoder Model for Re-ranking ====
cross_encoder = CrossEncoder("BAAI/bge-reranker-base")

# ==== Pinecone Category Mapping ====
PINECONE_CATEGORY_MAPPING = {
    'tshirt': 'T-Shirt', 't-shirt': 'T-Shirt', 'tee': 'T-Shirt', 'tees': 'T-Shirt', 't': 'T-Shirt', 'casual top': 'T-Shirt',
    'shirt': 'Shirt', 'shirts': 'Shirt', 'button-up': 'Shirt', 'button-down': 'Shirt', 'flannel': 'Shirt', 'dress shirt': 'Shirt', 'top': 'Shirt',
    'hoodie': 'Hoodie', 'hoodies': 'Hoodie', 'sweatshirt': 'Hoodie', 'pullover': 'Hoodie', 'fleece': 'Hoodie',
    'jacket': 'Jacket', 'jackets': 'Jacket', 'coat': 'Jacket', 'blazer': 'Jacket', 'outerwear': 'Jacket', 'parka': 'Jacket', 'windbreaker': 'Jacket', 'bomber': 'Jacket',
    'bottoms': 'Bottoms', 'jeans': 'Bottoms', 'pants': 'Bottoms', 'trousers': 'Bottoms', 'shorts': 'Bottoms', 'denim': 'Bottoms', 'chinos': 'Bottoms', 'cargos': 'Bottoms', 'cargo pants': 'Bottoms', 'leggings': 'Bottoms', 'skirt': 'Bottoms', 'culottes': 'Bottoms',
    'corset': 'Corset', 'corsets': 'Corset', 'bustier': 'Corset', 'bodice': 'Corset',
    'bodysuit': 'Bodysuit', 'bodysuits': 'Bodysuit',
    'clothes': 'clothing', 'clothing': 'clothing', 'recommendation': 'clothing', 'apparel': 'clothing', 'outfit': 'clothing', 'fashion': 'clothing', 'wear': 'clothing'
}

# ==== Known Fashion Filters ====
KNOWN_CATEGORIES = list(set(PINECONE_CATEGORY_MAPPING.keys()))
KNOWN_MATERIALS = [
    'denim', 'cotton', 'leather', 'wool', 'suede', 'fleece', 'spandex', 'mesh', 'poly', 'polyester', 'polysuiting', 'pu leather', 'vegan leather',
    'jersey', 'flannel', 'linen', 'twill', 'satin', 'lycra', 'modal', 'bemberg', 'terry'
]
CLOTHING_KEYWORDS = KNOWN_CATEGORIES + KNOWN_MATERIALS + ['clothing', 'wear', 'outfit', 'style', 'fashion', 'garment', 'apparel']

# ==== Fuzzy Matcher ====
def fuzzy_match(term: str, choices: List[str], threshold: int = 80) -> Optional[str]:
    if not choices:
        return None
    result, score = process.extractOne(term, choices, scorer=fuzz.token_sort_ratio)
    return result if score >= threshold else None

# ==== Extract Filters (Category, Material, Price) ====
async def extract_filters(query: str) -> Dict:
    query_lower = query.lower()
    filters = {}

    # Price extraction
    price_patterns = [
        r'budget\s*(?:is|:)?\s*₹?(\d+)', r'price\s*(?:is|:)?\s*₹?(\d+)', r'under\s*₹?(\d+)',
        r'below\s*₹?(\d+)', r'less than\s*₹?(\d+)', r'max(?:imum)?\s*₹?(\d+)', r'₹(\d+)', r'rs\.?\s*(\d+)'
    ]
    for pattern in price_patterns:
        if match := re.search(pattern, query_lower):
            filters['max_price'] = float(match.group(1))
            break

    # Category extraction
    found_category = None
    for word in query_lower.split():
        if word in PINECONE_CATEGORY_MAPPING:
            found_category = PINECONE_CATEGORY_MAPPING[word]
            break
    if not found_category:
        for word in query_lower.split():
            stripped_word = word.strip('s')
            if fuzzy_match(stripped_word, list(PINECONE_CATEGORY_MAPPING.keys()), threshold=80):
                for k, v in PINECONE_CATEGORY_MAPPING.items():
                    if fuzzy_match(stripped_word, [k], threshold=80):
                        found_category = v
                        break
            if found_category:
                break
    if found_category:
        filters['category'] = found_category

    # Material extraction
    found_material = None
    if "jeans" in query_lower or "denim" in query_lower:
        found_material = "denim"
    else:
        for word in query_lower.split():
            if match := fuzzy_match(word, KNOWN_MATERIALS, threshold=80):
                found_material = match
                break
    if found_material:
        filters['material'] = found_material

    # Contextual fixes
    if 'denim' in filters.get('material', '') and ('jacket' in query_lower or 'coat' in query_lower) and not filters.get('category'):
        filters['category'] = 'Jacket'
    if 'leather' in filters.get('material', '') and ('jacket' in query_lower or 'coat' in query_lower) and not filters.get('category'):
        filters['category'] = 'Jacket'
    if 'bodysuit' in query_lower and not filters.get('category'):
        filters['category'] = 'Bodysuit'
    if 'corset' in query_lower and not filters.get('category'):
        filters['category'] = 'Corset'

    logger.info(f"Extracted filters: {filters}")
    return filters

# ==== Rewrite Query for Semantic Search ====
async def rewrite_query(query: str) -> str:
    for attempt in range(3):
        try:
            response = await openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a fashion search expert. Rewrite the query to be short and semantically rich, "
                            "including standard category, material, and price if mentioned. Emphasize 'denim' for 'jeans' queries. "
                            "Prioritize keywords: shirts, tshirts, jackets, clothes, bottoms, corset, bodysuit, jeans. "
                            "For general queries like 'clothes' or 'recommendation', use broad terms like "
                            "'vasavi clothing, new arrivals, fashion items'. Examples: "
                            "'Show me jeans under 3000' -> 'denim jeans, blue pants, under 3000';"
                            "'Show me jackets' -> 'jacket, outerwear, vasavi jackets';"
                            "'Recommend shirts' -> 'shirt, casual shirt, vasavi shirts';"
                            "'Tshirts under 2000' -> 't-shirt, casual t-shirt, under 2000';"
                            "'Clothes' -> 'vasavi clothing, new arrivals, fashion items';"
                            "'A black item' -> 'black clothing, dark apparel, black fashion';"
                        )
                    },
                    {"role": "user", "content": f"Rewrite this query: {query}"}
                ],
                temperature=0.3
            )
            rewritten = response.choices[0].message.content.strip()
            logger.info(f"Rewritten query: {rewritten}")
            return rewritten
        except Exception as e:
            logger.warning(f"Query rewrite failed on attempt {attempt + 1}: {e}")
            if attempt == 2:
                logger.error(f"Query rewrite failed after 3 attempts. Using original query.")
                return query

# ==== Embedding Function ====
async def get_embedding(text: str) -> List[float]:
    for attempt in range(3):
        try:
            response = await openai_client.embeddings.create(
                input=[text],
                model="text-embedding-3-large"
            )
            return response.data[0].embedding
        except Exception as e:
            logger.warning(f"Embedding error on attempt {attempt + 1}: {e}")
            if attempt == 2:
                logger.error(f"Embedding failed after 3 attempts: {e}")
                return []

# ==== Main Semantic RAG Entry Point ====
async def semantic_rag(query: str, category: Optional[str] = None) -> List[Dict]:
    logger.info(f"Starting RAG process for query: '{query}', Category: {category}")

    # Step 1: Extract filters
    filters = await extract_filters(query)
    if category:  # Override category if provided by recommendation_agent_tool
        filters['category'] = category
    logger.info(f"Final filters: {filters}")

    # Check if this is a non-fashion query
    is_fashion_query = bool(filters.get('category') or filters.get('material') or filters.get('max_price')) or \
                       any(fuzzy_match(word, CLOTHING_KEYWORDS, 75) for word in query.lower().split())

    if not is_fashion_query:
        logger.info("Non-fashion query detected, falling back to general handler.")
        return [{"response": await handle_general_query(query), "intent": "general"}]

    # Step 2: Rewrite query
    rewritten_query = await rewrite_query(query)

    # Step 3: Get embeddings
    dense_vector = await get_embedding(rewritten_query)
    if not dense_vector:
        logger.error("Failed to generate embedding for the query.")
        return []

    # Step 4: Query Pinecone
    pinecone_top_k = 50
    pinecone_filter = {}
    if filters.get('category') and filters['category'] != 'clothing':
        pinecone_filter["category"] = filters['category']
    if 'jeans' in query.lower() and 'category' not in filters:
        pinecone_filter["category"] = "Bottoms"
        logger.info("Query contains 'jeans', setting Pinecone filter category to 'Bottoms'.")

    logger.info(f"Querying Pinecone with top_k={pinecone_top_k}, filter: {pinecone_filter or 'None'}")
    results = []
    for attempt in range(2):
        try:
            pinecone_results = index.query(
                vector=dense_vector,
                top_k=pinecone_top_k,
                include_metadata=True,
                filter=pinecone_filter if pinecone_filter and attempt == 0 else None
            )
            matches = pinecone_results.get('matches', [])
            logger.info(f"Pinecone attempt {attempt + 1}: {len(matches)} matches found")

            if matches:
                logger.info(f"Top 5 matches (style_name, category, score):")
                for m in matches[:5]:
                    logger.info(f"  - {m['metadata'].get('style_name', 'N/A')} ({m['metadata'].get('category', 'N/A')}) - Score: {m['score']:.4f}")

            # Process matches
            for m in matches:
                metadata = m.get('metadata', {})
                if metadata.get('style_name') and metadata.get('category'):
                    results.append({
                        "style_name": metadata.get('style_name'),
                        "category": metadata.get('category'),
                        "price": metadata.get('price', 'N/A'),
                        "fabric": metadata.get('fabric', 'N/A'),
                        "description": metadata.get('description', 'No description'),
                        "product_link": metadata.get('product_link', '#')
                    })

            if results or attempt == 1:
                break
        except Exception as e:
            logger.warning(f"Pinecone query failed on attempt {attempt + 1}: {e}")
            if attempt == 1:
                logger.error(f"Pinecone query failed after 2 attempts: {e}")
                return []

    if not results:
        logger.info("No matches found after Pinecone queries.")
        return []

    # Step 5: Re-rank using cross encoder
    rerank_inputs = [
        (rewritten_query, f"{m['category']} | {m['description']} | {m['fabric']}")
        for m in results
    ]
    scores = cross_encoder.predict(rerank_inputs)
    for i, score in enumerate(scores):
        results[i]['cross_encoder_score'] = float(score)

    # Step 6: Apply Soft Filtering/Boosting
    logger.info("Applying soft filters/boosting...")
    for m in results:
        boost = 0.0
        if 'category' in filters and filters['category'] != 'clothing':
            if m['category'].lower() == filters['category'].lower():
                boost += 0.5
            elif fuzzy_match(filters['category'].lower(), [m['category'].lower()], threshold=75):
                boost += 0.2
        if 'material' in filters:
            metadata_fabric = m['fabric'].lower()
            if filters['material'].lower() in metadata_fabric:
                boost += 0.4
            elif fuzzy_match(filters['material'].lower(), [metadata_fabric], threshold=75):
                boost += 0.1
        if 'max_price' in filters:
            try:
                item_price = float(m['price'])
                if item_price > filters['max_price']:
                    m['cross_encoder_score'] *= 0.5
                elif item_price <= filters['max_price']:
                    boost += 0.1
            except (ValueError, TypeError):
                logger.warning(f"Invalid price for {m['style_name']}: {m['price']}")
                m['cross_encoder_score'] *= 0.8

        m['cross_encoder_score'] += boost

    # Sort after boosting
    results = sorted(results, key=lambda x: x['cross_encoder_score'], reverse=True)

    logger.info(f"Top 5 results after re-ranking (style_name, category, CE score):")
    for m in results[:5]:
        logger.info(f"  - {m['style_name']} ({m['category']}) - CE Score: {m['cross_encoder_score']:.4f}")

    return results[:10]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Vasavi RAG system with a given query.")
    parser.add_argument("query", type=str, nargs='+', help="The fashion query to process (e.g., 'show me a denim jacket').")
    args = parser.parse_args()
    user_query = " ".join(args.query)

    print(f"\n--- Initiating Vasavi RAG for query: '{user_query}' ---\n")
    answer = asyncio.run(semantic_rag(user_query))
    print("\n--- Final RAG Answer ---")
    print(json.dumps(answer, indent=2))
    print("\n--------------------------\n")