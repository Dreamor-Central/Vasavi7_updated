import os
import logging
import asyncio
import re
from typing import List, Dict, Optional
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pinecone import Pinecone
from sentence_transformers import CrossEncoder
from semanticrag import semantic_rag
from fuzzywuzzy import process, fuzz
import argparse

# ==== Setup Logging ====
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VasaviRAG")

# ==== Load Environment Variables ====
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# ==== Clients ====
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
pinecone_client = Pinecone(api_key=PINECONE_API_KEY)  # Corrected to api_key
index = pinecone_client.Index("vasavi3")

# ==== Cross Encoder Model for Re-ranking ====
cross_encoder = CrossEncoder("BAAI/bge-reranker-base")

# ==== Pinecone Category Mapping ====
PINECONE_CATEGORY_MAPPING = {
    'tshirt': 'T-Shirt',
    't-shirt': 'T-Shirt',
    'tee': 'T-Shirt',
    'tees': 'T-Shirt',
    't': 'T-Shirt',
    'casual top': 'T-Shirt',
    'shirt': 'Shirt',
    'shirts': 'Shirt',
    'button-up': 'Shirt',
    'button-down': 'Shirt',
    'flannel': 'Shirt',
    'dress shirt': 'Shirt',
    'top': 'Shirt',
    'hoodie': 'Hoodie',
    'hoodies': 'Hoodie',
    'sweatshirt': 'Hoodie',
    'pullover': 'Hoodie',
    'fleece': 'Hoodie',
    'jacket': 'Jacket',
    'jackets': 'Jacket',
    'coat': 'Jacket',
    'blazer': 'Jacket',
    'outerwear': 'Jacket',
    'parka': 'Jacket',
    'windbreaker': 'Jacket',
    'bomber': 'Jacket',
    'bottoms': 'Bottoms',
    'jeans': 'Bottoms',
    'pants': 'Bottoms',
    'trousers': 'Bottoms',
    'shorts': 'Bottoms',
    'denim': 'Bottoms',
    'chinos': 'Bottoms',
    'cargos': 'Bottoms',
    'cargo pants': 'Bottoms',
    'leggings': 'Bottoms',
    'skirt': 'Bottoms',
    'culottes': 'Bottoms',
    'corset': 'Corset',
    'corsets': 'Corset',
    'bustier': 'Corset',
    'bodice': 'Corset',
    'clothes': 'clothing',
    'clothing': 'clothing',
    'recommendation': 'clothing',
    'apparel': 'clothing',
    'outfit': 'clothing',
    'fashion': 'clothing',
    'wear': 'clothing',
    'bodysuit': 'bodysuit'
}

# ==== Known Fashion Filters ====
KNOWN_CATEGORIES = list(set(PINECONE_CATEGORY_MAPPING.keys()))
KNOWN_MATERIALS = [
    'denim', 'cotton', 'leather', 'wool', 'suede', 'fleece', 'spandex',
    'mesh', 'poly', 'polyester', 'polysuiting', 'pu leather', 'vegan leather',
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

    # Category extraction using PINECONE_CATEGORY_MAPPING
    found_category = None
    for word in query_lower.split():
        word = word.strip('s')  # Handle plural forms
        if word in PINECONE_CATEGORY_MAPPING:
            found_category = PINECONE_CATEGORY_MAPPING[word]
            break
        if match := fuzzy_match(word, KNOWN_CATEGORIES, threshold=80):
            found_category = PINECONE_CATEGORY_MAPPING.get(match, match)
            break
    if found_category:
        filters['category'] = found_category

    # Material extraction with jeans implying denim
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

    # Contextual category fixes
    if 'denim' in filters.get('material', '') and ('jacket' in query_lower or 'coat' in query_lower) and not filters.get('category'):
        filters['category'] = 'Jacket'
    if 'leather' in filters.get('material', '') and ('jacket' in query_lower or 'coat' in query_lower) and not filters.get('category'):
        filters['category'] = 'Jacket'
    if 'bodysuit' in query_lower and not filters.get('category'):
        filters['category'] = PINECONE_CATEGORY_MAPPING.get('bodysuit', 'bodysuit')
    if 'corset' in query_lower and not filters.get('category'):
        filters['category'] = PINECONE_CATEGORY_MAPPING.get('corset', 'Corset')

    logger.info(f"Extracted filters: {filters}")
    return filters

# ==== Rewrite Query for Semantic Search ====
async def rewrite_query(query: str) -> str:
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a fashion search expert. Rewrite the query to be short and semantically rich, "
                        "including standard category, material, and price if mentioned. Emphasize 'denim' for 'jeans' queries. "
                        "Prioritize keywords: shirts, tshirts, jackets, clothes, bottoms, corset, recommendation, jeans. "
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
        logger.warning(f"Query rewrite failed: {e}. Using original query.")
        return query

# ==== Embedding Function ====
async def get_embedding(text: str) -> List[float]:
    try:
        response = await openai_client.embeddings.create(
            input=[text],
            model="text-embedding-3-large"
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        return []

# ==== Format Output Prompt ====
def build_prompt(matches: List[Dict], query: str) -> str:
    system_prompt = """
You are an expert assistant for Vasavi clothing, specializing in shirts, t-shirts, hoodies, jackets, bottoms, and corsets. Recommend items ONLY from Vasavi data. Format:
**Product Name (₹Price)**
- Category: ..., Fabric: ...
- Description
- [Product Link](URL)

For queries like 'jeans', consider items with denim fabric or similar pants (e.g., cargo pants, chinos) as close matches. If no exact match, show closest options based on category, material, and price. If none are relevant, reply:
"I'm not sure based on the current Vasavi data."
"""

    if not matches:
        return f"{system_prompt}\n\nQuestion: {query}\nAnswer: I'm not sure based on the current Vasavi data."

    context = "\n\n".join([
        f"**{m['metadata'].get('style_name', 'N/A')} (₹{m['metadata'].get('price', 'N/A')})**\n"
        f"- Category: {m['metadata'].get('category', 'N/A')}, Fabric: {m['metadata'].get('fabric', 'N/A')}\n"
        f"- {m['metadata'].get('description', 'N/A')}\n"
        f"- [Product Link]({m['metadata'].get('product_link', '#')})"
        for m in matches
    ])
    return f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"

# ==== GPT Final Answer ====
async def generate_answer(prompt: str) -> str:
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Follow the given format and use only Vasavi product context. Be flexible with near-matches for materials like denim for jeans."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Answer generation error: {e}")
        return "I'm not sure based on the current Vasavi data."

# ==== GPT Fallback for General Queries ====
async def handle_general_query(query: str) -> str:
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You're a helpful assistant. Answer clearly and concisely. If it's about fashion, try to provide a general fashion-related response or ask for clarification."},
                {"role": "user", "content": query}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Fallback GPT error: {e}")
        return "Sorry, I couldn’t understand that."

# ==== Main Semantic RAG Entry Point ====
async def semantic_rag(query: str) -> str:
    logger.info(f"Starting RAG process for query: '{query}'")

    # Step 1: Extract filters
    filters = await extract_filters(query)

    # Check if this is a non-fashion query
    is_fashion_query = bool(filters.get('category') or filters.get('material') or filters.get('max_price')) or \
                       any(fuzzy_match(word, CLOTHING_KEYWORDS, 75) for word in query.lower().split())
    if not is_fashion_query:
        logger.info("Non-fashion query detected, falling back to general handler.")
        return await handle_general_query(query)
    elif not filters.get('category') and any(word in query_lower for word in ['clothes', 'recommendation', 'apparel', 'fashion']):
        logger.info("General clothing query detected, broadening search.")
        filters['category'] = 'clothing'

    # Step 2: Rewrite query
    rewritten_query = await rewrite_query(query)

    # Step 3: Get embeddings
    dense_vector = await get_embedding(rewritten_query)
    if not dense_vector:
        logger.error("Failed to generate embedding for the query.")
        return "Sorry, I couldn't process your request."

    # Step 4: Query Pinecone with increased top_k
    pinecone_top_k = 50

    # Apply Pinecone filtering selectively
    pinecone_filter = {}
    if 'category' in filters and filters['category'] != 'clothing' and 'jeans' not in query.lower():
        pinecone_category = filters['category']
        if pinecone_category in ['T-Shirt', 'Shirt', 'Jacket', 'Bottoms', 'Corset', 'Hoodie']:
            pinecone_filter["category"] = pinecone_category
        else:
            logger.info(f"Skipping Pinecone filter for non-standard category: {pinecone_category}")

    logger.info(f"Querying Pinecone with top_k={pinecone_top_k} and filter: {pinecone_filter}")
    results = index.query(vector=dense_vector, top_k=pinecone_top_k, include_metadata=True, filter=pinecone_filter if pinecone_filter else None)
    matches = results.matches

    logger.info(f"Pinecone initial retrieval: {len(matches)} matches found (top {pinecone_top_k}).")
    if matches:
        logger.info(f"Top 5 initial matches (style_name, category, score):")
        for m in matches[:5]:
            logger.info(f"  - {m['metadata'].get('style_name', 'N/A')} ({m['metadata'].get('category', 'N/A')}) - Score: {m['score']:.4f}")
    else:
        logger.info("No initial matches from Pinecone.")
        return await generate_answer(build_prompt([], query))

    # Step 5: Re-rank using cross encoder
    rerank_inputs = [
        (rewritten_query, f"{m['metadata'].get('category', '')} | {m['metadata'].get('description', '')} | {m['metadata'].get('fabric', '')}")
        for m in matches
    ]
    scores = cross_encoder.predict(rerank_inputs)
    for i, score in enumerate(scores):
        matches[i]['cross_encoder_score'] = float(score)

    # Step 6: Apply Soft Filtering/Boosting
    logger.info("Applying soft filters/boosting...")
    for m in matches:
        boost = 0.0
        if 'category' in filters and filters['category'] != 'clothing':
            if m['metadata'].get('category', '').lower() == filters['category'].lower():
                boost += 0.5
            elif fuzzy_match(filters['category'], [m['metadata'].get('category', '').lower()], threshold=75):
                boost += 0.2

        if 'material' in filters:
            metadata_fabric = m['metadata'].get('fabric', '').lower()
            if filters['material'].lower() in metadata_fabric:
                boost += 0.4
            elif fuzzy_match(filters['material'], [metadata_fabric], threshold=75):
                boost += 0.1

        if 'max_price' in filters:
            try:
                item_price = float(m['metadata'].get('price', 0))
                if item_price > filters['max_price']:
                    m['cross_encoder_score'] *= 0.5
                elif item_price <= filters['max_price']:
                    boost += 0.1
            except ValueError:
                logger.warning(f"Invalid price for {m['metadata'].get('style_name', 'N/A')}: {m['metadata'].get('price', 'N/A')}")
                m['cross_encoder_score'] *= 0.8

        m['cross_encoder_score'] += boost

    # Sort after boosting
    matches = sorted(matches, key=lambda x: x['cross_encoder_score'], reverse=True)

    logger.info(f"Matches after boosting and re-sorting (top 5, CE Score):")
    for m in matches[:5]:
        logger.info(f"  - {m['metadata'].get('style_name', 'N/A')} ({m['metadata'].get('category', 'N/A')}) - CE Score: {m['cross_encoder_score']:.4f} (Original Pinecone Score: {m['score']:.4f})")

    # Step 7: Prepare final prompt
    num_results_for_llm = 10
    final_results_for_llm = matches[:num_results_for_llm]

    logger.info(f"Passing {len(final_results_for_llm)} items to LLM for final answer generation.")
    prompt = build_prompt(final_results_for_llm, query)

    # Step 8: Generate final answer
    return await generate_answer(prompt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Vasavi RAG system with a given query.")
    parser.add_argument(
        "query",
        type=str,
        nargs='+',
        help="The fashion query to process (e.g., 'show me a denim jacket')."
    )

    args = parser.parse_args()
    user_query = " ".join(args.query)

    print(f"\n--- Initiating Vasavi RAG for query: '{user_query}' ---\n")
    answer = asyncio.run(semantic_rag(user_query))
    print("\n--- Final RAG Answer ---")
    print(answer)
    print("\n--------------------------\n")