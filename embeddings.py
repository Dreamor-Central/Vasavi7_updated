# ==== Imports ====
import os
import uuid
import logging
import asyncio
import pandas as pd
import csv
import nltk
from dotenv import load_dotenv
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# ==== Environment Setup ====
load_dotenv()
nltk.download('punkt', quiet=True)

# ==== Logging ====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==== API Keys ====
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY or PINECONE_API_KEY in environment variables")

# ==== Pinecone Setup ====
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "vasavi3"
dimension = 3072
batch_size = 100

# ==== OpenAI Client ====
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# ==== Local Embeddings (for fallback or hybrid) ====
sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

# ==== Generate Embedding (Batch) ====
async def generate_embeddings_batch(texts: list, retries: int = 3, delay: int = 5) -> list:
    for attempt in range(retries):
        try:
            res = await openai_client.embeddings.create(
                input=texts,
                model="text-embedding-3-large"
            )
            return [r.embedding for r in res.data]
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(delay * (2 ** attempt))  # Exponential backoff
    logger.error(f"Batch embedding generation failed after {retries} attempts")
    return [[] for _ in texts]

# ==== Populate Pinecone Index ====
async def populate_index(df: pd.DataFrame, index_name: str):
    # Check existing indexes
    existing_indexes = pc.list_indexes().names()
    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        logger.info(f"Created Pinecone index: {index_name}")
    
    index = pc.Index(index_name)
    stats = index.describe_index_stats()
    if stats['total_vector_count'] > 0:
        logger.info(f"Index {index_name} already contains {stats['total_vector_count']} vectors")

    # Combine fields for richer embeddings
    descriptions = [
        f"{row['Category']}: {row['Style Name']} - {row['Description']} - {row['Fabric Description']}"
        for _, row in df.iterrows()
    ]
    logger.info(f"Generating embeddings for {len(descriptions)} descriptions")

    # Generate embeddings in batches
    embeddings = await generate_embeddings_batch(descriptions)
    logger.info(f"Generated {len(embeddings)} embeddings")

    vectors = []
    for i, row in df.iterrows():
        if not embeddings[i]:  # Skip failed embeddings
            logger.warning(f"Skipping product {row['Style Name']} due to empty embedding")
            continue
        meta = {
            "style_name": str(row["Style Name"]) or "Unknown",
            "category": str(row["Category"]) or "Unknown",
            "fabric": str(row["Fabric Description"]) or "Unknown",
            "price": float(row["Price"]) if pd.notna(row["Price"]) else 0.0,
            "product_link": str(row["Product Link"]) or "",
            "description": str(row["Description"]) or ""
        }
        vectors.append({
            "id": str(uuid.uuid4()),
            "values": embeddings[i],
            "metadata": meta
        })

    # Upsert in batches
    for i in range(0, len(vectors), batch_size):
        try:
            index.upsert(vectors=vectors[i:i + batch_size])
            logger.info(f"Upserted batch {i // batch_size + 1} with {len(vectors[i:i + batch_size])} vectors")
        except Exception as e:
            logger.error(f"Upsert error for batch {i // batch_size + 1}: {e}")

    logger.info("✅ All vectors upserted to Pinecone.")

# ==== Main ====
async def main():
    try:
        df = pd.read_csv("vasavi3.csv", quoting=csv.QUOTE_NONNUMERIC, encoding="utf-8").dropna()
        df["Price"] = df["Price"].astype(float)
        await populate_index(df, index_name)
    except FileNotFoundError:
        logger.error("Error: vasavi3.csv file not found")
    except Exception as e:
        logger.error(f"Main execution failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())