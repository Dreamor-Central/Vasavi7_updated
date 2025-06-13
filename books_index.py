import os
import logging
from typing import List, Dict
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
INDEX_NAME = "salesman-index"
DIMENSION = 3072
EMBEDDING_MODEL = "text-embedding-3-large"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

def parse_sales_knowledge(file_path: str) -> List[Dict]:
    """Parses the sales knowledge file into a list of book dictionaries."""
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    books = []
    current_book = None
    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        if line.startswith(tuple(f"{i}. " for i in range(1, 10))):
            if current_book:
                books.append(current_book)
            title = line
            book_name = title.split("–")[0].strip()
            current_book = {"title": title, "name": book_name, "content": "", "tags": []}
        elif line.startswith("Behavior Tags:"):
            current_book["tags"] = [tag.strip() for tag in line.replace("Behavior Tags:", "").split(",")]
        elif current_book:
            current_book["content"] += line + "\n"
    if current_book:
        books.append(current_book)
    logging.info(f"Parsed {len(books)} books from {file_path}")
    return books

def initialize_pinecone():
    """Initialize Pinecone connection and return index."""
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        logging.info(f"Index {INDEX_NAME} created")
    else:
        logging.info(f"Index {INDEX_NAME} already exists")
    return pc.Index(INDEX_NAME)

def embed_and_upsert(index, file_path: str):
    """Splits text, embeds chunks, and upserts to Pinecone."""
    if not os.getenv("OPENAI_API_KEY"):
        logging.error("OPENAI_API_KEY environment variable not set")
        return
    
    try:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI Embeddings: {e}")
        return
    
    books = parse_sales_knowledge(file_path)
    if not books:
        logging.error("No books found. Exiting.")
        return
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    
    total_chunks = 0
    for i, book in enumerate(books):
        splits = text_splitter.split_text(book["content"])
        for j, split in enumerate(splits):
            text = f"{book['title']}: {split}"
            try:
                vector = embeddings.embed_documents([text])[0]  # Use embed_documents for consistency
                index.upsert(vectors=[(
                    f"{i}-{j}",
                    vector,
                    {
                        "book": book["name"],
                        "title": book["title"],
                        "content": split,
                        "tags": ",".join(book["tags"])
                    }
                )])
                total_chunks += 1
            except Exception as e:
                logging.error(f"Error embedding chunk {i}-{j}: {e}")
                continue
    logging.info(f"Upserted {total_chunks} chunks to Pinecone")

def run_indexing_pipeline(file_path: str):
    """Run the indexing pipeline."""
    if not os.getenv("PINECONE_API_KEY"):
        logging.error("PINECONE_API_KEY environment variable not set")
        return
    index = initialize_pinecone()
    embed_and_upsert(index, file_path)

if __name__ == "__main__":
    load_dotenv()
    run_indexing_pipeline("CORE SALES PSYCHOLOGY.txt")