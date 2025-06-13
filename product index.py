import os
import logging
from typing import List, Tuple, Dict
import pandas as pd
import sqlite3
from openai import OpenAI
import pinecone
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class ProductIndexer:
    """Handles indexing of product data into SQLite FTS5 and Pinecone."""

    def __init__(self):
        """Initialize OpenAI client, Pinecone, and configurations."""
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Pinecone configuration
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.pinecone_index_name = "vasavi"
        self.dimension = 3072  # ✅ match Pinecone index
        self.pinecone_env = "us-east-1-aws"
        self.batch_size = 100

        # Initialize Pinecone (Serverless mode)
        try:
            pinecone.init(api_key=self.pinecone_api_key, environment=self.pinecone_env)
            self.index = pinecone.Index(self.pinecone_index_name)
            logger.info("Connected to Pinecone index: vasavi")
        except Exception as e:
            logger.error(f"Failed to connect to Pinecone: {e}")
            raise

    def load_data(self, csv_path: str) -> pd.DataFrame:
        """Load and preprocess CSV data."""
        try:
            df = pd.read_csv(csv_path)
            df['search_text'] = df.apply(self._create_search_text, axis=1)
            logger.info(f"Loaded {len(df)} products from {csv_path}")
            return df
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            raise

    def _create_search_text(self, row: pd.Series) -> str:
        """Concatenate fields for semantic embedding."""
        return " | ".join(map(str, [
            row.get('Category', ''),
            row.get('Style Name', ''),
            row.get('Description', ''),
            row.get('Fabric Description', '')
        ]))

    def setup_sqlite_fts(self, db_path: str, df: pd.DataFrame) -> sqlite3.Connection:
        """Create FTS5-enabled SQLite database for keyword search."""
        try:
            conn = sqlite3.connect(db_path)
            c = conn.cursor()

            c.execute("PRAGMA journal_mode=WAL;")
            c.execute("PRAGMA synchronous=NORMAL;")
            c.execute("PRAGMA cache_size=-20000;")

            c.execute("DROP TABLE IF EXISTS products_fts")
            c.execute("""
                CREATE VIRTUAL TABLE products_fts USING fts5(
                    style_name,
                    category,
                    description,
                    fabric_description,
                    price UNINDEXED,
                    product_link UNINDEXED,
                    content='',
                    tokenize='porter'
                );
            """)

            for _, row in df.iterrows():
                c.execute("""
                    INSERT INTO products_fts(style_name, category, description, fabric_description, price, product_link)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    row.get('Style Name', ''),
                    row.get('Category', ''),
                    row.get('Description', ''),
                    row.get('Fabric Description', ''),
                    row.get('Price', ''),
                    row.get('Product Link', '')
                ))

            conn.commit()
            logger.info("SQLite FTS5 table created and populated")
            return conn
        except Exception as e:
            logger.error(f"SQLite FTS setup failed: {e}")
            raise

    def backup_database(self, conn: sqlite3.Connection, backup_path: str):
        """Backup the SQLite database."""
        try:
            backup_conn = sqlite3.connect(backup_path)
            conn.backup(backup_conn)
            backup_conn.close()
            logger.info(f"SQLite DB backed up to {backup_path}")
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            raise

    def get_embedding(self, text: str) -> List[float]:
        """Generate a 1536-dim embedding twice and concatenate to 3072-dim."""
        try:
            # Call embedding API twice and concatenate to match 3072-dim
            response1 = self.openai_client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            emb1 = response1.data[0].embedding

            response2 = self.openai_client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            emb2 = response2.data[0].embedding

            return emb1 + emb2  # => 3072 dimension
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise

    def upsert_to_pinecone(self, df: pd.DataFrame):
        """Push product embeddings to Pinecone."""
        try:
            vectors = []
            for idx, row in df.iterrows():
                embedding = self.get_embedding(row['search_text'])
                vectors.append((str(idx), embedding, {
                    "style_name": row['Style Name'],
                    "category": row['Category'],
                    "description": row['Description'],
                    "fabric_description": row['Fabric Description'],
                    "price": row['Price'],
                    "product_link": row['Product Link']
                }))

            for i in range(0, len(vectors), self.batch_size):
                batch = vectors[i:i + self.batch_size]
                self.index.upsert(vectors=batch)
                logger.info(f"Upserted batch {i // self.batch_size + 1} to Pinecone")

            logger.info("All vectors upserted to Pinecone")
        except Exception as e:
            logger.error(f"Pinecone upsert failed: {e}")
            raise

    def index(self, csv_path: str, db_path: str, backup_path: str):
        """Full indexing pipeline."""
        try:
            df = self.load_data(csv_path)

            if not os.path.exists(db_path):
                conn = self.setup_sqlite_fts(db_path, df)
                self.backup_database(conn, backup_path)
                conn.close()
            else:
                logger.info("SQLite DB already exists. Skipping rebuild.")

            self.upsert_to_pinecone(df)

            logger.info("Indexing process completed.")
        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            raise

if __name__ == "__main__":
    indexer = ProductIndexer()
    indexer.index(
        csv_path="vasavi2.csv",
        db_path="products_fts.db",
        backup_path="products_fts_backup.db"
    )
