import pandas as pd
import json
import logging
import os
import uuid
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import asyncio
from pathlib import Path
import re
from urllib.parse import urlparse
import fitz
import io
from fastapi import UploadFile, HTTPException
import motor.motor_asyncio
import asyncpg
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, String, Integer, Float, Text, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import JSONB
import psycopg2
from psycopg2.extras import RealDictCursor


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


Base = declarative_base()

class Product(Base):
    __tablename__ = "products"
    
    id = Column(String, primary_key=True)
    product_id = Column(String, index=True)
    category = Column(String, index=True)
    style_name = Column(String)
    description = Column(Text)
    fabric_description = Column(Text)
    price = Column(Float)
    image_url = Column(String)
    product_link = Column(String)
    product_metadata = Column(JSONB)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)

class ProductManager:
    def __init__(self):
        self.mongo_client = None
        self.postgres_engine = None
        self.postgres_session = None
        self.mongo_db = None
        self.threshold_rows = 20000
        
        # Initialize databases
        self._init_databases()
    
    def _init_databases(self):
        """Initialize database connections"""
        try:
            # MongoDB connection
            mongo_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
            self.mongo_client = motor.motor_asyncio.AsyncIOMotorClient(mongo_uri)
            self.mongo_db = self.mongo_client.product_database
            
            # PostgreSQL connection
            postgres_uri = os.getenv("POSTGRES_URI", "sqlite+aiosqlite:///./test_products.db")
            self.postgres_engine = create_async_engine(postgres_uri)
            
            logger.info("Database connections initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize databases: {e}")
            raise
    
    async def create_tables(self):
        """Create necessary database tables"""
        try:
            async with self.postgres_engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise
    
    def _extract_image_urls_from_pdf(self, pdf_content: bytes) -> List[str]:
        """Extract image URLs from PDF content"""
        image_urls = []
        try:
            pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                
                # Extract text and look for URLs
                text = page.get_text()
                url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+'
                urls = re.findall(url_pattern, text)
                
                # Filter for image URLs
                for url in urls:
                    if any(ext in url.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']):
                        image_urls.append(url)
                
                # Extract images from PDF
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = pdf_document.extract_image(xref)
                        image_bytes = base_image["image"]
                        
                        # For now, we'll store image data as base64 or save to file system
                        # In production, you might want to upload to cloud storage
                        image_urls.append(f"extracted_image_{page_num}_{img_index}")
                    except Exception as e:
                        logger.warning(f"Failed to extract image {img_index} from page {page_num}: {e}")
            
            pdf_document.close()
            return list(set(image_urls))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Failed to extract image URLs from PDF: {e}")
            return []
    
    def _normalize_product_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize product data to standard format"""
        # Generate product_id if not present
        product_id = data.get("product_id", data.get("Product ID", ""))
        if not product_id:
            # Generate from style name or create unique ID
            style_name = data.get("style_name", data.get("Style Name", data.get("name", "")))
            if style_name:
                product_id = f"VAS_{style_name.replace(' ', '_').upper()}_{uuid.uuid4().hex[:6]}"
            else:
                product_id = f"VAS_{uuid.uuid4().hex[:8]}"
        
        normalized = {
            "id": str(uuid.uuid4()),
            "product_id": product_id,
            "category": data.get("category", data.get("Category", "")),
            "style_name": data.get("style_name", data.get("Style Name", data.get("name", ""))),
            "description": data.get("description", data.get("Description", "")),
            "fabric_description": data.get("fabric_description", data.get("Fabric Description", "")),
            "price": self._parse_price(data.get("price", data.get("Price", 0))),
            "image_url": data.get("image_url", data.get("Image URL", "")),  # For actual image URLs
            "product_link": data.get("product_link", data.get("Product Link", "")),  # For product pages
            "product_metadata": {
                "original_data": data,
                "processed_at": datetime.utcnow().isoformat()
            }
        }
        return normalized
    
    def _parse_price(self, price_value: Any) -> float:
        """Parse price value to float"""
        if isinstance(price_value, (int, float)):
            return float(price_value)
        
        if isinstance(price_value, str):
            # Remove currency symbols and commas
            cleaned = re.sub(r'[^\d.]', '', price_value)
            try:
                return float(cleaned) if cleaned else 0.0
            except ValueError:
                return 0.0
        
        return 0.0
    
    async def process_csv_upload(self, file: UploadFile) -> Dict[str, Any]:
        """Process CSV file upload"""
        try:
            # Read CSV file
            content = await file.read()
            df = pd.read_csv(io.BytesIO(content))
            
            # Convert to list of dictionaries
            products_data = df.to_dict('records')
            row_count = len(products_data)
            
            # Normalize data
            normalized_products = []
            for product_data in products_data:
                normalized = self._normalize_product_data(product_data)
                normalized_products.append(normalized)
            
            # Determine storage method based on row count
            if row_count > self.threshold_rows:
                storage_method = "mongodb"
                result = await self._store_in_mongodb(normalized_products)
            else:
                storage_method = "postgresql"
                result = await self._store_in_postgresql(normalized_products)
            
            return {
                "success": True,
                "message": f"Successfully processed {row_count} products",
                "storage_method": storage_method,
                "row_count": row_count,
                "file_name": file.filename,
                "upload_id": str(uuid.uuid4()),
                "result": result
            }
            
        except Exception as e:
            logger.error(f"Failed to process CSV upload: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to process CSV: {str(e)}")
    
    async def process_pdf_upload(self, file: UploadFile) -> Dict[str, Any]:
        """Process PDF file upload"""
        try:
            # Read PDF file
            content = await file.read()
            
            # Extract image URLs from PDF
            image_urls = self._extract_image_urls_from_pdf(content)
            
            # For PDFs, we'll need to extract tabular data
            # This is a simplified approach - in production you might want more sophisticated PDF parsing
            pdf_document = fitz.open(stream=content, filetype="pdf")
            
            extracted_data = []
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                text = page.get_text()
                
                # Simple text extraction - in production, use more sophisticated parsing
                # This is a placeholder for actual PDF table extraction
                lines = text.split('\n')
                for line in lines:
                    line = line.strip()
                    if line:  # Add any non-empty line
                        extracted_data.append({"raw_text": line, "page": page_num})
            
            pdf_document.close()
            
            # For now, we'll create a basic product entry from PDF
            # In production, you'd want more sophisticated PDF parsing
            if extracted_data:
                # Try to extract meaningful information from the first few lines
                first_lines = [item["raw_text"] for item in extracted_data[:5]]
                description = " | ".join(first_lines[:3])  # Use first 3 lines as description
                
                product_data = {
                    "product_id": f"pdf_{uuid.uuid4().hex[:8]}",
                    "category": "PDF_Extracted",
                    "style_name": f"PDF Product from {file.filename}",
                    "description": description[:500],  # Limit description length
                    "fabric_description": "PDF extracted content",
                    "price": 0.0,
                    "image_url": image_urls[0] if image_urls else "",
                    "product_link": "",
                    "metadata": {
                        "source": "pdf",
                        "file_name": file.filename,
                        "extracted_data": extracted_data,
                        "image_urls": image_urls
                    }
                }
                
                normalized_product = self._normalize_product_data(product_data)
                
                # Store in PostgreSQL (PDFs typically have fewer entries)
                result = await self._store_in_postgresql([normalized_product])
                
                return {
                    "success": True,
                    "message": f"Successfully processed PDF with {len(extracted_data)} data points",
                    "storage_method": "postgresql",
                    "row_count": 1,
                    "file_name": file.filename,
                    "upload_id": str(uuid.uuid4()),
                    "image_urls_found": len(image_urls),
                    "result": result
                }
            else:
                raise HTTPException(status_code=400, detail="No product data found in PDF")
                
        except Exception as e:
            logger.error(f"Failed to process PDF upload: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to process PDF: {str(e)}")
    
    async def _store_in_mongodb(self, products: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Store products in MongoDB"""
        try:
            collection = self.mongo_db.products
            
            # Insert products
            result = await collection.insert_many(products)
            
            return {
                "inserted_count": len(result.inserted_ids),
                "database": "mongodb",
                "collection": "products"
            }
            
        except Exception as e:
            logger.error(f"Failed to store in MongoDB: {e}")
            raise
    
    async def _store_in_postgresql(self, products: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Store products in PostgreSQL"""
        try:
            async with self.postgres_engine.begin() as conn:
                inserted_count = 0
                for product in products:
                    # Convert metadata to JSON string for SQLite compatibility
                    metadata = product.get('product_metadata', {})
                    if isinstance(metadata, dict):
                        metadata = json.dumps(metadata)
                    
                    # Prepare product data with JSON string metadata
                    product_data = {
                        'id': product.get('id'),
                        'product_id': product.get('product_id'),
                        'category': product.get('category'),
                        'style_name': product.get('style_name'),
                        'description': product.get('description'),
                        'fabric_description': product.get('fabric_description'),
                        'price': product.get('price'),
                        'image_url': product.get('image_url'),
                        'product_link': product.get('product_link'),
                        'product_metadata': metadata
                    }
                    
                    # Insert product
                    await conn.execute(
                        text("""
                            INSERT INTO products (id, product_id, category, style_name, description, 
                                                 fabric_description, price, image_url, product_link, product_metadata)
                            VALUES (:id, :product_id, :category, :style_name, :description,
                                    :fabric_description, :price, :image_url, :product_link, :product_metadata)
                            ON CONFLICT (id) DO UPDATE SET
                                updated_at = EXCLUDED.updated_at,
                                product_metadata = EXCLUDED.product_metadata
                        """),
                        product_data
                    )
                    inserted_count += 1
                
                return {
                    "inserted_count": inserted_count,
                    "database": "postgresql",
                    "table": "products"
                }
                
        except Exception as e:
            logger.error(f"Failed to store in PostgreSQL: {e}")
            raise
    
    async def search_products(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search products across both databases"""
        try:
            results = []
            
            # Search in PostgreSQL
            postgres_results = await self._search_postgresql(query, filters)
            results.extend(postgres_results)
            
            # Search in MongoDB
            mongo_results = await self._search_mongodb(query, filters)
            results.extend(mongo_results)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search products: {e}")
            raise
    
    async def _search_postgresql(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search products in PostgreSQL"""
        try:
            async with self.postgres_engine.begin() as conn:
                sql_query = """
                    SELECT * FROM products 
                    WHERE is_active = true 
                    AND (
                        LOWER(style_name) LIKE LOWER(:query) OR
                        LOWER(description) LIKE LOWER(:query) OR
                        LOWER(category) LIKE LOWER(:query) OR
                        LOWER(fabric_description) LIKE LOWER(:query)
                    )
                """
                
                params = {"query": f"%{query}%"}
                
                # Add filters
                if filters:
                    if filters.get("category"):
                        sql_query += " AND LOWER(category) = LOWER(:category)"
                        params["category"] = filters["category"]
                    
                    if filters.get("min_price"):
                        sql_query += " AND price >= :min_price"
                        params["min_price"] = filters["min_price"]
                    
                    if filters.get("max_price"):
                        sql_query += " AND price <= :max_price"
                        params["max_price"] = filters["max_price"]
                
                sql_query += " ORDER BY created_at DESC LIMIT 100"
                
                result = await conn.execute(text(sql_query), params)
                rows = result.fetchall()
                
                return [dict(row._mapping) for row in rows]
                
        except Exception as e:
            logger.error(f"Failed to search PostgreSQL: {e}")
            return []
    
    async def _search_mongodb(self, query: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search products in MongoDB"""
        try:
            collection = self.mongo_db.products
            
            # Build search query
            search_query = {
                "$or": [
                    {"style_name": {"$regex": query, "$options": "i"}},
                    {"description": {"$regex": query, "$options": "i"}},
                    {"category": {"$regex": query, "$options": "i"}},
                    {"fabric_description": {"$regex": query, "$options": "i"}}
                ]
            }
            
            # Add filters
            if filters:
                if filters.get("category"):
                    search_query["category"] = {"$regex": filters["category"], "$options": "i"}
                
                if filters.get("min_price") or filters.get("max_price"):
                    price_query = {}
                    if filters.get("min_price"):
                        price_query["$gte"] = filters["min_price"]
                    if filters.get("max_price"):
                        price_query["$lte"] = filters["max_price"]
                    search_query["price"] = price_query
            
            cursor = collection.find(search_query).limit(100)
            results = await cursor.to_list(length=100)
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to search MongoDB: {e}")
            return []
    
    async def export_products(self, format: str = "csv", filters: Optional[Dict[str, Any]] = None) -> Union[str, bytes]:
        """Export products in specified format"""
        try:
            # Get all products
            products = await self.search_products("", filters)
            
            if format.lower() == "csv":
                df = pd.DataFrame(products)
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                return csv_buffer.getvalue()
            
            elif format.lower() == "json":
                return json.dumps(products, indent=2, default=str)
            
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Failed to export products: {e}")
            raise
    
    async def get_product_statistics(self) -> Dict[str, Any]:
        """Get product statistics"""
        try:
            stats = {
                "total_products": 0,
                "categories": {},
                "price_range": {"min": 0, "max": 0, "avg": 0},
                "storage_distribution": {"postgresql": 0, "mongodb": 0}
            }
            
            # Get PostgreSQL stats
            async with self.postgres_engine.begin() as conn:
                result = await conn.execute(text("SELECT COUNT(*) FROM products WHERE is_active = true"))
                postgres_count = result.scalar()
                stats["storage_distribution"]["postgresql"] = postgres_count
                stats["total_products"] += postgres_count
                
                # Get category distribution
                result = await conn.execute(text("""
                    SELECT category, COUNT(*) as count 
                    FROM products 
                    WHERE is_active = true 
                    GROUP BY category
                """))
                categories = result.fetchall()
                for category in categories:
                    stats["categories"][category[0]] = category[1]
                
                # Get price stats
                result = await conn.execute(text("""
                    SELECT MIN(price), MAX(price), AVG(price) 
                    FROM products 
                    WHERE is_active = true
                """))
                price_stats = result.fetchone()
                if price_stats:
                    stats["price_range"]["min"] = price_stats[0] or 0
                    stats["price_range"]["max"] = price_stats[1] or 0
                    stats["price_range"]["avg"] = price_stats[2] or 0
            
            # Get MongoDB stats
            collection = self.mongo_db.products
            mongo_count = await collection.count_documents({})
            stats["storage_distribution"]["mongodb"] = mongo_count
            stats["total_products"] += mongo_count
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get product statistics: {e}")
            raise

# Global instance
product_manager = ProductManager() 