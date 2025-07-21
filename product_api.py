from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any, List
import json
import io
import asyncio
from datetime import datetime
import logging
from product_manager import product_manager
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from enhanced_chat import enhanced_chat

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize LLM for chat queries (optional)
try:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    AI_ENABLED = True
except Exception as e:
    logger.warning(f"OpenAI API key not configured. AI chat features will be disabled. Error: {e}")
    llm = None
    AI_ENABLED = False

# Create FastAPI app
app = FastAPI(
    title="Product Management API",
    description="API for managing product data with intelligent storage routing",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chat prompt template for product queries
CHAT_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful product assistant for Vasavi clothing brand. You have access to product data and can help users find products, get recommendations, and answer questions about the catalog.

Available product categories: Jacket, Shirt, T-Shirt, Hoodie, Corset, Bottoms

User Query: {query}

Product Data: {product_data}

Please provide a helpful response that:
1. Answers the user's question
2. Mentions relevant products if applicable
3. Provides styling suggestions when appropriate
4. Uses a friendly, casual tone with emojis

Response format:
{{
    "response": "Your helpful response here",
    "products_mentioned": ["product1", "product2"],
    "category": "relevant_category",
    "confidence": 0.95
}}
""")

@app.on_event("startup")
async def startup_event():
    """Initialize database tables on startup"""
    try:
        await product_manager.create_tables()
        logger.info("Product management system initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize product management system: {e}")

@app.post("/upload/csv")
async def upload_csv(file: UploadFile = File(...)):
    """Upload and process CSV file with product data"""
    try:
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV")
        
        result = await product_manager.process_csv_upload(file)
        return {
            "success": True,
            "message": result["message"],
            "upload_id": result["upload_id"],
            "storage_method": result["storage_method"],
            "row_count": result["row_count"],
            "file_name": result["file_name"]
        }
    except Exception as e:
        logger.error(f"CSV upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process PDF file with product data"""
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="File must be a PDF")
        
        result = await product_manager.process_pdf_upload(file)
        return {
            "success": True,
            "message": result["message"],
            "upload_id": result["upload_id"],
            "storage_method": result["storage_method"],
            "row_count": result["row_count"],
            "file_name": result["file_name"],
            "image_urls_found": result["image_urls_found"]
        }
    except Exception as e:
        logger.error(f"PDF upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
async def search_products(
    query: str = Query(..., description="Search query"),
    category: Optional[str] = Query(None, description="Filter by category"),
    min_price: Optional[float] = Query(None, description="Minimum price"),
    max_price: Optional[float] = Query(None, description="Maximum price"),
    limit: int = Query(50, description="Maximum number of results")
):
    """Search products across both databases"""
    try:
        filters = {}
        if category:
            filters["category"] = category
        if min_price is not None:
            filters["min_price"] = min_price
        if max_price is not None:
            filters["max_price"] = max_price
        
        results = await product_manager.search_products(query, filters)
        
        # Limit results
        results = results[:limit]
        
        return {
            "success": True,
            "query": query,
            "filters": filters,
            "results_count": len(results),
            "products": results
        }
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat")
async def chat_query(
    query: str = Query(..., description="Chat query about products"),
    limit: int = Query(10, description="Number of products to consider")
):
    """Chat-based product query with AI assistance"""
    try:
        if not AI_ENABLED:
            return {
                "success": False,
                "error": "AI features are disabled. Please configure OPENAI_API_KEY to enable chat functionality.",
                "suggestion": "You can still use the /search endpoint for product queries."
            }
        
        # Search for relevant products
        products = await product_manager.search_products(query, limit=limit)
        
        # Prepare product data for LLM
        product_data = []
        for product in products:
            product_data.append({
                "name": product.get("style_name", ""),
                "category": product.get("category", ""),
                "price": product.get("price", 0),
                "description": product.get("description", ""),
                "fabric": product.get("fabric_description", "")
            })
        
        # Generate AI response
        parser = JsonOutputParser()
        
        chain = CHAT_PROMPT | llm | parser
        
        response = await chain.ainvoke({
            "query": query,
            "product_data": json.dumps(product_data, indent=2)
        })
        
        return {
            "success": True,
            "query": query,
            "ai_response": response,
            "products_considered": len(product_data),
            "relevant_products": product_data[:5]  # Top 5 most relevant
        }
        
    except Exception as e:
        logger.error(f"Chat query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def generate_chat_stream_response(query: str, checkpoint_id: Optional[str] = None):
    """Generate enhanced streaming chat response with context awareness"""
    try:
        # Use the enhanced chat system
        response = await enhanced_chat.generate_response(query, checkpoint_id)
        
        # Stream the response character by character for a natural effect
        for char in response:
            yield f"data: {json.dumps({'type': 'success', 'content': char})}\n\n"
            await asyncio.sleep(0.01)  # Small delay for streaming effect
        
        yield f"data: {json.dumps({'type': 'end'})}\n\n"
        
    except Exception as e:
        logger.error(f"Enhanced streaming chat failed: {e}")

        
    except Exception as e:
        logger.error(f"Streaming chat failed: {e}")
        yield f"data: {json.dumps({'type': 'error', 'content': 'System error. Please try again!'})}\n\n"

@app.post("/chat/stream")
async def chat_stream(
    message: str = Query(..., description="User query"),
    checkpoint_id: Optional[str] = Query(None, description="Optional checkpoint ID for session persistence")
):
    """Streaming chat endpoint for real-time responses"""
    logger.info(f"Streaming chat request: message='{message}', checkpoint_id='{checkpoint_id}'")
    return StreamingResponse(
        generate_chat_stream_response(message, checkpoint_id),
        media_type="text/event-stream"
    )

@app.get("/export/csv")
async def export_csv(
    category: Optional[str] = Query(None, description="Filter by category"),
    min_price: Optional[float] = Query(None, description="Minimum price"),
    max_price: Optional[float] = Query(None, description="Maximum price")
):
    """Export products as CSV"""
    try:
        filters = {}
        if category:
            filters["category"] = category
        if min_price is not None:
            filters["min_price"] = min_price
        if max_price is not None:
            filters["max_price"] = max_price
        
        csv_content = await product_manager.export_products("csv", filters)
        
        return StreamingResponse(
            io.StringIO(csv_content),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=products_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"}
        )
    except Exception as e:
        logger.error(f"CSV export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/export/json")
async def export_json(
    category: Optional[str] = Query(None, description="Filter by category"),
    min_price: Optional[float] = Query(None, description="Minimum price"),
    max_price: Optional[float] = Query(None, description="Maximum price")
):
    """Export products as JSON"""
    try:
        filters = {}
        if category:
            filters["category"] = category
        if min_price is not None:
            filters["min_price"] = min_price
        if max_price is not None:
            filters["max_price"] = max_price
        
        json_content = await product_manager.export_products("json", filters)
        
        return JSONResponse(
            content=json.loads(json_content),
            headers={"Content-Disposition": f"attachment; filename=products_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"}
        )
    except Exception as e:
        logger.error(f"JSON export failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/statistics")
async def get_statistics():
    """Get product statistics"""
    try:
        stats = await product_manager.get_product_statistics()
        return {
            "success": True,
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Statistics retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/categories")
async def get_categories():
    """Get all available product categories"""
    try:
        stats = await product_manager.get_product_statistics()
        categories = list(stats.get("categories", {}).keys())
        
        return {
            "success": True,
            "categories": categories,
            "category_counts": stats.get("categories", {})
        }
    except Exception as e:
        logger.error(f"Categories retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test database connections
        stats = await product_manager.get_product_statistics()
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "total_products": stats.get("total_products", 0)
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Product Management API",
        "version": "1.0.0",
        "endpoints": {
            "upload_csv": "/upload/csv",
            "upload_pdf": "/upload/pdf",
            "search": "/search",
            "chat": "/chat",
            "export_csv": "/export/csv",
            "export_json": "/export/json",
            "statistics": "/statistics",
            "categories": "/categories",
            "health": "/health"
        }
    } 