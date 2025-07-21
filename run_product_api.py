#!/usr/bin/env python3
"""
Product Management API Server
Runs the FastAPI server for product data management
"""

import uvicorn
import os
import asyncio
from product_api import app
from product_manager import product_manager

async def init_databases():
    """Initialize database connections and tables"""
    try:
        await product_manager.create_tables()
        print("Database tables created successfully")
    except Exception as e:
        print(f" Database initialization warning: {e}")
        print("Continuing with server startup...")

def main():
    """Main function to start the server"""
    print("🚀 Starting Product Management API Server...")
    
    # Initialize databases
    asyncio.run(init_databases())
    
    # Get configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    
    print(f"📡 Server will run on http://{host}:{port}")
    print(f"🔄 Auto-reload: {reload}")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("🔍 Health Check: http://localhost:8000/health")
    
    # Start server
    uvicorn.run(
        "product_api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    main() 