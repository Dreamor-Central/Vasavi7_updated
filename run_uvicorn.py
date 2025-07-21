#!/usr/bin/env python3
"""
Uvicorn runner for Vasavi GenZ Product API
"""

import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

if __name__ == "__main__":
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    
    print(f"🚀 Starting Vasavi GenZ API Server...")
    print(f"📍 Host: {host}")
    print(f"🔌 Port: {port}")
    print(f"🔄 Reload: {reload}")
    print(f"🌐 Access: http://localhost:{port}")
    print(f"📚 API Docs: http://localhost:{port}/docs")
    print("-" * 50)
    
    # Start server
    uvicorn.run(
        "product_api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    ) 