# Vasavi GenZ - AI Fashion Chat System

An intelligent fashion product chatbot with CSV/PDF upload, RAG search, and context-aware conversations.

## 🚀 Quick Start

### Backend Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables (create .env file)
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
DATABASE_URL=postgresql://user:pass@localhost/dbname
MONGODB_URL=mongodb://localhost:27017

# Run backend (choose one method)
python run_product_api.py          # Method 1: Direct FastAPI
python run_uvicorn.py              # Method 2: Uvicorn with reload
uvicorn product_api:app --reload   # Method 3: Direct uvicorn
```

### Frontend Setup
```bash
# Navigate to client directory
cd client

# Install dependencies
npm install

# Run frontend
npm run dev
```

## 🌐 Access Points
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 🔧 System Architecture

### Core Components
- **FastAPI Backend**: Product management, file upload, AI chat
- **Next.js Frontend**: Chat interface, file upload UI
- **PostgreSQL**: Primary product database
- **MongoDB**: Secondary storage (fallback)
- **OpenAI GPT-4**: AI chat responses
- **Pinecone**: Vector search (RAG)

### Key Features
- **CSV/PDF Upload**: Bulk product import
- **Intelligent Search**: RAG-powered product discovery
- **Context-Aware Chat**: Session-based conversation memory
- **Real-time Streaming**: Live chat responses
- **Product Management**: Search, export, statistics

### Data Flow
1. **Upload**: CSV/PDF → Parse → Store in PostgreSQL
2. **Search**: Query → RAG + Database → Ranked results
3. **Chat**: User input → Context analysis → AI response
4. **Follow-up**: Session memory → Product matching → Details

## 📁 Project Structure
```
Vasavi7_updated/
├── product_manager.py      # Database operations
├── enhanced_chat.py        # AI chat system
├── product_api.py          # FastAPI endpoints
├── run_product_api.py      # Server runner (Method 1)
├── run_uvicorn.py          # Uvicorn runner (Method 2)
├── requirements.txt        # Python dependencies
├── client/                 # Next.js frontend
│   ├── src/
│   │   ├── app/           # Pages and routing
│   │   └── components/    # React components
│   └── package.json
└── README.md
```

## 🎯 Usage Examples

### Upload Products
- Go to http://localhost:3000/products
- Upload CSV/PDF files
- Products stored in shared database

### Chat Interface
- Go to http://localhost:3000
- Ask: "Show me cotton shirts"
- Follow-up: "Tell me more about the second one"

### API Endpoints
- `GET /search?query=jacket` - Product search
- `POST /upload/csv` - Upload CSV
- `POST /chat/stream` - AI chat
- `GET /statistics` - Product stats

## 🔑 Environment Variables
```env
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
DATABASE_URL=postgresql://...
MONGODB_URL=mongodb://...
```

## 🚀 Production Deployment
- Use production database (PostgreSQL)
- Set up proper environment variables
- Configure CORS for frontend-backend communication
- Enable HTTPS for security 