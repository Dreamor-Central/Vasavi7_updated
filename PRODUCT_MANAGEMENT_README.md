# Product Management System

A comprehensive product data management system for Vasavi clothing brand with intelligent storage routing, AI-powered chat queries, and modern web interface.

## 🚀 Features

### 📁 File Upload & Processing
- **CSV Upload**: Handles product data with automatic storage routing
  - ≤20K rows → PostgreSQL (SQL database)
  - >20K rows → MongoDB (NoSQL database)
- **PDF Upload**: Extracts product data and image URLs from PDF files
- **Image URL Extraction**: Automatically detects and stores image URLs from PDFs

### 🔍 Search & Query
- **Cross-Database Search**: Searches across both PostgreSQL and MongoDB
- **Advanced Filtering**: By category, price range, and custom queries
- **AI-Powered Chat**: Natural language queries with product recommendations

### 📊 Export & Analytics
- **CSV Export**: Download filtered product data as CSV
- **JSON Export**: Download filtered product data as JSON
- **Statistics Dashboard**: Real-time product analytics and storage distribution

### 🤖 AI Integration
- **Chat-Based Queries**: Ask questions about products in natural language
- **Product Recommendations**: AI-powered styling and product suggestions
- **Smart Categorization**: Automatic product categorization and tagging

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   FastAPI       │    │   Databases     │
│   (Next.js)     │◄──►│   Backend       │◄──►│   PostgreSQL    │
│                 │    │                 │    │   MongoDB       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📋 Prerequisites

- Python 3.8+
- Node.js 18+
- Docker & Docker Compose
- PostgreSQL (via Docker)
- MongoDB (via Docker)

## 🛠️ Installation

### 1. Clone and Setup

```bash
cd Vasavi7_updated
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Frontend Dependencies

```bash
cd client
npm install
```

### 4. Start Databases

```bash
# Start MongoDB and PostgreSQL
docker-compose -f docker-compose-db.yml up -d
```

### 5. Environment Variables

Create a `.env` file in the root directory:

```env
# Database Configuration
MONGODB_URI=mongodb://admin:password@localhost:27017
POSTGRES_URI=postgresql://user:password@localhost:5432/products

# API Configuration
HOST=0.0.0.0
PORT=8000
RELOAD=true

# OpenAI Configuration (for AI features)
OPENAI_API_KEY=your_openai_api_key_here
```

## 🚀 Running the Application

### 1. Start the Product Management API

```bash
# From the root directory
python run_product_api.py
```

The API will be available at:
- **API**: http://localhost:8000
- **Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### 2. Start the Frontend

```bash
cd client
npm run dev
```

The frontend will be available at:
- **Main App**: http://localhost:3000
- **Product Management**: http://localhost:3000/products

## 📖 API Endpoints

### File Upload
- `POST /upload/csv` - Upload CSV file with product data
- `POST /upload/pdf` - Upload PDF file with product data

### Search & Query
- `GET /search` - Search products with filters
- `GET /chat` - AI-powered chat queries

### Export
- `GET /export/csv` - Export products as CSV
- `GET /export/json` - Export products as JSON

### Analytics
- `GET /statistics` - Get product statistics
- `GET /categories` - Get available categories
- `GET /health` - Health check

## 📊 Data Schema

### Product Model
```json
{
  "id": "uuid",
  "product_id": "string",
  "category": "string",
  "style_name": "string",
  "description": "text",
  "fabric_description": "text",
  "price": "float",
  "image_url": "string",
  "product_link": "string",
  "metadata": "jsonb",
  "created_at": "datetime",
  "updated_at": "datetime",
  "is_active": "boolean"
}
```

### CSV Format
Expected CSV columns:
- `Category`, `Style Name`, `Description`, `Fabric Description`, `Price`, `Product Link`
- Additional columns will be stored in metadata

## 🔧 Configuration

### Storage Threshold
The system automatically routes data based on row count:
- **Threshold**: 20,000 rows (configurable in `product_manager.py`)
- **≤20K rows**: PostgreSQL (better for structured queries)
- **>20K rows**: MongoDB (better for large datasets)

### Database Connections
- **PostgreSQL**: Structured product data with ACID compliance
- **MongoDB**: Large datasets with flexible schema

## 🤖 AI Features

### Chat Queries
The AI assistant can help with:
- Product recommendations
- Styling advice
- Category-specific queries
- Price range suggestions
- Trend analysis

### Example Queries
- "Show me jackets under ₹5000"
- "What's trending in streetwear?"
- "Recommend a casual outfit"
- "Find products with leather fabric"

## 📱 Frontend Features

### Upload Interface
- Drag-and-drop file upload
- Real-time upload progress
- File validation and error handling
- Storage method indication

### Search Interface
- Advanced filtering options
- Real-time search results
- Product cards with images
- Price and category filters

### Chat Interface
- Natural language queries
- AI response streaming
- Product recommendations
- Confidence scoring

### Analytics Dashboard
- Real-time statistics
- Storage distribution
- Category breakdown
- Price range analysis

## 🔍 Troubleshooting

### Common Issues

1. **Database Connection Errors**
   ```bash
   # Check if databases are running
   docker ps
   
   # Restart databases
   docker-compose -f docker-compose-db.yml restart
   ```

2. **Port Conflicts**
   ```bash
   # Check port usage
   lsof -i :8000
   lsof -i :3000
   ```

3. **Missing Dependencies**
   ```bash
   # Reinstall Python dependencies
   pip install -r requirements.txt
   
   # Reinstall Node dependencies
   cd client && npm install
   ```

### Logs
- **API Logs**: Check console output from `run_product_api.py`
- **Frontend Logs**: Check browser console and terminal
- **Database Logs**: `docker logs vasavi_postgresql` or `docker logs vasavi_mongodb`

## 🧪 Testing

### API Testing
```bash
# Test health endpoint
curl http://localhost:8000/health

# Test search endpoint
curl "http://localhost:8000/search?query=jacket&limit=5"

# Test statistics endpoint
curl http://localhost:8000/statistics
```

### File Upload Testing
```bash
# Test CSV upload
curl -X POST -F "file=@vasavi2.csv" http://localhost:8000/upload/csv

# Test PDF upload
curl -X POST -F "file=@sample.pdf" http://localhost:8000/upload/pdf
```

## 🔒 Security Considerations

- Database credentials should be stored securely
- API endpoints should be protected in production
- File uploads should be validated and sanitized
- CORS should be configured for production domains

## 📈 Performance Optimization

- Database indexing on frequently queried fields
- Connection pooling for database connections
- Caching for frequently accessed data
- Pagination for large result sets

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is part of the Vasavi clothing brand system.

## 🆘 Support

For support and questions:
- Check the troubleshooting section
- Review API documentation at `/docs`
- Check logs for error details
- Contact the development team

---

**Happy Product Managing! 🎉** 