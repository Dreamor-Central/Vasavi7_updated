#!/bin/bash

# Quick Start Script for Product Management System
# This script sets up and runs the entire system

set -e  # Exit on any error

echo "🚀 Starting Product Management System Setup..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    print_success "Docker is running"
}

# Check if required tools are installed
check_requirements() {
    print_status "Checking requirements..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed"
        exit 1
    fi
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        print_error "Node.js is not installed"
        exit 1
    fi
    
    # Check npm
    if ! command -v npm &> /dev/null; then
        print_error "npm is not installed"
        exit 1
    fi
    
    print_success "All requirements are met"
}

# Install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    pip install -r requirements.txt
    print_success "Python dependencies installed"
}

# Install Node.js dependencies
install_node_deps() {
    print_status "Installing Node.js dependencies..."
    cd client
    npm install
    cd ..
    print_success "Node.js dependencies installed"
}

# Start databases
start_databases() {
    print_status "Starting databases..."
    docker-compose -f docker-compose-db.yml up -d
    
    # Wait for databases to be ready
    print_status "Waiting for databases to be ready..."
    sleep 10
    
    # Check if databases are running
    if docker ps | grep -q "vasavi_postgresql" && docker ps | grep -q "vasavi_mongodb"; then
        print_success "Databases are running"
    else
        print_error "Failed to start databases"
        exit 1
    fi
}

# Create .env file if it doesn't exist
create_env_file() {
    if [ ! -f .env ]; then
        print_status "Creating .env file..."
        cat > .env << EOF
# Database Configuration
MONGODB_URI=mongodb://admin:password@localhost:27017
POSTGRES_URI=postgresql://user:password@localhost:5432/products

# API Configuration
HOST=0.0.0.0
PORT=8000
RELOAD=true

# OpenAI Configuration (for AI features)
OPENAI_API_KEY=your_openai_api_key_here
EOF
        print_success ".env file created"
        print_warning "OpenAI API key is optional. AI chat features will be disabled without it."
    else
        print_status ".env file already exists"
    fi
}

# Start the API server
start_api() {
    print_status "Starting Product Management API..."
    python run_product_api.py &
    API_PID=$!
    echo $API_PID > .api_pid
    
    # Wait for API to be ready
    sleep 5
    
    # Check if API is running
    if curl -s http://localhost:8000/health > /dev/null; then
        print_success "API is running on http://localhost:8000"
    else
        print_error "Failed to start API"
        exit 1
    fi
}

# Start the frontend
start_frontend() {
    print_status "Starting frontend..."
    cd client
    npm run dev &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > ../.frontend_pid
    cd ..
    
    # Wait for frontend to be ready
    sleep 10
    
    print_success "Frontend is running on http://localhost:3000"
}

# Test the system
test_system() {
    print_status "Testing system..."
    
    # Test API health
    if curl -s http://localhost:8000/health | grep -q "healthy"; then
        print_success "API health check passed"
    else
        print_warning "API health check failed"
    fi
    
    # Test frontend
    if curl -s http://localhost:3000 > /dev/null; then
        print_success "Frontend is accessible"
    else
        print_warning "Frontend health check failed"
    fi
}

# Show status
show_status() {
    echo ""
    echo "🎉 Product Management System is running!"
    echo ""
    echo "📡 API Server:     http://localhost:8000"
    echo "📚 API Docs:       http://localhost:8000/docs"
    echo "🔍 Health Check:   http://localhost:8000/health"
    echo ""
    echo "🌐 Frontend:       http://localhost:3000"
    echo "📦 Product Mgmt:   http://localhost:3000/products"
    echo ""
    echo "🗄️  Databases:"
    echo "   PostgreSQL:     localhost:5432"
    echo "   MongoDB:        localhost:27017"
    echo ""
    echo "📝 Logs:"
    echo "   API:            Check terminal output"
    echo "   Frontend:       Check browser console"
    echo "   Databases:      docker logs vasavi_postgresql"
    echo "                   docker logs vasavi_mongodb"
    echo ""
    echo "🛑 To stop the system, run: ./stop_system.sh"
    echo ""
}

# Main execution
main() {
    echo "=========================================="
    echo "  Product Management System Quick Start"
    echo "=========================================="
    echo ""
    
    check_docker
    check_requirements
    install_python_deps
    install_node_deps
    create_env_file
    start_databases
    start_api
    start_frontend
    test_system
    show_status
}

# Run main function
main "$@" 