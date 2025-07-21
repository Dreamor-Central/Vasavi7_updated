#!/bin/bash

# Stop Script for Product Management System
# This script cleanly shuts down the entire system

echo "🛑 Stopping Product Management System..."

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

# Stop API server
stop_api() {
    print_status "Stopping API server..."
    if [ -f .api_pid ]; then
        API_PID=$(cat .api_pid)
        if kill -0 $API_PID 2>/dev/null; then
            kill $API_PID
            print_success "API server stopped"
        else
            print_warning "API server was not running"
        fi
        rm -f .api_pid
    else
        print_warning "No API PID file found"
    fi
}

# Stop frontend
stop_frontend() {
    print_status "Stopping frontend..."
    if [ -f .frontend_pid ]; then
        FRONTEND_PID=$(cat .frontend_pid)
        if kill -0 $FRONTEND_PID 2>/dev/null; then
            kill $FRONTEND_PID
            print_success "Frontend stopped"
        else
            print_warning "Frontend was not running"
        fi
        rm -f .frontend_pid
    else
        print_warning "No frontend PID file found"
    fi
}

# Stop databases
stop_databases() {
    print_status "Stopping databases..."
    docker-compose -f docker-compose-db.yml down
    print_success "Databases stopped"
}

# Clean up any remaining processes
cleanup_processes() {
    print_status "Cleaning up processes..."
    
    # Kill any remaining Python processes running our API
    pkill -f "run_product_api.py" 2>/dev/null || true
    
    # Kill any remaining Node processes running our frontend
    pkill -f "next dev" 2>/dev/null || true
    
    print_success "Process cleanup completed"
}

# Show final status
show_final_status() {
    echo ""
    echo "✅ Product Management System stopped successfully!"
    echo ""
    echo "📝 Summary:"
    echo "   API Server:     Stopped"
    echo "   Frontend:       Stopped"
    echo "   Databases:      Stopped"
    echo "   Processes:      Cleaned up"
    echo ""
    echo "🚀 To restart the system, run: ./quick_start.sh"
    echo ""
}

# Main execution
main() {
    echo "=========================================="
    echo "  Product Management System Stop"
    echo "=========================================="
    echo ""
    
    stop_api
    stop_frontend
    stop_databases
    cleanup_processes
    show_final_status
}

# Run main function
main "$@" 