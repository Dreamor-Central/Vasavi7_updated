#!/usr/bin/env python3
"""
Integration script to connect Product Management System with existing Vasavi app
"""

import asyncio
import logging
from product_manager import product_manager
from semanticrag import semantic_rag
from recommendation import get_recommendations

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductSystemIntegration:
    """Integrates product management system with existing Vasavi functionality"""
    
    def __init__(self):
        self.product_manager = product_manager
    
    async def initialize(self):
        """Initialize the integration"""
        try:
            # Create database tables
            await self.product_manager.create_tables()
            logger.info("✅ Product management system integrated successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize product system integration: {e}")
            return False
    
    async def enhanced_semantic_search(self, query: str, category: str = None):
        """Enhanced semantic search using both existing and new product data"""
        try:
            # Get products from both databases
            products = await self.product_manager.search_products(query, limit=50)
            
            # Combine with existing semantic search
            existing_results = await semantic_rag(query, category=category)
            
            # Merge and rank results
            all_results = []
            
            # Add existing results
            if existing_results and isinstance(existing_results, list):
                for result in existing_results:
                    if isinstance(result, dict):
                        result['source'] = 'existing_semantic'
                        all_results.append(result)
            
            # Add new product results
            for product in products:
                product_result = {
                    'style_name': product.get('style_name', ''),
                    'category': product.get('category', ''),
                    'price': product.get('price', 0),
                    'description': product.get('description', ''),
                    'fabric': product.get('fabric_description', ''),
                    'product_link': product.get('product_link', ''),
                    'source': 'product_management',
                    'storage_location': 'postgresql' if product.get('id') else 'mongodb'
                }
                all_results.append(product_result)
            
            # Sort by relevance (you could implement more sophisticated ranking)
            all_results.sort(key=lambda x: x.get('price', 0))  # Simple price-based sorting
            
            return all_results[:20]  # Return top 20 results
            
        except Exception as e:
            logger.error(f"Enhanced semantic search failed: {e}")
            return []
    
    async def get_product_statistics_for_dashboard(self):
        """Get comprehensive product statistics for dashboard"""
        try:
            stats = await self.product_manager.get_product_statistics()
            
            # Add additional analytics
            enhanced_stats = {
                **stats,
                'system_health': {
                    'postgresql_products': stats['storage_distribution']['postgresql'],
                    'mongodb_products': stats['storage_distribution']['mongodb'],
                    'total_categories': len(stats['categories']),
                    'avg_price_range': f"₹{stats['price_range']['min']:.0f} - ₹{stats['price_range']['max']:.0f}"
                },
                'recommendations': {
                    'storage_optimization': 'Consider moving frequently accessed data to PostgreSQL' if stats['storage_distribution']['mongodb'] > stats['storage_distribution']['postgresql'] else 'Storage distribution looks good',
                    'category_balance': 'Consider adding more variety to categories' if len(stats['categories']) < 5 else 'Good category diversity'
                }
            }
            
            return enhanced_stats
            
        except Exception as e:
            logger.error(f"Failed to get enhanced statistics: {e}")
            return {}
    
    async def export_products_for_existing_system(self, format: str = "json"):
        """Export products in format compatible with existing system"""
        try:
            products = await self.product_manager.export_products(format)
            
            if format == "json":
                # Convert to existing system format
                import json
                products_data = json.loads(products)
                
                # Transform to match existing format
                transformed_products = []
                for product in products_data:
                    transformed_product = {
                        'name': product.get('style_name', ''),
                        'category': product.get('category', ''),
                        'price': product.get('price', 0),
                        'description': product.get('description', ''),
                        'fabric': product.get('fabric_description', ''),
                        'link': product.get('product_link', ''),
                        'image': product.get('image_url', ''),
                        'metadata': product.get('metadata', {})
                    }
                    transformed_products.append(transformed_product)
                
                return json.dumps(transformed_products, indent=2)
            
            return products
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return None
    
    async def search_products_with_recommendations(self, query: str, user_preferences: dict = None):
        """Search products and provide AI recommendations"""
        try:
            # Get products
            products = await self.product_manager.search_products(query, limit=20)
            
            # Get AI recommendations
            if user_preferences:
                recommendations = await get_recommendations(query, user_preferences)
            else:
                recommendations = []
            
            # Combine results
            result = {
                'products': products,
                'recommendations': recommendations,
                'query': query,
                'total_found': len(products),
                'ai_suggestions': len(recommendations)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Search with recommendations failed: {e}")
            return {'products': [], 'recommendations': [], 'error': str(e)}

# Global integration instance
product_integration = ProductSystemIntegration()

async def main():
    """Main function to test integration"""
    print("🔧 Testing Product Management System Integration...")
    
    # Initialize
    success = await product_integration.initialize()
    if not success:
        print("❌ Integration failed")
        return
    
    print("✅ Integration successful!")
    
    # Test enhanced search
    print("\n🔍 Testing enhanced semantic search...")
    results = await product_integration.enhanced_semantic_search("jacket", "Jacket")
    print(f"Found {len(results)} products")
    
    # Test statistics
    print("\n📊 Testing statistics...")
    stats = await product_integration.get_product_statistics_for_dashboard()
    print(f"Total products: {stats.get('total_products', 0)}")
    
    # Test export
    print("\n📤 Testing export...")
    export_data = await product_integration.export_products_for_existing_system("json")
    if export_data:
        print("Export successful")
    
    print("\n🎉 Integration testing completed!")

if __name__ == "__main__":
    asyncio.run(main()) 