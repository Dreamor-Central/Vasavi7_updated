#!/usr/bin/env python3
"""
Enhanced Chat System with RAG Integration and Context Awareness
"""

import json
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)

from product_manager import product_manager
try:
    from semanticrag import semantic_rag
    SEMANTIC_RAG_AVAILABLE = True
except Exception as e:
    logger.warning(f"Semantic RAG not available: {e}")
    semantic_rag = None
    SEMANTIC_RAG_AVAILABLE = False
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

class EnhancedChatSystem:
    def __init__(self):
        self.conversation_history = {}
        self.current_context = {}
        
        # Initialize LLM
        try:
            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
            self.AI_ENABLED = True
        except Exception as e:
            logger.warning(f"OpenAI API key not configured. Error: {e}")
            self.llm = None
            self.AI_ENABLED = False
    
    def _extract_session_id(self, checkpoint_id: Optional[str] = None) -> str:
        """Extract or generate session ID"""
        return checkpoint_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _get_conversation_context(self, session_id: str) -> Dict:
        """Get conversation context for a session"""
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = {
                "messages": [],
                "last_products": [],
                "current_focus": None,
                "created_at": datetime.now()
            }
        return self.conversation_history[session_id]
    
    def _update_context(self, session_id: str, message: str, products: List[Dict], response: str):
        """Update conversation context"""
        context = self._get_conversation_context(session_id)
        context["messages"].append({
            "user": message,
            "assistant": response,
            "timestamp": datetime.now(),
            "products_mentioned": products
        })
        
        if products:
            context["last_products"] = products[:3]  # Keep last 3 products
            context["current_focus"] = products[0] if products else None
        
        # Keep only last 10 messages for context
        if len(context["messages"]) > 10:
            context["messages"] = context["messages"][-10:]
    
    def _detect_follow_up_intent(self, query: str, context: Dict) -> str:
        """Detect if this is a follow-up question about previous products"""
        query_lower = query.lower()
        follow_up_indicators = [
            "tell me more", "more about", "explain", "details", "what about",
            "this one", "that one", "the hoodie", "the shirt", "the jacket",
            "price", "cost", "how much", "fabric", "material", "quality"
        ]
        
        if any(indicator in query_lower for indicator in follow_up_indicators):
            return "follow_up"
        return "new_query"
    
    def _get_product_details(self, product: Dict) -> str:
        """Generate detailed product description"""
        details = f"**{product.get('style_name', 'Unknown Product')}**\n\n"
        details += f"**Price**: ₹{product.get('price', 0)}\n"
        details += f"**Material**: {product.get('fabric_description', 'Not specified')}\n"
        details += f"**Description**: {product.get('description', 'No description available')}\n\n"
        
        if product.get('product_link'):
            details += f"**Available at**: {product.get('product_link')}\n\n"
        
        usage = self._get_usage_suggestions(product)
        details += f"Perfect for {usage}."
        
        return details
    
    def _get_usage_suggestions(self, product: Dict) -> str:
        """Generate usage suggestions based on product category and fabric"""
        category = product.get('category', '').lower()
        fabric = product.get('fabric_description', '').lower()
        
        suggestions = {
            't-shirt': 'casual everyday wear, layering, or casual outings',
            'hoodie': 'cold weather, casual comfort, or outdoor activities',
            'jacket': 'formal occasions, outdoor activities, or fashion statements',
            'shirt': 'professional settings, casual wear, or semi-formal events'
        }
        
        for cat, suggestion in suggestions.items():
            if cat in category:
                return suggestion
        
        return "various occasions and styling needs"
    
    async def _handle_follow_up_query(self, query: str, context: Dict) -> str:
        """Handle follow-up questions about previously mentioned products"""
        if not context.get("last_products"):
            return "I don't have any previous products to tell you more about. Could you please ask about a specific product or category?"
        
        # Try to match specific product names in the query first
        query_lower = query.lower()
        matched_product = None
        
        for product in context["last_products"]:
            product_name = product.get('style_name', '').lower()
            if product_name and product_name in query_lower:
                matched_product = product
                break
        
        # If we found a specific product match, provide details
        if matched_product:
            return self._get_product_details(matched_product)
        
        # If there's a current focus product and no specific match, provide details about it
        if context.get("current_focus"):
            product = context["current_focus"]
            return self._get_product_details(product)
        
        # If multiple products but no specific match, ask for clarification
        if len(context["last_products"]) > 1:
            product_list = "\n".join([
                f"{i+1}. {p.get('style_name', 'Unknown')} ({p.get('category', 'Unknown')}) - ₹{p.get('price', 0)}"
                for i, p in enumerate(context["last_products"])
            ])
            
            return f"""I found several products in our previous conversation. Which one would you like to know more about?

{product_list}

Please specify which product you'd like details about, or ask about a specific aspect like price, fabric, or usage."""
        
        # If only one product, provide details
        return self._get_product_details(context["last_products"][0])
    
    def _extract_search_terms(self, query: str) -> str:
        """Extract key product search terms from natural language query"""
        # Common product categories and terms
        product_terms = [
            'jacket', 'shirt', 't-shirt', 'hoodie', 'pants', 'jeans', 'dress', 'skirt',
            'cotton', 'leather', 'wool', 'silk', 'denim', 'linen', 'polyester',
            'casual', 'formal', 'sport', 'winter', 'summer', 'spring', 'fall'
        ]
        
        query_lower = query.lower()
        
        # Look for product terms in the query
        found_terms = []
        for term in product_terms:
            if term in query_lower:
                found_terms.append(term)
        
        # If we found specific terms, use them
        if found_terms:
            return ' '.join(found_terms)
        
        # Otherwise, try to extract meaningful words (3+ characters)
        words = [word for word in query_lower.split() if len(word) >= 3]
        return ' '.join(words[:3])  # Use up to 3 most relevant words
    
    async def _enhanced_product_search(self, query: str) -> List[Dict]:
        """Enhanced product search using both RAG and database"""
        try:
            all_products = []
            seen_names = set()
            
            # Try semantic RAG search if available
            if SEMANTIC_RAG_AVAILABLE and semantic_rag:
                try:
                    rag_results = await semantic_rag(query)
                    
                    # Add RAG results first (they're more semantically relevant)
                    for result in rag_results:
                        if isinstance(result, dict) and result.get('style_name'):
                            product_name = result['style_name']
                            if product_name not in seen_names:
                                all_products.append(result)
                                seen_names.add(product_name)
                except Exception as e:
                    logger.warning(f"RAG search failed: {e}")
            
            # Also search the product database
            db_results = await product_manager.search_products(query)
            logger.info(f"Enhanced chat search for '{query}' returned {len(db_results)} results from database")
            
            # Add database results
            for product in db_results:
                product_name = product.get('style_name', '')
                if product_name and product_name not in seen_names:
                    all_products.append(product)
                    seen_names.add(product_name)
            
            return all_products[:5]  # Return top 5 results
            
        except Exception as e:
            logger.error(f"Enhanced product search failed: {e}")
            # Fallback to database search only
            return await product_manager.search_products(query)
    
    async def generate_response(self, query: str, checkpoint_id: Optional[str] = None) -> str:
        """Generate enhanced response with context awareness"""
        session_id = self._extract_session_id(checkpoint_id)
        context = self._get_conversation_context(session_id)
        
        # Detect if this is a follow-up question
        intent = self._detect_follow_up_intent(query, context)
        if intent == "follow_up":
            response = await self._handle_follow_up_query(query, context)
            self._update_context(session_id, query, context.get("last_products", []), response)
            return response
        
        # New query - perform enhanced search
        # Extract key product terms from the query
        search_terms = self._extract_search_terms(query)
        products = await self._enhanced_product_search(search_terms)
        
        if not products:
            response = f"I couldn't find any products matching '{query}'. Try searching for different terms like 'cotton', 'jacket', 'hoodie', etc."
        else:
            # Generate concise, professional response
            if len(products) == 1:
                product = products[0]
                response = f"I found a {product.get('category', 'product')} that might interest you:\n\n"
                response += f"**{product.get('style_name', 'Unknown')}** - ₹{product.get('price', 0)}\n"
                response += f"{product.get('description', 'No description available')[:80]}..."
            else:
                response = f"I found {len(products)} products for you:\n\n"
                for i, product in enumerate(products[:3], 1):  # Show only top 3
                    response += f"{i}. **{product.get('style_name', 'Unknown')}** - ₹{product.get('price', 0)}\n"
                
                if len(products) > 3:
                    response += f"\n... and {len(products) - 3} more. Would you like to see specific details about any of these?"
            response += "\n\nNeed more details about any specific item?"
        
        # Update context
        self._update_context(session_id, query, products, response)
        return response

# Global instance
enhanced_chat = EnhancedChatSystem() 