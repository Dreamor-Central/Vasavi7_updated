import os
import json
import re
from typing import Dict, List, Optional
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import logging
import asyncio
from fuzzywuzzy import fuzz

try:
    # We now import the main semantic_rag function directly
    from semanticrag import semantic_rag
except ImportError as e:
    logging.warning(f"Failed to import semantic_rag: {e}")
    semantic_rag = None

# ==== Logging Setup ====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("styling_agent.log"), logging.StreamHandler()]
)
logger = logging.getLogger("VasaviStylingAgent")

# ==== Environment Variables ====
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set in environment variables")

# ==== Initialize LLM ====
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=0.4)

# ==== Vasavi Catalog Reference (for typo correction and fallback) ====
VASAVI_CATALOG = [
    # Jackets
    {"name": "SASHIMI JACKET", "category": "Jacket", "price": 2699, "fabric": "blended wool", "description": "A stylish, versatile jacket that pairs well with a variety of outfits, offering both warmth and a trendy look.", "link": "https://vasavi.co/products/sashimi"},
    {"name": "ZIP & GO JACKET", "category": "Jacket", "price": 2699, "fabric": "cotton drill", "description": "Overlocked edges add a modern design element, ideal for day-to-night wear.", "link": "https://vasavi.co/products/zip-n-go"},
    {"name": "ECHOES OF PAST JACKET", "category": "Jacket", "price": 3299, "fabric": "fleece", "description": "Features Warli art prints, blending contemporary and traditional aesthetics.", "link": "https://vasavi.co/products/echos-of-the-past"},
    # Bottoms
    {"name": "CHILLINOS PANTS", "category": "Bottoms", "price": 2399, "fabric": "cotton drill", "description": "Overlocked edges add a modern design element, enhancing style and durability.", "link": "https://vasavi.co/products/chillinos"},
    {"name": "PARACHUTE PANTS", "category": "Bottoms", "price": 2399, "fabric": "poly fabric", "description": "Elasticated waist and zipper designs for a cool, modern edge.", "link": "https://vasavi.co/products/parachute-pants"},
    {"name": "BLUE CARGO PANTS", "category": "Bottoms", "price": 2899, "fabric": "denim", "description": "Baggy pants with an attached crop top and adjustable straps.", "link": "https://vasavi.co/products/blue-cargo"},
    # Shirts
    {"name": "LUCIDNET", "category": "Shirt", "price": 2299, "fabric": "Crocia fabric", "description": "Raglan sleeve t-shirt with a techno-infused design.", "link": "https://vasavi.co/products/lucidnet"},
    {"name": "FIELD FLANNEL", "category": "Shirt", "price": 2299, "fabric": "plaid", "description": "High-quality plaid fabric with a relaxed unisex fit.", "link": "https://vasavi.co/products/field-and-flannel"},
    {"name": "DESERT BLOOM", "category": "Shirt", "price": 2499, "fabric": "jersey", "description": "Cactus embroidery for a laid-back, nature-inspired aesthetic.", "link": "https://vasavi.co/products/desert-bloom"},
    # T-Shirts
    {"name": "BLUE BLOCK", "category": "T-Shirt", "price": 1999, "fabric": "jersey", "description": "Features a stylish denim patch with embroidered brand logo.", "link": "https://vasavi.co/products/blue-bloc"},
    {"name": "GOOD OL' DAYS", "category": "T-Shirt", "price": 1999, "fabric": "jersey", "description": "Towel embroidery evokes a nostalgic, vintage touch.", "link": "https://vasavi.co/products/good-ol-days"},
    {"name": "KHOVAR FLOW", "category": "T-Shirt", "price": 1999, "fabric": "jersey", "description": "Bold Khovar-inspired print merges cultural depth with modern design.", "link": "https://vasavi.co/products/khovar-flow"},
    # Hoodies
    {"name": "HOODLINK HOODIE", "category": "Hoodie", "price": 3000, "fabric": "terry", "description": "Unique drawstring detailing on back and sleeves with a Vasavi embroidered patch.", "link": "https://vasavi.co/products/hoodlink"},
    {"name": "THREADED HARMONY HOODIE", "category": "Hoodie", "price": 3899, "fabric": "terry", "description": "Embossed Hindi Vasavi detailing blends cultural depth with comfort.", "link": "https://vasavi.co/products/threaded-harmony"},
    {"name": "SKYDASH HOODIE", "category": "Hoodie", "price": 4199, "fabric": "premium fabric", "description": "Oversized with mesh overlay and adjustable sleeve flaps for a bold aesthetic.", "link": "https://vasavi.co/products/skydash"},
    # Corsets
    {"name": "CORSETICA CORSET", "category": "Corset", "price": 1699, "fabric": "rib", "description": "Body-fitted with adjustable waist belt and drawstrings for bold elegance.", "link": "https://vasavi.co/products/corsetica"},
    {"name": "RAW ALLURE CORSET", "category": "Corset", "price": 2399, "fabric": "PU leather/spandex", "description": "Black PU leather and grey spandex for a modern, high-fashion look.", "link": "https://vasavi.co/products/raw-allure"},
    {"name": "ELYSIAN CURVE CORSET", "category": "Corset", "price": 3499, "fabric": "denim", "description": "Soft, comfortable denim offers flexibility and contemporary sophistication.", "link": "https://vasavi.co/products/elysian-curve"}
]

# ==== Generic Clothing Types (for general queries) ====
GENERIC_CLOTHING = {
    "jacket": {"name": "Jacket", "category": "Jacket", "fabric": "various", "description": "Versatile outerwear for style and warmth."},
    "jeans": {"name": "Jeans", "category": "Bottoms", "fabric": "denim", "description": "Classic denim jeans, versatile for casual looks."},
    "t-shirt": {"name": "T-Shirt", "category": "Top", "fabric": "cotton", "description": "Basic cotton t-shirt, perfect for casual outfits."},
    "shirt": {"name": "Shirt", "category": "Top", "fabric": "cotton", "description": "Casual or formal shirt, great for layering."},
    "pants": {"name": "Pants", "category": "Bottoms", "fabric": "various", "description": "Versatile bottoms for casual or semi-formal looks."},
    "hoodie": {"name": "Hoodie", "category": "Hoodie", "fabric": "cotton/fleece", "description": "Cozy, casual hoodie for relaxed styling."},
    "corset": {"name": "Corset", "category": "Corset", "fabric": "various", "description": "Structured top for bold, sculpted looks."}
}

# ==== Prompt Generator ====
class EnhancedPromptGenerator:
    async def rewrite_query(self, user_input: str) -> str:
        """Rewrite query for clarity using LLM."""
        prompt = """
You are a query rewriter for a fashion e-commerce platform. Rewrite the user's query to be clear, grammatically correct, and specific to fashion items, preserving the original intent. Handle case-insensitive or misspelled inputs. If the query is general (e.g., "look cool and stylish"), make it specific to outfit recommendations. Return only the rewritten query as a string.

Examples:
- "shashmi jacket" -> "SASHIMI JACKET"
- "style tips for jacket" -> "Style tips for wearing a jacket"
- "i want to look cool and stylish" -> "What are some tips for creating cool and stylish outfits?"
- "whats trendy" -> "What are trendy clothing items?"

Query: "{user_input}"
"""
        try:
            response = await llm.ainvoke(prompt.format(user_input=user_input))
            return response.content.strip()
        except Exception as e:
            logger.warning(f"Error rewriting query: {e}")
            return user_input

    def is_followup_question(self, user_input: str) -> bool:
        """Check if the query is a follow-up (e.g., 'what bag with this')."""
        user_input_lower = user_input.lower().strip()
        followup_patterns = [
            r'\bwhat.?bag\b', r'\bwhich.?bag\b', r'\bbag\s*with\b',
            r'\bwhat.?shoe\b', r'\bwhich.?shoe\b', r'\bfootwear\s*with\b',
            r'\bwhat.?jewel(ry|ery)\b', r'\bwhat.?accessories\b',
            r'\bwith.?this\b', r'\bwith.?that\b', r'\bto.?go.?with\b',
            r'\bto.*?match\b', r'\bcomplement\b', r'\bpair.*?with\b'
        ]
        return any(re.search(pattern, user_input_lower) for pattern in followup_patterns)

    def extract_accessory_type(self, user_input: str) -> str:
        """Extract accessory type from follow-up query."""
        user_input_lower = user_input.lower()
        accessory_mapping = {
            'bag': ['bag', 'purse', 'handbag', 'clutch', 'tote'],
            'shoes': ['shoe', 'footwear', 'boot', 'sneaker', 'sandal'],
            'jewelry': ['jewelry', 'necklace', 'earring', 'bracelet', 'ring', 'locket', 'chain'],
            'accessories': ['accessory', 'belt', 'scarf', 'watch']
        }
        for category, keywords in accessory_mapping.items():
            if any(keyword in user_input_lower for keyword in keywords):
                return category
        return 'accessories'

    def correct_typo(self, user_input: str) -> str:
        """Correct potential typos in product names using fuzzy matching."""
        user_input_lower = user_input.lower()
        for item in VASAVI_CATALOG:
            if fuzz.ratio(user_input_lower, item['name'].lower()) > 80:
                logger.info(f"Corrected typo: '{user_input}' to '{item['name']}'")
                return item['name']
        for generic in GENERIC_CLOTHING:
            if fuzz.ratio(user_input_lower, generic.lower()) > 80:
                logger.info(f"Matched generic clothing: '{user_input}' to '{GENERIC_CLOTHING[generic]['name']}'")
                return GENERIC_CLOTHING[generic]['name']
        return user_input

    async def create_styling_prompt(self, user_input: str, recommendations: List[Dict], user_data: Dict, is_followup: bool) -> str:
        occasion = user_data.get('occasion', 'casual')
        # Ensure recommendations are properly formatted for the prompt
        # We want to pass the actual product details for the LLM to work with
        recommendation_str = json.dumps(recommendations, indent=2) if recommendations else "No recommendations provided"

        if is_followup:
            accessory_type = self.extract_accessory_type(user_input)
            prompt = f"""
You are a professional fashion stylist for Vasavi, an online clothing e-commerce platform, with a trendy Gen Z tone (use slang like 'slay,' 'drip,' 'vibes'). The user has a follow-up question about {accessory_type} to complement provided items.

RECOMMENDATIONS:
{recommendation_str}

USER QUERY: "{user_input}"
OCCASION: {occasion}

INSTRUCTIONS:
- Focus ONLY on {accessory_type} recommendations that pair with the provided items.
- Suggest 2-3 specific {accessory_type} items (e.g., 'gold chain necklace', 'leather crossbody bag').
- Explain why each accessory vibes with the outfit.
- Keep it under 100 words, direct, and trendy.
- Do NOT suggest new clothing items outside accessories.
- Output in JSON format: {{"accessories": [{{"name": str, "reason": str}}]}}.

Example:
{{"accessories": [
    {{"name": "Gold chain necklace", "reason": "Adds bold drip to the jacket's urban vibe."}},
    {{"name": "Black leather watch", "reason": "Sleek and classy, matches the jacket's edge."}}
]}}
"""
        else:
            prompt = f"""
You are a professional fashion stylist for Vasavi, an online clothing e-commerce platform, with a trendy Gen Z tone (use slang like 'slay,' 'drip,' 'vibes'). The recommendation agent provided clothing items from Vasavi's catalog, or the query refers to a specific or generic item (e.g., cool outfits). Your role is to enhance these with styling tips.

RECOMMENDATIONS:
{recommendation_str}

USER QUERY: "{user_input}"
OCCASION: {occasion}

INSTRUCTIONS:
- For each recommended item (or query-referenced item if no recommendations), suggest:
  - 1-2 clothing pairings from Vasavi's catalog (Bottoms, Shirts, T-Shirts, Hoodies, Corsets) or generic items if appropriate.
  - 2-3 accessories (e.g., chains, lockets, watches, bags).
  - 1-2 grooming tips (e.g., hairstyle, makeup, nails).
- If no recommendations, assume the query refers to a trendy outfit and suggest a complete look using Vasavi items or generic clothing.
- Ensure pairings vibe with the item's style, fabric, and occasion.
- Match colors and aesthetics for a cohesive look.
- Keep each item's styling under 100 words.
- Output in JSON format: {{"styling_tips": [{{"item_name": str, "pairings": [str], "accessories": [str], "grooming": [str]}}]}}.

Example:
{{"styling_tips": [
    {{"item_name": "SASHIMI JACKET", "pairings": ["CHILLINOS PANTS for a sleek look", "BLUE BLOCK T-SHIRT for casual vibes"], "accessories": ["Silver chain necklace", "Black leather watch"], "grooming": ["Slicked-back hair", "Matte finish makeup"]}}
]}}
"""
        return prompt

    async def evaluate_styling(self, user_input: str, styling_tips: List[Dict], user_data: Dict) -> List[Dict]:
        """Evaluate styling suggestions for relevance using LLM."""
        prompt = """
Evaluate the relevance of each styling suggestion to the user's query and occasion. Return a JSON array of objects with:
- item_name: Product name (string)
- relevance_score: Integer from 0 to 10
- reason: Brief explanation (string)
Query: "{user_input}"
Styling Tips: {styling_tips}
Occasion: {occasion}
Return only the JSON array.

Example:
[
    {"item_name": "SASHIMI JACKET", "relevance_score": 8, "reason": "Matches casual vibe well"},
    {"item_name": "EBON AURA", "relevance_score": 5, "reason": "Too formal for casual"}
]
"""
        try:
            occasion = user_data.get('occasion', 'casual')
            styling_tips_str = json.dumps(styling_tips, indent=2)
            response = await llm.ainvoke(prompt.format(user_input=user_input, styling_tips=styling_tips_str, occasion=occasion))
            evaluated = json.loads(response.content.strip())
            if not isinstance(evaluated, list):
                logger.warning("Invalid evaluation response, returning original tips")
                return styling_tips
            for tip in styling_tips:
                for eval_item in evaluated:
                    if eval_item.get('item_name') == tip.get('item_name'):
                        tip['relevance_score'] = eval_item.get('relevance_score', 5)
                        tip['relevance_reason'] = eval_item.get('reason', '')
            return sorted(styling_tips, key=lambda x: x.get('relevance_score', 5), reverse=True)
        except Exception as e:
            logger.error(f"Error evaluating styling: {e}")
            return styling_tips

# ==== Styling Agent ====
class StylingAgent:
    def __init__(self):
        self.prompt_generator = EnhancedPromptGenerator()
        self.llm = llm

    async def get_styling_advice(self, user_input: str, recommendations: List[Dict] = None, user_data: Dict = None) -> Dict:
        """
        Generate styling advice to complement recommendation agent's output or query-referenced items.
        Args:
            user_input (str): User's query (e.g., "i want to look cool and stylish").
            recommendations (List[Dict]): Recommendation agent's output (optional).
            user_data (Dict): Additional user info (e.g., {"occasion": "casual"}).
        Returns:
            Dict: Styling advice in JSON format or error message.
        """
        try:
            if user_data is None:
                user_data = {}
            user_data['request'] = user_input

            # Rewrite query for clarity
            rewritten_query = await self.prompt_generator.rewrite_query(user_input)
            logger.info(f"Rewritten query: '{user_input}' -> '{rewritten_query}'")
            current_user_input = rewritten_query # Use a new variable for current state

            # Correct typo in query
            corrected_input = self.prompt_generator.correct_typo(current_user_input)
            if corrected_input != current_user_input:
                current_user_input = corrected_input
                user_data['request'] = current_user_input # Update user_data with corrected input

            # Initialize recommendations if not provided
            if recommendations is None:
                recommendations = []
                # Call the main semantic_rag function that returns structured data
                if semantic_rag:
                    try:
                        rag_results = await semantic_rag(current_user_input)
                        if isinstance(rag_results, list):
                            recommendations = rag_results
                            logger.info(f"Semantic RAG returned {len(recommendations)} structured recommendations.")
                        else:
                            # If semantic_rag returns a string (e.g., "I'm not sure..."), handle it
                            logger.warning(f"Semantic RAG returned a non-list type (likely a message): {rag_results}")
                            # In this case, we'll proceed to fallbacks or indicate no recommendations found.
                            recommendations = [] # Ensure recommendations is an empty list if not structured
                    except Exception as e:
                        logger.warning(f"Error calling semantic_rag: {e}")
                        recommendations = []

            # Fallback if no recommendations (from RAG or initial input)
            if not recommendations:
                logger.info("No recommendations from semantic RAG or initial input. Attempting catalog/generic fallbacks.")
                user_input_lower = current_user_input.lower()
                found_in_catalog = False

                # Check for direct catalog item mention
                for item in VASAVI_CATALOG:
                    if item['name'].lower() in user_input_lower or \
                       fuzz.ratio(user_input_lower, item['name'].lower()) > 85: # Increased fuzzy matching threshold for direct match
                        recommendations = [item]
                        logger.info(f"Fallback: Using catalog item {item['name']} for styling based on direct mention.")
                        found_in_catalog = True
                        break
                
                if not found_in_catalog:
                    # Check for generic clothing type mention
                    for generic_key, generic_item in GENERIC_CLOTHING.items():
                        if generic_key.lower() in user_input_lower or \
                           fuzz.ratio(user_input_lower, generic_key.lower()) > 85: # Increased fuzzy matching threshold
                            recommendations = [generic_item]
                            logger.info(f"Fallback: Using generic clothing {generic_item['name']} for styling based on generic mention.")
                            found_in_catalog = True
                            break
                
                if not found_in_catalog:
                    # Default to trendy outfit for vague queries
                    recommendations = [
                        VASAVI_CATALOG[0],  # SASHIMI JACKET
                        VASAVI_CATALOG[6],  # LUCIDNET
                        VASAVI_CATALOG[3]   # CHILLINOS PANTS
                    ]
                    logger.info("Fallback: Using trendy catalog items for vague query as no specific recommendations or mentions found.")

            is_followup = self.prompt_generator.is_followup_question(current_user_input)
            prompt = await self.prompt_generator.create_styling_prompt(current_user_input, recommendations, user_data, is_followup)
            response = await self.llm.ainvoke(prompt)
            ai_response = response.content.strip()

            # Parse JSON response
            try:
                if ai_response.startswith("```json"):
                    ai_response = ai_response[7:-3].strip()
                styling_data = json.loads(ai_response)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON response from LLM: {ai_response}, Error: {e}")
                # Provide a more user-friendly error or a default response if parsing fails
                return {"error": "Oops! I couldn't understand the styling advice. Please try rephrasing your request.", "success": False}

            # Evaluate and rerank styling tips if present
            if styling_data.get('styling_tips'):
                styling_data['styling_tips'] = await self.prompt_generator.evaluate_styling(current_user_input, styling_data['styling_tips'], user_data)
            elif styling_data.get('accessories'):
                # For accessory suggestions, we don't have a separate evaluation step currently defined.
                # If you want to evaluate the quality of accessory suggestions, you'd need to adapt evaluate_styling
                # or create a new evaluation function for it.
                logger.info("Accessory recommendations generated (no separate relevance evaluation for them).")

            return {
                "styling_advice": styling_data,
                "request_type": "followup" if is_followup else "styling",
                "success": True
            }
        except Exception as e:
            logger.error(f"Critical error generating styling advice: {e}", exc_info=True) # Log full traceback
            return {"error": f"Looks like something went wrong! Please try again or rephrase your request. Error: {str(e)}", "success": False}

# ==== CLI Test Function ====
async def main():
    stylist = StylingAgent()
    print("Vasavi Styling Agent CLI")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        user_input = input("\nEnter your styling query (e.g., 'style tips for jacket' or 'what bag with this'): ").strip()
        if user_input.lower() in ('exit', 'quit'):
            print("Exiting CLI...")
            break
        print(f"\nProcessing query: {user_input}")
        try:
            result = await stylist.get_styling_advice(
                user_input=user_input,
                recommendations=None, # Let the styling agent call semantic rag
                user_data={"occasion": "casual"}
            )
            if result["success"]:
                print("\nStyling Advice:")
                print(json.dumps(result["styling_advice"], indent=2))
            else:
                print(f"Error: {result['error']}")
        except Exception as e:
            logger.error(f"Error processing query '{user_input}': {e}", exc_info=True)
            print(f"Oops, something broke! Error: {e}")
        print("\n" + "="*50)

if __name__ == "__main__":
    asyncio.run(main())