import os
import logging
import time
import random
from typing import List, Dict, Any, Optional
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from pinecone import Pinecone
from pinecone.exceptions import PineconeException
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAIError
import asyncio
from dotenv import load_dotenv

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("sales_agent_tool.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
PINECONE_INDEX_NAME = "salesman-index"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
DIMENSION = 3072  # Matches text-embedding-3-large
TIMEOUT_SECONDS = 10

# Vasavi Info
VASAVI_INFO = {
    "about": "Vasavi is the streetwear brand for those who wear their story with pride. Handmade and crafted with fearless creatives, each piece reflects unapologetic authenticity for those who refuse to blend in. At Vasavi, we believe that clothing isn't just fabric — it's a statement. Break the mold with Vasavi. Each piece is crafted to reflect your bold spirit and unique identity. For those who dare to be different, our designs let you wear your true self and make a statement.",
    "contact": {
        "email": "Support@vasavi.co",
        "phone": "99990109690",
        "address": "Mumbai, India (exact address not listed)",
    },
    "terms_and_conditions": {
        "general": "This website is operated by Vasavi. By visiting our site and/or purchasing something from us, you engage in our 'Service' and agree to be bound by these terms and conditions.",
        "eligibility": "You must be at least 18 years of age or accessing the website under the supervision of a parent or legal guardian to use this site.",
        "products_and_pricing": "All products listed on the website are subject to availability. We reserve the right to discontinue any product at any time. Prices for our products are subject to change without notice. We make every effort to display accurate pricing, but errors may occur.",
        "order_and_payment": "Orders will be confirmed only after successful payment. We accept payments through [UPI, Cards, Net Banking, etc.]. In case of any payment failure, the order will not be processed.",
        "shipping_and_delivery": "We usually dispatch orders within [X] business days. Delivery timelines may vary depending on the location and courier service. Vasavi is not responsible for delays caused by courier partners.",
        "returns_and_refunds": "Returns and exchanges are accepted within 15 days from the purchase date. Products must be unworn, unwashed, and returned with original tags and packaging. All items purchased from the Archive Sales are final sale. These items are not eligible for return, exchange, or cancellation.",
        "intellectual_property": "All content on this site, including images, text, graphics, and logos, is the property of Vasavi and is protected by applicable copyright and trademark laws.",
        "limitation_of_liability": "Vasavi shall not be liable for any direct, indirect, incidental, or consequential damages resulting from the use or inability to use our services or products.",
        "privacy_policy_link": "Please refer to our https://mct3zk-ga.myshopify.com/pages/privacy-policy to understand how we handle your personal information.",
        "governing_law": "These Terms and Conditions are governed by and construed in accordance with the laws of India.",
        "changes_to_terms": "We reserve the right to update or modify these Terms at any time without prior notice. It is your responsibility to check this page periodically.",
    },
    "shipping": {
        "domestic": "Estimated delivery time: 5–7 business days. Orders are processed within 1–2 business days. A tracking link will be sent via WhatsApp once your order is dispatched.",
        "international": "Estimated delivery time: 10–14 business days. Orders are processed within 1–2 business days. A tracking link will be sent via WhatsApp as soon as your order is ready to ship. Please note that delivery times may vary slightly due to customs procedures and regional courier service capabilities.",
        "free_shipping": "Free global shipping on all orders.",
        "free_returns": "Returns within 30 days receive a full refund.",
        "worldwide_shipping": "Ship anywhere, rates available at checkout.",
    },
    "support": {
        "support_email": "Support@vasavi.co",
        "phone": "1(800) 555-1234",
        "support": "24/7 support"
    },
    "social": {
        "instagram": "https://www.instagram.com/vasavi.official/",
        "linkedin": "https://www.linkedin.com/company/vasaviofficial/",
    },
    "returns_and_exchanges": "Returns and exchanges are accepted within 15 days from the purchase date. Products must be unworn, unwashed, and returned with original tags and packaging. Returns can be initiated through our Product Returns Portal."
}

# Objection Handling
OBJECTIONS = {
    "too expensive": {
        "response": [
            {
                "text": "Hey, I get it; budget's a thing! It feels like a splurge, right? [#Empathy] But let's unpack the *investment* here...",
                "techniques": ["Voss: Empathy"],
                "follow_up": {
                    "type": "question",
                    "text": "Where's the pinch point? Is it the sticker price, or are you weighing cost-per-wear?",
                    "options": ["Sticker Shock", "Cost-Per-Wear", "Brand Comparison"],
                }
            },
            {
                "text": "Think of it: this isn't fast fashion. It's a meticulously crafted piece designed to elevate your *whole vibe* for seasons to come [#JonesPhrasing]. We're talking quality that *slaps*, not fades after three washes.",
                "techniques": ["Jones: Magical Phrasing", "Hormozi: Value Framing"],
                "follow_up": {
                    "type": "statement",
                    "text": "Imagine the Insta posts! This piece is a *lewk*.",
                }
            },
            {
                "text": "Seriously, our DMs are flooded with customers saying they get *endless* compliments and that the quality-to-price ratio is unreal [#HormoziValue][#SocialProof]. It's an *investment*, period.",
                "techniques": ["Hormozi: Value Framing", "Cialdini: Social Proof"],
            },
            {
                "text": "To make it *easier* to drop this into your wardrobe, we've got some flex options: Klarna, Afterpay, the works [#SPINSelling]. Wanna peep those?",
                "techniques": ["SPIN: Need-payoff"],
                "follow_up": {
                    "type": "choice",
                    "text": "Payment Flex?",
                    "options": ["Klarna/Afterpay Details", "Installment Breakdown", "Later Date Reminder"],
                }
            }
        ],
        "techniques": ["Voss: Empathy", "Jones: Magical Phrasing", "Hormozi: Value Framing", "SPIN: Need-payoff", "Cialdini: Social Proof"],
        "additional_info": {
            "focus": "Positioning as investment, emphasizing quality, style longevity, and flexible payment",
            "counter_arguments": [
                "Highlighting cost-per-wear and versatility",
                "Showcasing unique design elements and craftsmanship",
                "Leveraging social proof and influencer endorsements",
            ],
        }
    },
    "not sure": {
        "response": [
            {
                "text": "Totally feel you. Big decisions need headspace [#Empathy]. What's the *mood* here? What's making you pause?",
                "techniques": ["Voss: Empathy"],
                "follow_up": {
                    "type": "question",
                    "text": "Is it the fit, the fabric, or are you just not feeling the *vibe* yet?",
                    "options": ["Fit Check", "Fabric Feels", "Style Confidence"],
                }
            },
            {
                "text": "Let's zoom in: What's the *real* blocker? If I could break down how this piece *actually* solves your [specific need], would that help you *add to cart* with confidence? [#VossLabeling][#SPINSelling]",
                "techniques": ["Voss: Labeling", "SPIN: Implication"],
                "follow_up": {
                    "type": "statement",
                    "text": "Pro-tip: Imagine styling it with your fave pieces!",
                }
            },
            {
                "text": "Like, imagine how fire this [product] would look for [specific occasion]! It's designed to turn heads and boost your confidence [#SPINSelling].",
                "techniques": ["SPIN: Need-payoff"],
            },
            {
                "text": "BTW, check the reviews! People are *obsessed* – even the ones who were on the fence at first [#SocialProof].",
                "techniques": ["Cialdini: Social Proof"],
            }
        ],
        "techniques": ["Voss: Empathy", "Voss: Labeling", "SPIN: Implication", "SPIN: Need-payoff", "Cialdini: Social Proof"],
        "additional_info": {
            "focus": "Uncovering hesitation, addressing concerns with info and social proof, and building excitement",
            "counter_arguments": [
                "Offering virtual styling sessions or fit guides",
                "Highlighting easy returns and exchanges",
                "Emphasizing the product's unique selling points and benefits",
            ],
        }
    },
    "not interested": {
        "response": [
            {
                "text": "Aight, I hear that. Not feeling it [#VossMirroring].",
                "techniques": ["Voss: Mirroring"],
                "follow_up": {
                    "type": "question",
                    "text": "No stress! Just curious, what's not *sparking joy* for you right now?",
                    "options": ["Style Mismatch", "Timing's Off", "Need Something Else"],
                }
            },
            {
                "text": "Help me help you! What would make it a *hard pass*? Any feedback is gold so we can level up.",
                "techniques": [],
            },
            {
                "text": "Even though it's not a *yes* today, tons of our customers totally flipped when they saw how our pieces transformed their looks [#SocialProof]. It's a whole *glow-up*!",
                "techniques": ["Cialdini: Social Proof"],
            },
            {
                "text": "No worries if it's not your thing *right now*. But would you be down to stay in the loop for drops and exclusive offers? [#SPINSelling]",
                "techniques": ["SPIN: Need-payoff"],
                "follow_up": {
                    "type": "choice",
                    "text": "Stay Tuned?",
                    "options": ["Email List", "Follow Us", "Occasional DMs"],
                }
            }
        ],
        "techniques": ["Voss: Mirroring", "Cialdini: Social Proof", "SPIN: Need-payoff"],
        "additional_info": {
            "focus": "Respecting the decision, gathering feedback, and nurturing a future relationship",
            "counter_arguments": [
                "Avoiding pressure and focusing on understanding",
                "Offering personalized recommendations based on style preferences",
                "Providing a seamless way to stay updated",
            ],
        }
    },
    "i'll think about it": {
        "response": [
            {
                "text": "Cool, I get that. Gotta marinate on it! [#Empathy] What's the *top thing* on your mind as you mull it over?",
                "techniques": ["Voss: Empathy"],
                "follow_up": {
                    "type": "question",
                    "text": "Is it the budget, picturing it in your closet, or just general *vibes*?",
                    "options": ["Wallet Watch", "Style It Out", "Trust the Process"],
                }
            },
            {
                "text": "Just spitballing here, what's the *biggest* thing holding you back from hitting 'add to cart' right now? [#SPIN: Implication]",
                "techniques": ["SPIN: Implication"],
            },
            {
                "text": "To help you *visualize*, can I drop some extra info? Like, a quick vid on styling or the 411 on our *no-drama* returns? [#SPINSelling]",
                "techniques": ["SPIN: Need-payoff"],
                "follow_up": {
                    "type": "choice",
                    "text": "Helpful Nudges?",
                    "options": ["Styling Reel", "Returns Deets", "Customer Hype"],
                }
            },
            {
                "text": "Heads up though, this piece is *trending*, and sizes are flying off the shelves! Don't wanna see you miss out if you're seriously feeling it [#Cialdini: Scarcity].",
                "techniques": ["Cialdini: Scarcity"],
            }
        ],
        "techniques": ["Voss: Empathy", "SPIN: Implication", "SPIN: Need-payoff", "Cialdini: Scarcity"],
        "additional_info": {
            "focus": "Encouraging a decision by addressing hesitation and creating a sense of urgency (FOMO)",
            "counter_arguments": [
                "Offering a limited-time discount or free shipping",
                "Providing a personalized lookbook or style board",
                "Reiterating the exclusivity and desirability of the item",
            ],
        }
    }
}

# Data Model
class QueryResponse(BaseModel):
    response: str = Field(description="The response text")
    sources: Optional[List[Dict[str, Any]]] = Field(default=None, description="Retrieved knowledge sources")
    tags: Optional[List[str]] = Field(default=None, description="Applied sales techniques")
    raw_response: Optional[str] = Field(default=None)
    intent: Optional[str] = Field(default=None)
    confidence: float = Field(default=1.0)

class SalesAgent:
    def __init__(self):
        self.pinecone_client = None
        self.openai_client = None
        self.index = None
        self._init_pinecone()
        self._init_openai()

    def _init_pinecone(self):
        """Initialize Pinecone client"""
        try:
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
            if not pinecone_api_key:
                raise ValueError("PINECONE_API_KEY not set")
            self.pinecone_client = Pinecone(api_key=pinecone_api_key)
            index_names = self.pinecone_client.list_indexes().names()
            if PINECONE_INDEX_NAME not in index_names:
                logger.error(f"Index {PINECONE_INDEX_NAME} not found")
                raise ValueError(f"Pinecone index {PINECONE_INDEX_NAME} not found")
            self.index = self.pinecone_client.Index(PINECONE_INDEX_NAME)
            index_stats = self.index.describe_index_stats()
            if index_stats["dimension"] != DIMENSION:
                logger.error(f"Index dimension mismatch: expected {DIMENSION}, got {index_stats['dimension']}")
                raise ValueError(f"Index dimension mismatch: expected {DIMENSION}")
            logger.info(f"Connected to Pinecone index: {PINECONE_INDEX_NAME}")
        except Exception as e:
            logger.error(f"Pinecone initialization failed: {e}")
            raise

    def _init_openai(self):
        """Initialize OpenAI client"""
        try:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY not set")
            self.openai_client = OpenAI(api_key=openai_api_key)
            logger.info("OpenAI client initialized")
        except Exception as e:
            logger.error(f"OpenAI initialization failed: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((PineconeException, ConnectionError)),
        reraise=True
    )
    async def retrieve_knowledge(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve knowledge from Pinecone"""
        try:
            async with asyncio.timeout(TIMEOUT_SECONDS):
                embeddings = self.openai_client.embeddings.create(
                    input=query,
                    model=EMBEDDING_MODEL
                )
                query_vector = embeddings.data[0].embedding
                if not query_vector:
                    logger.warning("Empty query vector returned")
                    return []
                result = self.index.query(
                    vector=query_vector,
                    top_k=top_k,
                    include_metadata=True
                )
                matches = result.get("matches", [])
                return [{"score": m["score"], **m["metadata"]} for m in matches]
        except Exception as e:
            logger.warning(f"Retrieval failed: {e}")
            return []

    def _build_prompt(self, query: str, context: str) -> List[Dict[str, str]]:
        """Build system prompt"""
        system_prompt = """
You are Vasavi's Advanced Sales Agent, trained on sales books:
- Cialdini’s Influence (Reciprocity, Scarcity, Authority, Consistency, Liking, Social Proof)
- SPIN Selling (Situation, Problem, Implication, Need-payoff)
- Voss’s Never Split the Difference (mirroring, labeling, empathy)
- Hormozi’s 100M Offers (irresistible offers, 10x value)
- Jones’s Exactly What to Say (“Just imagine…”)
- Belfort’s Straight Line Selling (tonality, objection handling)
- Cardone’s Closer’s Survival Guide (confident closing)
- Suby’s Sell Like Crazy (storytelling, hooks)
- Tracy’s Psychology of Selling (emotional guidance, desire creation)

When responding:
1. Use retrieved context to select persuasion tactics.
2. Apply emotional guidance, urgency, social proof, or scarcity.
3. Use confident, friendly text-based tonality with emojis.
4. Tag responses with behavior markers (e.g., [#Urgency]).
5. Offer next steps or help.
6. For objections, use techniques like Voss’s mirroring.
7. Never fabricate information or testimonials.
8. If unsure, suggest emailing Support@vasavi.co.
9. Keep responses concise and conversational, ideally one line for simple queries.

Retrieved Knowledge:
{context}
"""
        return [
            {"role": "system", "content": system_prompt.format(context=context)},
            {"role": "user", "content": query}
        ]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((OpenAIError,)),
        reraise=True
    )
    async def generate_response(self, query: str) -> QueryResponse:
        """Generate response using LLM"""
        try:
            async with asyncio.timeout(TIMEOUT_SECONDS):
                results = await self.retrieve_knowledge(query)
                context = "\n".join([f"{r['book']} - {r['title']}\n{r['content']}" for r in results])
                messages = self._build_prompt(query, context)
                response = self.openai_client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=500
                )
                answer = response.choices[0].message.content.strip()
                tags = set()
                for chunk in results:
                    if "tags" in chunk:
                        tags.update(chunk["tags"] if isinstance(chunk["tags"], list) else chunk["tags"].split(","))
                formatted_tags = " ".join(sorted(f"[#{tag.strip('[# ]')}]" for tag in tags)) if tags else ""
                return QueryResponse(
                    response=f"{answer}\n\n{formatted_tags}",
                    sources=results,
                    tags=list(tags),
                    raw_response=answer,
                    intent="complex_query",
                    confidence=0.9 if results else 0.7
                )
        except Exception as e:
            logger.exception(f"Error generating response: {e}")
            return QueryResponse(
                response="I'm having trouble answering right now. Please email Support@vasavi.co.",
                intent="error",
                confidence=0.1
            )

    def handle_intent(self, query: str) -> Optional[QueryResponse]:
        """Handle hardcoded intents"""
        q = query.lower().strip()
        if not q:
            return QueryResponse(
                response="I didn't catch that. Could you ask something about Vasavi?",
                intent="error",
                confidence=1.0
            )

        greeting_words = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"]
        if any(w in q for w in greeting_words):
            return QueryResponse(
                response=f"{self._get_greeting()}",
                intent="greeting",
                confidence=1.0
            )

        farewell_words = ["bye", "goodbye", "tata", "see you", "farewell", "later"]
        if any(w in q for w in farewell_words):
            return QueryResponse(
                response=self._get_farewell(),
                intent="farewell",
                confidence=1.0
            )

        about_words = ["about", "brand", "what is vasavi", "tell me about you", "who are you"]
        if any(w in q for w in about_words):
            return QueryResponse(
                response=f"{VASAVI_INFO['about']}",
                intent="about",
                confidence=1.0
            )

        contact_words = ["contact", "email", "phone", "address", "reach you", "get in touch"]
        if any(w in q for w in contact_words):
            c = VASAVI_INFO["contact"]
            return QueryResponse(
                response=f"📧 Email: {c['email']} | 📞 Phone: {c['phone']} | 🏠 Address: {c['address']}",
                intent="contact",
                confidence=1.0
            )

        shipping_words = ["ship", "delivery", "shipping", "when arrive", "how long to ship", "postage"]
        if any(w in q for w in shipping_words):
            s = VASAVI_INFO["shipping"]
            return QueryResponse(
                response=f"🚚 Domestic: {s['domestic']} | 🌍 International: {s['international']} | 🎁 Free shipping: {s['free_shipping']}",
                intent="shipping",
                confidence=1.0
            )

        returns_words = ["return", "exchange", "refund", "policy", "returns", "exchanges"]
        if any(w in q for w in returns_words):
            return QueryResponse(
                response="I'm so sorry you're feeling frustrated! 😔 You can return unworn items within 15 days via our Product Returns Portal. Need help with the process? [#VossEmpathy]",
                intent="returns",
                tags=["Voss: Empathy"],
                confidence=1.0
            )

        social_words = ["instagram", "linkedin", "social", "facebook", "twitter", "insta"]
        if any(w in q for w in social_words):
            s = VASAVI_INFO["social"]
            response_parts = []
            if any(w in q for w in ["instagram", "insta"]):
                response_parts.append(f"📸 Instagram: {s['instagram']}")
            if "linkedin" in q:
                response_parts.append(f"💼 LinkedIn: {s['linkedin']}")
            if response_parts:
                return QueryResponse(
                    response="\n".join(response_parts),
                    intent="social",
                    confidence=1.0
                )

        support_words = ["human", "agent", "support", "representative", "want to speak to someone"]
        if any(w in q for w in support_words):
            sup = VASAVI_INFO["support"]
            return QueryResponse(
                response=f"📧 Email: {sup['support_email']} | 📞 Phone: {sup['phone']} | 💬 Support: {sup['support']}",
                intent="human_support",
                confidence=1.0
            )

        terms_words = ["terms", "conditions", "policy", "legal"]
        if any(w in q for w in terms_words):
            tc = VASAVI_INFO["terms_and_conditions"]
            response = "Here are some of our terms:\n"
            for key, value in tc.items():
                response += f"- {key.replace('_', ' ').title()}: {value[:100]}...\n"
            return QueryResponse(
                response=response,
                intent="terms_and_conditions",
                confidence=0.9
            )

        for objection, data in OBJECTIONS.items():
            if objection in q:
                response_data = random.choice(data["response"])
                response_text = response_data["text"]
                return QueryResponse(
                    response=response_text,
                    tags=response_data.get("techniques", []),
                    intent="objection",
                    confidence=0.95
                )

        return QueryResponse(
            response="Let me see if I can find more information on that...",
            intent="sales",
            confidence=0.6
        )

    def _get_greeting(self) -> str:
        """Generates a random, engaging greeting"""
        greetings = [
            "Hey there, fashion trailblazer! 👋 What's the vibe today?",
            "Hi! 😊 Ready to dive into Vasavi's bold streetwear?",
            "Hello! 🌟 How can I help you stand out with style?",
            "Greetings! ✨ What's sparking your fashion interest?",
            "Welcome to Vasavi! 🚀 What's your style story?",
            "Namaste! 🙏 Ready to break the mold with Vasavi?",
            "Good to see you! 😎 What's the fashion plan?",
            "Hey! 🔥 Looking for something that *slaps*?",
            "Hi there! 🛍️ Let's explore Vasavi's collection!",
            "Hello, trendsetter! 👑 What's next for your wardrobe?"
        ]
        return random.choice(greetings)

    def _get_farewell(self) -> str:
        """Generates a random, friendly farewell"""
        farewells = [
            "Catch ya later, style star! 🌟 Stay bold!",
            "Until next time! 😊 Keep rocking it!",
            "Take care! ✌️ Back soon for more Vasavi vibes?",
            "Farewell! ✨ Keep shining your unique style!",
            "Peace out! 🚀 Wear your story proud!",
            "Great connecting! 😎 See you at Vasavi soon!",
            "Bye for now! 🛍️ Stay stylish!",
            "Adios! 👋 Keep breaking the fashion mold!",
            "Later, vibe master! 😊 Watch for new drops!",
            "Signing off! 🌈 Keep slaying the style game!"
        ]
        return random.choice(farewells)

    def _get_help_prompt(self) -> str:
        """Generates a comprehensive help prompt"""
        return (
            "I'm here to help with Vasavi! Ask about products, shipping, returns, contact, or more! What's up?"
        )

    async def respond(self, query: str) -> QueryResponse:
        """Main method to respond to user input"""
        try:
            intent_response = self.handle_intent(query)
            if intent_response.intent != "complex_query":
                return intent_response
            return await self.generate_response(query)
        except Exception as e:
            logger.exception(f"Error responding to query: {e}")
            return QueryResponse(
                response="I'm having trouble answering right now. Please email Support@vasavi.co.",
                intent="error",
                confidence=0.1
            )

class SalesAgentTool(BaseTool):
    name: str = "sales_agent_tool"
    description: str = "LangChain tool for Vasavi's Sales Agent, for LangGraph integration."

    def _run(self, query: str) -> Dict[str, Any]:
        """Synchronous run method for BaseTool"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self._arun(query))

    async def _arun(self, query: str) -> Dict[str, Any]:
        """Asynchronous run method"""
        try:
            agent = SalesAgent()
            result = await agent.respond(query)
            return {
                "output": result.response,
                "sources": result.sources,
                "tags": result.tags,
                "intent": result.intent,
                "confidence": result.confidence
            }
        except Exception as e:
            logger.exception(f"Error in sales_agent_tool: {e}")
            return {
                "output": "I'm having trouble answering right now. Please email Support@vasavi.co.",
                "intent": "error",
                "confidence": 0.1
            }

# Instantiate the tool
sales_agent_tool = SalesAgentTool()

if __name__ == "__main__":
    async def main():
        query = "I had a bad experience with online shopping; why should I trust Vasavi?."
        result = await sales_agent_tool._arun(query)
        print(f"Query: {query}")
        print(f"Response: {result['output']}")
        print(f"Sources: {result['sources']}")
        print(f"Tags: {result['tags']}")
        print(f"Intent: {result['intent']}")
        print(f"Confidence: {result['confidence']}")

    asyncio.run(main())