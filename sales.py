import os
import logging
import time
import random
from typing import List, Dict, Any, Optional, Tuple
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from pinecone import Pinecone
from pinecone.exceptions import PineconeException
from openai import OpenAI, OpenAIError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import asyncio
from dotenv import load_dotenv
import re
from datetime import datetime

# Configure Logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("sales_agent_tool.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
PINECONE_INDEX_NAME = "salesman-index"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
DIMENSION = 3072  # Matches text-embedding-3-large
TIMEOUT_SECONDS = 15
RETRIEVAL_TOP_K = 10
RERANK_CANDIDATES = 5
CONFIDENCE_THRESHOLD = 0.75

# Vasavi Info (unchanged from your original)
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

# Objection Handling (expanded with trust objection)
OBJECTIONS = {
    "too expensive": {
        "response": [
            {
                "text": "I understand, budgets matter! It feels like a splurge, right? [#Empathy] Let’s explore the *investment* in a Vasavi piece...",
                "techniques": ["Voss: Empathy"],
                "follow_up": {
                    "type": "question",
                    "text": "Where’s the concern? Is it the initial cost, or how often you’d wear it?",
                    "options": ["Sticker Shock", "Cost-Per-Wear", "Brand Comparison"],
                }
            },
            {
                "text": "This isn’t fast fashion—imagine a meticulously crafted piece elevating your style for years! [#JonesPhrasing] Quality that endures, not fades. [#HormoziValue]",
                "techniques": ["Jones: Magical Phrasing", "Hormozi: Value Framing"],
                "follow_up": {
                    "type": "statement",
                    "text": "Picture the compliments on Insta! A true style statement.",
                }
            },
            {
                "text": "Our customers rave about endless compliments and unmatched quality—worth every penny! [#HormoziValue][#SocialProof]",
                "techniques": ["Hormozi: Value Framing", "Cialdini: Social Proof"],
            },
            {
                "text": "To ease you in, we offer Klarna, Afterpay—flexible payments to make it yours! [#SPINSelling] Curious about these?",
                "techniques": ["SPIN: Need-payoff"],
                "follow_up": {
                    "type": "choice",
                    "text": "Payment Options?",
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
                "text": "Totally feel you. Big decisions need headspace [#Empathy]. What’s the *mood* here? What’s making you pause?",
                "techniques": ["Voss: Empathy"],
                "follow_up": {
                    "type": "question",
                    "text": "Is it the fit, the fabric, or are you just not feeling the *vibe* yet?",
                    "options": ["Fit Check", "Fabric Feels", "Style Confidence"],
                }
            },
            {
                "text": "Let’s zoom in: What’s the *real* blocker? If I could break down how this piece *actually* solves your [specific need], would that help you *add to cart* with confidence? [#VossLabeling][#SPINSelling]",
                "techniques": ["Voss: Labeling", "SPIN: Implication"],
                "follow_up": {
                    "type": "statement",
                    "text": "Pro-tip: Imagine styling it with your fave pieces!",
                }
            },
            {
                "text": "Like, imagine how fire this [product] would look for [specific occasion]! It’s designed to turn heads and boost your confidence [#SPINSelling].",
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
                "text": "Even though it’s not a *yes* today, tons of our customers totally flipped when they saw how our pieces transformed their looks [#SocialProof]. It’s a whole *glow-up*!",
                "techniques": ["Cialdini: Social Proof"],
            },
            {
                "text": "No worries if it’s not your thing *right now*. But would you be down to stay in the loop for drops and exclusive offers? [#SPINSelling]",
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
                "text": "Cool, I get that. Gotta marinate on it! [#Empathy] What’s the *top thing* on your mind as you mull it over?",
                "techniques": ["Voss: Empathy"],
                "follow_up": {
                    "type": "question",
                    "text": "Is it the budget, picturing it in your closet, or just general *vibes*?",
                    "options": ["Wallet Watch", "Style It Out", "Trust the Process"],
                }
            },
            {
                "text": "Just spitballing here, what’s the *biggest* thing holding you back from hitting 'add to cart' right now? [#SPIN: Implication]",
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
                "text": "Heads up though, this piece is *trending*, and sizes are flying off the shelves! Don’t wanna see you miss out if you’re seriously feeling it [#Cialdini: Scarcity].",
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
    },
    "trust": {
        "response": [
            {
                "text": "I’m so sorry to hear about your past experience! 😔 We get it—trust is everything. [#VossEmpathy] At Vasavi, we’re committed to quality and your satisfaction.",
                "techniques": ["Voss: Empathy"],
                "follow_up": {
                    "type": "question",
                    "text": "What part of your last experience worries you most? Shipping, quality, or something else?",
                    "options": ["Shipping Delays", "Product Quality", "Customer Service"],
                }
            },
            {
                "text": "Our customers rave about Vasavi’s handmade quality and seamless service—hundreds of 5-star reviews! [#SocialProof] We back it with easy 15-day returns and 24/7 support.",
                "techniques": ["Cialdini: Social Proof", "Hormozi: Value Framing"],
                "follow_up": {
                    "type": "statement",
                    "text": "Just imagine the confidence of a bold, reliable Vasavi piece in your wardrobe! 🌟",
                }
            },
            {
                "text": "We stand by our craft: every piece is handmade, quality-checked, and shipped with care—free globally! [#Authority] Ready to trust the Vasavi difference? [#SPINSelling]",
                "techniques": ["Cialdini: Authority", "SPIN: Need-payoff"],
                "follow_up": {
                    "type": "choice",
                    "text": "Next Step?",
                    "options": ["Explore Products", "Returns Info", "Contact Support"],
                }
            }
        ],
        "techniques": ["Voss: Empathy", "Cialdini: Social Proof", "Hormozi: Value Framing", "Cialdini: Authority", "SPIN: Need-payoff"],
        "additional_info": {
            "focus": "Building trust through empathy, social proof, quality assurance, and risk reduction",
            "counter_arguments": [
                "Highlighting rigorous quality control and handmade craftsmanship",
                "Emphasizing hassle-free returns and 24/7 support",
                "Leveraging customer testimonials and satisfaction stats",
            ],
        }
    }
}

# Data Models
class QueryContext(BaseModel):
    query: str = Field(description="Original user query")
    timestamp: str = Field(description="Query timestamp")
    cleaned_query: str = Field(description="Preprocessed query")
    keywords: List[str] = Field(description="Extracted keywords")
    intent: str = Field(description="Detected intent")
    confidence: float = Field(description="Intent detection confidence")
    history_context: str = Field(description="Recent query history for context")

class QueryResponse(BaseModel):
    response: str = Field(description="Final response text")
    sources: Optional[List[Dict[str, Any]]] = Field(default=None, description="Retrieved knowledge sources")
    tags: Optional[List[str]] = Field(default=None, description="Applied sales techniques")
    raw_response: Optional[str] = Field(default=None, description="Raw LLM response")
    intent: str = Field(description="Confirmed intent")
    confidence: float = Field(description="Response confidence")
    rerank_score: float = Field(description="Reranked quality score")
    follow_up: Optional[Dict[str, Any]] = Field(default=None, description="Follow-up question or action")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context or metrics")

class SalesAgent:
    def __init__(self):
        self.pinecone_client = None
        self.openai_client = None
        self.index = None
        self.query_history = []
        self._init_pinecone()
        self._init_openai()

    def _init_pinecone(self):
        """Initialize Pinecone client with robust error handling"""
        try:
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
            if not pinecone_api_key:
                logger.error("PINECONE_API_KEY not set in environment")
                raise ValueError("PINECONE_API_KEY not set")
            self.pinecone_client = Pinecone(api_key=pinecone_api_key)
            index_names = self.pinecone_client.list_indexes().names()
            if PINECONE_INDEX_NAME not in index_names:
                logger.error(f"Index {PINECONE_INDEX_NAME} not found in Pinecone")
                raise ValueError(f"Pinecone index {PINECONE_INDEX_NAME} not found")
            self.index = self.pinecone_client.Index(PINECONE_INDEX_NAME)
            index_stats = self.index.describe_index_stats()
            if index_stats["dimension"] != DIMENSION:
                logger.error(f"Index dimension mismatch: expected {DIMENSION}, got {index_stats['dimension']}")
                raise ValueError(f"Index dimension mismatch: expected {DIMENSION}")
            logger.info(f"Successfully connected to Pinecone index: {PINECONE_INDEX_NAME}")
        except Exception as e:
            logger.critical(f"Pinecone initialization failed: {str(e)}")
            raise

    def _init_openai(self):
        """Initialize OpenAI client with robust error handling"""
        try:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                logger.error("OPENAI_API_KEY not set in environment")
                raise ValueError("OPENAI_API_KEY not set")
            self.openai_client = OpenAI(api_key=openai_api_key)
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.critical(f"OpenAI initialization failed: {str(e)}")
            raise

    def _preprocess_query(self, query: str) -> QueryContext:
        """Preprocess query: clean, extract keywords, and build context"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cleaned = re.sub(r'[^\w\s]', '', query.lower().strip())
        keywords = [w for w in cleaned.split() if len(w) > 2]
        intent, confidence = self._detect_initial_intent(cleaned, keywords)
        history_context = "\n".join([f"Q: {h.query} (Intent: {h.intent})" for h in self.query_history[-3:]])
        context = QueryContext(
            query=query,
            timestamp=timestamp,
            cleaned_query=cleaned,
            keywords=keywords,
            intent=intent,
            confidence=confidence,
            history_context=history_context
        )
        self.query_history.append(context)
        logger.debug(f"Query context: {context.model_dump()}")
        return context

    def _detect_initial_intent(self, query: str, keywords: List[str]) -> Tuple[str, float]:
        """Detect intent with weighted keyword scoring and context"""
        intent_scores = {
            "greeting": 0.0, "farewell": 0.0, "about": 0.0, "contact": 0.0,
            "shipping": 0.0, "returns": 0.0, "social": 0.0, "support": 0.0,
            "terms": 0.0, "objection": 0.0, "product": 0.0, "trust": 0.0, "complex_query": 0.3
        }
        keyword_map = {
            "greeting": ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"],
            "farewell": ["bye", "goodbye", "tata", "see you", "farewell", "later"],
            "about": ["about", "brand", "what is vasavi", "tell me about you", "who are you"],
            "contact": ["contact", "email", "phone", "address", "reach you", "get in touch"],
            "shipping": ["ship", "delivery", "shipping", "when arrive", "how long to ship", "postage", "international", "deliveries"],
            "returns": ["return", "exchange", "refund", "policy", "returns", "exchanges"],
            "social": ["instagram", "linkedin", "social", "facebook", "twitter", "insta"],
            "support": ["human", "agent", "support", "representative", "want to speak to someone"],
            "terms": ["terms", "conditions", "policy", "legal"],
            "product": ["jacket", "sashimi", "shirt", "pants", "collection", "item", "product"],
            "trust": ["trust", "bad experience", "reliable", "quality", "why should i", "past experience", "worry"]
        }
        for intent, words in keyword_map.items():
            for word in words:
                if word in query:
                    intent_scores[intent] += 0.5
                if word in keywords:
                    intent_scores[intent] += 0.3  # Bonus for exact keyword match
        for objection in OBJECTIONS.keys():
            if objection in query:
                intent_scores["objection"] = max(intent_scores["objection"], 0.95)
                intent_scores["trust"] = max(intent_scores["trust"], 0.9) if "trust" in objection else intent_scores["trust"]
        best_intent = max(intent_scores, key=intent_scores.get)
        confidence = intent_scores[best_intent]
        logger.debug(f"Intent detection: {best_intent}, Score: {confidence}")
        return best_intent, min(confidence, 1.0)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        retry=retry_if_exception_type((PineconeException, ConnectionError)),
        reraise=True
    )
    async def retrieve_knowledge(self, query: str, keywords: List[str], top_k: int = RETRIEVAL_TOP_K) -> List[Dict]:
        """Retrieve and rank knowledge from Pinecone with advanced filtering"""
        try:
            async with asyncio.timeout(TIMEOUT_SECONDS):
                embeddings = self.openai_client.embeddings.create(
                    input=query,
                    model=EMBEDDING_MODEL
                )
                query_vector = embeddings.data[0].embedding
                if not query_vector:
                    logger.warning("Empty query vector returned from embedding")
                    return []
                filter_dict = {
                    "relevance": {"$gte": 0.5},  # Lowered threshold for broader recall
                    "keywords": {"$in": keywords[:5]}  # Match top keywords
                }
                if "trust" in keywords or "quality" in keywords:
                    filter_dict["category"] = {"$in": ["trust", "quality", "customer_satisfaction"]}
                result = self.index.query(
                    vector=query_vector,
                    top_k=top_k,
                    include_metadata=True,
                    filter=filter_dict
                )
                matches = result.get("matches", [])
                if not matches:
                    logger.debug("No Pinecone matches found")
                    return []
                ranked = []
                for m in matches:
                    score = m["score"]
                    metadata = m["metadata"]
                    keyword_hits = sum(1 for kw in keywords if kw in str(metadata.get("content", "")))
                    adjusted_score = score * (1 + 0.15 * keyword_hits)  # Boost for keyword overlap
                    ranked.append({"score": adjusted_score, **metadata})
                ranked = sorted(ranked, key=lambda x: x["score"], reverse=True)
                logger.debug(f"Retrieved and ranked {len(ranked)} documents")
                return ranked
        except Exception as e:
            logger.warning(f"Retrieval failed: {str(e)}")
            return []

    def _build_advanced_prompt(self, context: QueryContext, retrieved: List[Dict]) -> List[Dict[str, str]]:
        """Build a robust, context-aware prompt for elite salesmanship"""
        context_str = "\n".join([f"{r['book']} - {r['title']}\nContent: {r['content']}\nTags: {r.get('tags', '')}" for r in retrieved])
        system_prompt = """
You are Vasavi's Elite Sales Agent, a world-class professional clothing salesman radiating expertise, confidence, and charm. Trained on:
- Cialdini’s Influence: Reciprocity, Scarcity, Authority, Consistency, Liking, Social Proof
- SPIN Selling: Situation, Problem, Implication, Need-payoff
- Voss’s Never Split the Difference: mirroring, labeling, empathy
- Hormozi’s 100M Offers: irresistible offers, 10x value
- Jones’s Exactly What to Say: “Just imagine…” phrasing
- Belfort’s Straight Line Selling: tonality, objection handling
- Cardone’s Closer’s Survival Guide: confident closing
- Suby’s Sell Like Crazy: storytelling, hooks
- Tracy’s Psychology of Selling: emotional guidance, desire creation

Mission:
- Champion Vasavi’s ethos: streetwear for bold, authentic souls who refuse to blend in.
- Deliver precise, persuasive, polished responses, blending sophistication with Vasavi’s edgy vibe.
- Use retrieved knowledge and brand data to craft tailored, trust-building solutions.
- For trust concerns, lean on empathy, social proof, authority, and risk reduction (returns, support).
- Apply sales techniques: empathy for doubts, scarcity for urgency, social proof for confidence.
- Maintain an upscale, confident, yet warm tone with strategic emojis.
- Tag responses with techniques (e.g., [#Empathy], [#Scarcity]).
- Handle general queries (shipping, returns) with facts and persuasive flair.
- For objections, probe root causes, counter with value, quality, and trust.
- For products, spotlight craftsmanship, uniqueness, and personal fit.
- Never invent facts, prices, or testimonials—use data or RAG.
- If unsure, guide to Support@vasavi.co with elegance.
- Drive sales with clear next steps (e.g., order, explore, contact).

Brand Data:
- About: {about}
- Contact: Email: {contact_email}, Phone: {contact_phone}, Address: {contact_address}
- Shipping: Domestic: {ship_domestic}, International: {ship_international}, Free: {ship_free}
- Returns: {returns_policy}
- Social: Instagram: {social_insta}, LinkedIn: {social_linkedin}

Retrieved Knowledge:
{context}

Query History:
{history}

Intent: {intent}
Keywords: {keywords}
Query: {query}
"""
        return [
            {
                "role": "system",
                "content": system_prompt.format(
                    about=VASAVI_INFO["about"],
                    contact_email=VASAVI_INFO["contact"]["email"],
                    contact_phone=VASAVI_INFO["contact"]["phone"],
                    contact_address=VASAVI_INFO["contact"]["address"],
                    ship_domestic=VASAVI_INFO["shipping"]["domestic"],
                    ship_international=VASAVI_INFO["shipping"]["international"],
                    ship_free=VASAVI_INFO["shipping"]["free_shipping"],
                    returns_policy=VASAVI_INFO["returns_and_exchanges"],
                    social_insta=VASAVI_INFO["social"]["instagram"],
                    social_linkedin=VASAVI_INFO["social"]["linkedin"],
                    context=context_str,
                    history=context.history_context,
                    intent=context.intent,
                    keywords=", ".join(context.keywords),
                    query=context.query
                )
            },
            {"role": "user", "content": context.query}
        ]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=15),
        retry=retry_if_exception_type((OpenAIError,)),
        reraise=True
    )
    async def generate_response_candidates(self, context: QueryContext, retrieved: List[Dict]) -> List[QueryResponse]:
        """Generate multiple response candidates for reranking"""
        try:
            async with asyncio.timeout(TIMEOUT_SECONDS):
                messages = self._build_advanced_prompt(context, retrieved)
                candidates = []
                for i in range(RERANK_CANDIDATES):
                    response = self.openai_client.chat.completions.create(
                        model=OPENAI_MODEL,
                        messages=messages,
                        temperature=0.6 + (i * 0.1),  # Vary for diversity
                        max_tokens=600,
                        top_p=0.9
                    )
                    answer = response.choices[0].message.content.strip()
                    tags = set()
                    for chunk in retrieved:
                        if "tags" in chunk:
                            tags.update(chunk["tags"] if isinstance(chunk["tags"], list) else chunk["tags"].split(","))
                    formatted_tags = " ".join(sorted(f"[#{tag.strip('[# ]')}]" for tag in tags)) if tags else ""
                    follow_up = self._extract_follow_up(answer, context)
                    candidates.append(QueryResponse(
                        response=f"{answer}\n\n{formatted_tags}",
                        sources=retrieved,
                        tags=list(tags),
                        raw_response=answer,
                        intent=context.intent,
                        confidence=0.9 if retrieved else 0.7,
                        rerank_score=0.0,
                        follow_up=follow_up,
                        metadata={"generation_index": i, "timestamp": context.timestamp}
                    ))
                logger.debug(f"Generated {len(candidates)} response candidates")
                return candidates
        except Exception as e:
            logger.error(f"Error generating response candidates: {str(e)}")
            return [QueryResponse(
                response="My sincere apologies, I’m unable to assist right now. Please contact Support@vasavi.co for expert guidance! 🌟",
                intent="error",
                confidence=0.1,
                rerank_score=0.0,
                metadata={"error": str(e)}
            )]

    def _extract_follow_up(self, response: str, context: QueryContext) -> Optional[Dict[str, Any]]:
        """Extract follow-up actions or questions from response or objections"""
        for objection, data in OBJECTIONS.items():
            if objection in context.cleaned_query:
                response_data = random.choice(data["response"])
                if "follow_up" in response_data:
                    return response_data["follow_up"]
        if "?" in response:
            lines = response.split("\n")
            for line in lines:
                if line.strip().endswith("?"):
                    return {"type": "question", "text": line.strip()}
        return None

    async def _evaluate_and_rerank(self, context: QueryContext, candidates: List[QueryResponse]) -> QueryResponse:
        """Evaluate and rerank responses for quality, relevance, and sales impact"""
        try:
            async with asyncio.timeout(TIMEOUT_SECONDS):
                eval_prompt = """
You are an expert evaluator for Vasavi's Elite Sales Agent. Score each response (0-100) based on:
1. Relevance (0-30): Does it address the query and intent precisely?
2. Professionalism (0-25): Is the tone polished, confident, and Vasavi-aligned?
3. Persuasiveness (0-25): Are sales techniques (empathy, scarcity, social proof) effective?
4. Clarity (0-20): Is it concise, clear, and actionable?
Total: 0-100. Higher is better.

Criteria:
- Brand Alignment: Reflects Vasavi’s bold, authentic ethos?
- Trust-Building: For trust intents, does it reassure with quality, support, or proof?
- Actionability: Guides customer to a sale or next step?

Query: {query}
Intent: {intent}
Keywords: {keywords}
Responses:
{responses}

Output:
- Index: X, Score: Y, Breakdown: [Relevance: A, Professionalism: B, Persuasiveness: C, Clarity: D], Comment: "Z"
"""
                formatted_responses = "\n".join([f"Response {i}: {c.response}" for i, c in enumerate(candidates)])
                eval_response = self.openai_client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {
                            "role": "system",
                            "content": eval_prompt.format(
                                query=context.query,
                                intent=context.intent,
                                keywords=", ".join(context.keywords),
                                responses=formatted_responses
                            )
                        },
                        {"role": "user", "content": "Evaluate and score these responses."}
                    ],
                    temperature=0.2,
                    max_tokens=800
                )
                eval_text = eval_response.choices[0].message.content.strip()
                best_response = candidates[0]
                best_score = 0.0
                for line in eval_text.split("\n"):
                    if "Index:" in line:
                        try:
                            parts = line.split(", ")
                            index = int(parts[0].split(": ")[1])
                            score = float(parts[1].split(": ")[1])
                            comment = parts[-1].split(": ")[1].strip('"')
                            candidates[index].rerank_score = score / 100.0
                            candidates[index].metadata["eval_comment"] = comment
                            if score > best_score:
                                best_score = score
                                best_response = candidates[index]
                        except Exception as e:
                            logger.warning(f"Error parsing evaluation: {str(e)}")
                if best_response.rerank_score < CONFIDENCE_THRESHOLD:
                    logger.debug(f"Best rerank score {best_response.rerank_score} below threshold {CONFIDENCE_THRESHOLD}")
                    best_response = self._fallback_response(context)
                logger.debug(f"Best response: Score {best_response.rerank_score}, Intent: {best_response.intent}")
                return best_response
        except Exception as e:
            logger.error(f"Error in reranking: {str(e)}")
            return self._fallback_response(context)

    def _fallback_response(self, context: QueryContext) -> QueryResponse:
        """Generate a fallback response for low-quality or failed cases"""
        return QueryResponse(
            response="My sincere apologies, I’m unable to provide a perfect answer right now. Please contact Support@vasavi.co for expert guidance, or let me know how else I can assist! 🌟",
            intent="error",
            confidence=0.1,
            rerank_score=0.0,
            metadata={"fallback_reason": "Low quality or error in generation"}
        )

    def handle_intent(self, context: QueryContext) -> Optional[QueryResponse]:
        """Robust intent handling with persuasive, trust-building responses"""
        q = context.cleaned_query
        if not q:
            return QueryResponse(
                response="Pardon me, I didn’t catch that. How may I assist with Vasavi’s bold streetwear today? 🌟",
                intent="error",
                confidence=1.0,
                rerank_score=0.0,
                metadata={"reason": "empty_query"}
            )

        # Greeting
        greeting_words = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon", "good evening"]
        if any(w in q for w in greeting_words):
            return QueryResponse(
                response=f"{self._get_greeting()} Just imagine curating a standout look with Vasavi! 🌟",
                intent="greeting",
                confidence=1.0,
                rerank_score=0.0,
                tags=["Jones: Magical Phrasing"],
                metadata={"category": "engagement"}
            )

        # Farewell
        farewell_words = ["bye", "goodbye", "tata", "see you", "farewell", "later"]
        if any(w in q for w in farewell_words):
            return QueryResponse(
                response=self._get_farewell(),
                intent="farewell",
                confidence=1.0,
                rerank_score=0.0,
                metadata={"category": "closure"}
            )

        # About
        about_words = ["about", "brand", "what is vasavi", "tell me about you", "who are you"]
        if any(w in q for w in about_words):
            return QueryResponse(
                response=f"{VASAVI_INFO['about']} Just imagine wearing a piece that tells your unique story—ready to explore our collections? 🌟 [#JonesPhrasing][#Liking]",
                intent="about",
                tags=["Jones: Magical Phrasing", "Cialdini: Liking"],
                confidence=1.0,
                rerank_score=0.0,
                metadata={"category": "brand_info"}
            )

        # Contact
        contact_words = ["contact", "email", "phone", "address", "reach you", "get in touch"]
        if any(w in q for w in contact_words):
            c = VASAVI_INFO["contact"]
            return QueryResponse(
                response=f"Delighted to connect! 📧 Email: {c['email']} | 📞 Phone: {c['phone']} | 🏠 Address: {c['address']} We’re here to craft your perfect Vasavi experience—how may I assist? 🌟 [#Liking]",
                intent="contact",
                tags=["Cialdini: Liking"],
                confidence=1.0,
                rerank_score=0.0,
                metadata={"category": "contact"}
            )

        # Shipping
        shipping_words = ["ship", "delivery", "shipping", "when arrive", "how long to ship", "postage", "international", "deliveries"]
        if any(w in q for w in shipping_words):
            s = VASAVI_INFO["shipping"]
            response = f"🚚 Domestic: {s['domestic']}\n🌍 International: {s['international']}\n🎁 Free Shipping: {s['free_shipping']} We’re thrilled to deliver Vasavi’s bold style worldwide—ready to make it yours? 🌟 [#Liking][#SPINSelling]"
            return QueryResponse(
                response=response,
                intent="shipping",
                tags=["Cialdini: Liking", "SPIN: Need-payoff"],
                confidence=1.0,
                rerank_score=0.0,
                metadata={"category": "logistics"}
            )

        # Returns
        returns_words = ["return", "exchange", "refund", "policy", "returns", "exchanges"]
        if any(w in q for w in returns_words):
            return QueryResponse(
                response="I’m deeply sorry for any concern! 😔 Returns and exchanges are seamless—unworn items within 15 days via our Product Returns Portal. May I guide you through, or address any worries? 🌟 [#VossEmpathy][#SPINSelling]",
                intent="returns",
                tags=["Voss: Empathy", "SPIN: Need-payoff"],
                confidence=1.0,
                rerank_score=0.0,
                metadata={"category": "returns"}
            )

        # Social
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
                    response=f"{' | '.join(response_parts)} Follow us for exclusive style inspiration and updates! 🌟 Ready to join the Vasavi vibe? [#Liking][#SPINSelling]",
                    intent="social",
                    tags=["Cialdini: Liking", "SPIN: Need-payoff"],
                    confidence=1.0,
                    rerank_score=0.0,
                    metadata={"category": "social"}
                )

        # Support
        support_words = ["human", "agent", "support", "representative", "want to speak to someone"]
        if any(w in q for w in support_words):
            sup = VASAVI_INFO["support"]
            return QueryResponse(
                response=f"Always at your service! 📧 Email: {sup['support_email']} | 📞 Phone: {sup['phone']} | 💬 Support: {sup['support']} We’re dedicated to your Vasavi journey—how may I elevate it? 🌟 [#Liking][#SPINSelling]",
                intent="human_support",
                tags=["Cialdini: Liking", "SPIN: Need-payoff"],
                confidence=1.0,
                rerank_score=0.0,
                metadata={"category": "support"}
            )

        # Terms
        terms_words = ["terms", "conditions", "policy", "legal"]
        if any(w in q for w in terms_words):
            tc = VASAVI_INFO["terms_and_conditions"]
            response = "Our terms, designed with you in mind:\n"
            for key, value in tc.items():
                response += f"- {key.replace('_', ' ').title()}: {value}\n"
            response += "We’re committed to a seamless experience—any questions? 🌟 [#Liking][#SPINSelling]"
            return QueryResponse(
                response=response,
                intent="terms_and_conditions",
                tags=["Cialdini: Liking", "SPIN: Need-payoff"],
                confidence=0.9,
                rerank_score=0.0,
                metadata={"category": "legal"}
            )

        # Objections
        for objection, data in OBJECTIONS.items():
            if objection in q:
                response_data = random.choice(data["response"])
                response_text = response_data["text"]
                return QueryResponse(
                    response=response_text,
                    tags=response_data.get("techniques", []),
                    intent="objection",
                    confidence=0.95,
                    rerank_score=0.0,
                    follow_up=response_data.get("follow_up"),
                    metadata={"objection_type": objection, "category": "objection"}
                )

        # Product or Trust
        product_words = ["jacket", "sashimi", "shirt", "pants", "collection", "item", "product"]
        trust_words = ["trust", "bad experience", "reliable", "quality", "why should i", "past experience", "worry"]
        if any(w in q for w in product_words) or any(w in q for w in trust_words):
            return QueryResponse(
                response="Allow me to craft a world-class solution for you! 🌟 One moment while I tailor the perfect insights...",
                intent="complex_query",
                confidence=0.8,
                rerank_score=0.0,
                metadata={"category": "product_or_trust"}
            )

        return QueryResponse(
            response="Allow me to craft a world-class solution for you! 🌟 One moment while I tailor the perfect insights...",
            intent="complex_query",
            confidence=0.6,
            rerank_score=0.0,
            metadata={"category": "general"}
        )

    def _get_greeting(self) -> str:
        """Generates a sophisticated, engaging greeting"""
        greetings = [
            "Welcome, style visionary! 👑 How may I elevate your Vasavi journey today? 🌟",
            "Greetings, fashion pioneer! 😊 Ready to craft a bold, authentic look with Vasavi’s streetwear?",
            "Hello, trendsetter extraordinaire! ✨ What masterpiece shall we create for your wardrobe today?",
            "Good day, style connoisseur! 🚀 Excited to explore Vasavi’s daring designs—where shall we begin?",
            ">Welcome to Vasavi’s elite circle! 👋 How can I help you wear your story with pride? 🌟"
        ]
        return random.choice(greetings)

    def _get_farewell(self) -> str:
        """Generates a polished, encouraging farewell"""
        farewells = [
            "Farewell, style maestro! 🌟 Stay bold and return for Vasavi’s next masterpiece!",
            "Until we meet again! 😊 Keep shining in your unique Vasavi style!",
            "Goodbye for now, fashion trailblazer! ✌️ Eager to style you soon at Vasavi!",
            "Take care, bold creator! 🚀 Wear your story with pride—we’re here for you!",
            "Signing off with elegance! 🌈 Reach out anytime for Vasavi’s signature looks!"
        ]
        return random.choice(farewells)

    def _get_help_prompt(self) -> str:
        """Generates a detailed, professional help prompt"""
        return (
            "As Vasavi’s Elite Sales Agent, I’m your guide to unparalleled style! 🌟 Ask about our bold collections, "
            "shipping logistics, seamless returns, or anything else. How may I craft your perfect Vasavi experience today?"
        )

    async def respond(self, query: str) -> QueryResponse:
        """Main method to process queries with precision and persuasion"""
        try:
            start_time = time.time()
            context = self._preprocess_query(query)
            intent_response = self.handle_intent(context)
            if intent_response.intent != "complex_query":
                intent_response.metadata["processing_time"] = time.time() - start_time
                return intent_response
            retrieved = await self.retrieve_knowledge(context.query, context.keywords)
            candidates = await self.generate_response_candidates(context, retrieved)
            best_response = await self._evaluate_and_rerank(context, candidates)
            best_response.metadata["processing_time"] = time.time() - start_time
            best_response.metadata["retrieved_docs"] = len(retrieved)
            best_response.metadata["query_keywords"] = context.keywords
            return best_response
        except Exception as e:
            logger.critical(f"Error responding to query: {str(e)}")
            return QueryResponse(
                response="My sincere apologies, I’m unable to assist right now. Please contact Support@vasavi.co for expert guidance! 🌟",
                intent="error",
                confidence=0.1,
                rerank_score=0.0,
                metadata={"error": str(e), "processing_time": time.time() - start_time}
            )

class SalesAgentTool(BaseTool):
    name: str = "sales_agent_tool"
    description: str = "Advanced LangChain tool for Vasavi's Elite Sales Agent, optimized for LangGraph integration."

    def _run(self, query: str) -> Dict[str, Any]:
        """Synchronous run method for BaseTool"""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self._arun(query))

    async def _arun(self, query: str) -> Dict[str, Any]:
        """Asynchronous run method with comprehensive output"""
        try:
            agent = SalesAgent()
            result = await agent.respond(query)
            return {
                "output": result.response,
                "sources": result.sources,
                "tags": result.tags,
                "intent": result.intent,
                "confidence": result.confidence,
                "rerank_score": result.rerank_score,
                "follow_up": result.follow_up,
                "metadata": result.metadata
            }
        except Exception as e:
            logger.critical(f"Error in sales_agent_tool: {str(e)}")
            return {
                "output": "My sincere apologies, I’m unable to assist right now. Please contact Support@vasavi.co for expert guidance! 🌟",
                "intent": "error",
                "confidence": 0.1,
                "rerank_score": 0.0,
                "metadata": {"error": str(e)}
            }

# Instantiate the tool
sales_agent_tool = SalesAgentTool()

if __name__ == "__main__":
    async def main():
        test_queries = [
            # "i am getting frustated, i want to return clothes",
            # "i am curious, whats vasavi all about.",
            # "why should i shop from vasavi, why not other brands.",
            # "do you do international deliveries.",
             "I like the SASHIMI JACKET, but I’ll think about it later.",
            #"I had a bad experience with online shopping; why should I trust Vasavi?."
        ]
        for query in test_queries:
            result = await sales_agent_tool._arun(query)
            print(f"Query: {query}")
            print(f"Response: {result['output']}")
            print(f"Sources: {result['sources']}")
            print(f"Tags: {result['tags']}")
            print(f"Intent: {result['intent']}")
            print(f"Confidence: {result['confidence']}")
            print(f"Rerank Score: {result['rerank_score']}")
            print(f"Follow-Up: {result['follow_up']}")
            print(f"Metadata: {result['metadata']}")
            print("-" * 80)

    asyncio.run(main())