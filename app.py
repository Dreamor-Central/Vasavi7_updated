from typing import TypedDict, Annotated, Literal, Optional, Dict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
import logging
import os
import re
import uuid
import uvicorn
import mysql.connector
from mysql.connector import Error
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from sales import SalesAgentTool
from semanticrag import semantic_rag
from styling import StylingAgent
from difflib import SequenceMatcher

# === Logging Configuration ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("supervisor.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# === Load environment variables ===
load_dotenv()
required_env_vars = ["TAVILY_API_KEY", "OPENAI_API_KEY", "PINECONE_API_KEY"]
for var in required_env_vars:
    if not os.getenv(var):
        logger.error(f"Environment variable '{var}' is missing.")
        raise ValueValueError(f"Environment variable '{var}' is required.")

# === Database Configuration ===
def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host="localhost",
            database="chat_history",
            user="root",
            password="mysecretpassword"
        )
        if connection.is_connected():
            logger.info("Successfully connected to the MySQL database")
            return connection
    except Error as e:
        logger.error(f"Error connecting to MySQL database: {e}")
        return None

def store_chat_log(connection, session_id, user_message, response, intent, products=None, categories=None, sentiment=0.5, keywords=None):
    try:
        cursor = connection.cursor()
        query = """
        INSERT INTO chat_log (timestamp, session_id, user_message, response, intent, products, categories, sentiment, keywords, returns_count, exchanges_count)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        timestamp = datetime.now()
        response_json = json.dumps(response) if isinstance(response, dict) else response
        cursor.execute(query, (
            timestamp, session_id, user_message, response_json, intent,
            products, categories, sentiment, keywords, 0, 0
        ))
        connection.commit()
        logger.info(f"Stored chat log for session_id: {session_id}, intent: {intent}")
    except Error as e:
        logger.error(f"Error storing chat log: {e}")
    finally:
        cursor.close()

# === Initialize memory saver for checkpointing ===
memory = MemorySaver()

# === Define state schema ===
class State(TypedDict):
    messages: Annotated[list, "add_messages"]
    intent: Optional[Literal["sales", "recommendation", "styling", "trend", "greeting", "farewell", "error"]]
    session_id: Optional[str]
    agent_output: Optional[Dict]
    context: Optional[Dict]

# === Initialize LLM ===
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4, max_retries=3, streaming=False)

# === Load Tools ===
sales_agent_tool = SalesAgentTool()

@tool
async def recommendation_agent_tool(query: str) -> dict:
    """Provides product recommendations for Vasavi clothing."""
    try:
        # Normalize product terms
        product_map = {
            "jackets": "Jacket", "jacket": "Jacket", "jakets": "Jacket",
            "shirts": "Shirt", "shirt": "Shirt",
            "t-shirts": "T-Shirt", "tshirt": "T-Shirt", "tshirts": "T-Shirt", "tee": "T-Shirt", "tees": "T-Shirt",
            "hoodie": "Hoodie", "hoodies": "Hoodie",
            "corset": "Corset", "corsets": "Corset",
            "bodysuit": "Bodysuit", "bodysuits": "Bodysuit",
            "bottoms": "Bottoms", "jeans": "Jeans", "jean": "Jeans", "pants": "Bottoms", "trousers": "Bottoms"
        }
        query_lower = query.lower()
        category = None
        for term, mapped in product_map.items():
            if term in query_lower:
                category = mapped
                break

        logger.debug(f"Recommendation query: '{query}', Detected category: {category}")

        # Call semantic_rag with normalized category
        result = await semantic_rag(query, category=category)
        if not result or not isinstance(result, list) or (result and isinstance(result[0], dict) and result[0].get("intent") == "general"):
            logger.warning(f"No valid recommendations for query: {query}, category: {category}")
            fallback_suggestions = "Check out 'hoodies', 't-shirts', or 'jeans' for some dope Vasavi gear! 😎"
            return {
                "output": f"No {category.lower() if category else 'products'} found. {fallback_suggestions} 🌟",
                "intent": "error",
                "confidence": 0.1,
                "metadata": {"source": "recommendation_agent", "error": "No results", "category": category}
            }

        formatted_output = "\n".join([
            f"**{item['style_name']} (₹{item.get('price', 'N/A')})**\n"
            f"- Category: {item.get('category', 'N/A')}, Fabric: {item.get('fabric', 'N/A')}\n"
            f"- {item.get('description', 'No description')}\n"
            f"- [Product Link]({item.get('product_link', '#')})"
            for item in result
        ])
        return {
            "output": f"Check out these bold Vasavi pieces to slay your look! 🔥\n{formatted_output}",
            "intent": "recommendation",
            "confidence": 0.95,
            "metadata": {"source": "recommendation_agent", "recommendations": result, "category": category}
        }
    except Exception as e:
        logger.error(f"Recommendation agent error: {str(e)}", exc_info=True)
        return {
            "output": "Oops, something went wrong. Try again or contact Support@vasavi.co! 😎",
            "intent": "error",
            "confidence": 0.1,
            "metadata": {"error": str(e), "category": category}
        }

@tool
async def styling_agent_tool(query: str, recommendations: Optional[str] = None) -> dict:
    """Provides styling advice for Vasavi products."""
    try:
        stylist = StylingAgent()
        rec_list = json.loads(recommendations) if recommendations else []
        result = await stylist.get_styling_advice(
            user_input=query,
            recommendations=rec_list,
            user_data={"occasion": "casual", "vibe": "streetwear"}
        )
        if result["success"]:
            output = f"Here’s how to rock your Vasavi drip! 😎\n{json.dumps(result['styling_advice'], indent=2)}"
            return {
                "output": output,
                "intent": "styling",
                "confidence": 0.9,
                "metadata": {"source": "styling_agent", "styling_advice": result["styling_advice"]}
            }
        else:
            return {
                "output": f"Sorry, I couldn't generate styling advice: {result['error']}. Try asking for specific outfit ideas! 🌟",
                "intent": "error",
                "confidence": 0.1,
                "metadata": {"error": result["error"]}
            }
    except Exception as e:
        logger.error(f"Styling agent error: {str(e)}", exc_info=True)
        return {
            "output": "Sorry, I couldn't generate styling advice. Try again or contact Support@vasavi.co! 🌟",
            "intent": "error",
            "confidence": 0.1,
            "metadata": {"error": str(e)}
        }

@tool
async def trend_agent_tool(query: str) -> dict:
    """Fetches the latest fashion trends or general web search results."""
    try:
        tavily = TavilySearchResults(max_results=5)
        response = await tavily.ainvoke({"query": f"latest streetwear trends 2025 {query}"})
        results = [r.get("content", "") for r in response if isinstance(r, dict) and r.get("content")]
        if not results:
            return {
                "output": "Couldn't fetch the latest trends right now. Try asking for specific styles like 'streetwear trends'! 🌟",
                "intent": "error",
                "confidence": 0.1,
                "metadata": {"source": "trend_agent", "error": "No results"}
            }
        output = f"What's hot in streetwear? 🔥 Here's the latest:\n" + "\n".join(f"- {r[:200]}..." for r in results)
        return {
            "output": output,
            "intent": "trend",
            "confidence": 0.85,
            "metadata": {"source": "trend_agent", "search_results": results}
        }
    except Exception as e:
        logger.error(f"Trend agent error: {str(e)}", exc_info=True)
        return {
            "output": "Sorry, I couldn't fetch trend info. Try again or contact Support@vasavi.co! 🌟",
            "intent": "error",
            "confidence": 0.1,
            "metadata": {"error": str(e)}
        }

# === Bind tools to LLM ===
tools = [recommendation_agent_tool, sales_agent_tool, styling_agent_tool, trend_agent_tool]
llm_with_tools = llm.bind_tools(tools)

# === Supervisor System Prompt ===
supervisor_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are the supervisor for Vasavi's AI sales system, a premium streetwear e-commerce platform. Your role is to analyze user queries, detect intent, and route them to the appropriate agent using ReAct-style reasoning. Queries may be lowercase, uppercase, or grammatically incorrect. Use conversation history and context to ensure seamless follow-ups, maintaining Vasavi’s bold, trendy vibe.

Intents and tools:
- sales: Brand info, shipping, returns, objections, trust concerns, greetings, farewells -> sales_agent_tool
- recommendation: Product recommendations (category, price, style, e.g., 'jackets', 'jeans') -> recommendation_agent_tool
- styling: Styling advice, accessory pairings (e.g., 'how to style a jacket') -> styling_agent_tool
- trend: Fashion trends, general questions, or web search (e.g., 'what's trending') -> trend_agent_tool
- greeting: Simple greetings (e.g., 'hello') -> sales_agent_tool
- farewell: Farewell messages (e.g., 'bye') -> sales_agent_tool
- error: Unclear or unhandled queries -> sales_agent_tool

Reasoning steps:
1. Parse query for keywords, sentiment, context:
   - Keywords: 'expensive', 'trust', 'shipping' -> sales; 'jeans', 'jackets', 'suggest', 'under ₹X' -> recommendation; 'style', 'pairing', 'accessories' -> styling; 'trend', 'latest fashion' -> trend; 'hi', 'hello' -> greeting; 'bye' -> farewell
   - Sentiment: Frustration -> sales; Curiosity -> recommendation/styling/trend
   - Context: Check history (last 5 messages) and context (e.g., recent recommendations)
2. Assign intent with confidence:
   - High confidence (0.9+): Clear keyword matches
   - Medium confidence (0.5-0.9): Ambiguous but likely intent
   - Low confidence (<0.5): Unclear queries -> error
3. Select tool or respond directly, justifying reasoning.
4. For follow-ups, pass context to styling or sales tools.
5. Maintain Vasavi’s urban vibe (e.g., 'drip', 'slay', '🔥').

Output JSON in this exact format:
```json
{
  "intent": "sales|recommendation|styling|trend|greeting|farewell|error",
  "tool": "sales_agent_tool|recommendation_agent_tool|styling_agent_tool|trend_agent_tool|null",
  "reasoning": "Explanation of intent and tool selection",
  "confidence": 0.0
}
```
Ensure the response is valid JSON with properly escaped quotes.
"""),
    ("user", "Query: {query}\nHistory: {history}\nContext: {context}")
])

# Fallback prompt if primary prompt fails
fallback_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are the supervisor for Vasavi's AI sales system. Analyze the user query and select the appropriate intent and tool. Queries may be lowercase, uppercase, or grammatically incorrect. Return a JSON response with intent, tool, reasoning, and confidence. Maintain a bold, trendy tone.

Intents: sales, recommendation, styling, trend, greeting, farewell, error
Tools: sales_agent_tool, recommendation_agent_tool, styling_agent_tool, trend_agent_tool, null

Output JSON in this exact format:
```json
{
  "intent": "sales|recommendation|styling|trend|greeting|farewell|error",
  "tool": "sales_agent_tool|recommendation_agent_tool|styling_agent_tool|trend_agent_tool|null",
  "reasoning": "Explanation of intent and tool selection",
  "confidence": 0.0
}
```
Ensure the response is valid JSON with properly escaped quotes.
"""),
    ("user", "Query: {query}\nHistory: {history}\nContext: {context}")
])

# Input Validation
def validate_query(query: str) -> bool:
    """Validates query to prevent empty or malicious inputs."""
    if not query or not query.strip():
        logger.warning("Validation failed: Query is empty or whitespace only.")
        return False
    if re.search(r"[<>{};'\"]", query):
        logger.warning(f"Validation failed: Query contains potentially malicious characters: '{query}'")
        return False
    if len(query) > 500:
        logger.warning(f"Validation failed: Query too long ({len(query)} chars). Max 500 chars.")
        return False
    return True

def fuzzy_match(word: str, keywords: list, threshold: float = 0.8) -> Optional[str]:
    """Returns the closest matching word from keywords if similarity exceeds threshold."""
    for kw in keywords:
        similarity = SequenceMatcher(None, word.lower(), kw.lower()).ratio()
        if similarity >= threshold:
            return kw
    return None

async def supervisor_agent(state: State) -> State:
    """Determines intent and routes to the appropriate agent."""
    try:
        if not state.get("messages") or not isinstance(state["messages"], list) or not state["messages"]:
            logger.warning("Empty or invalid messages in state.")
            state["intent"] = "error"
            state["agent_output"] = {
                "output": "I couldn't understand your message. Please try again! 🌟",
                "intent": "error",
                "confidence": 1.0
            }
            state["messages"] = state.get("messages", []) + [AIMessage(content=state["agent_output"]["output"])]
            return state

        query = state["messages"][-1].content.strip()
        if not validate_query(query):
            logger.warning(f"Invalid query: '{query}'")
            state["intent"] = "error"
            state["agent_output"] = {
                "output": "That query seems invalid or too complex. Please rephrase and keep it simple! 🌟",
                "intent": "error",
                "confidence": 1.0
            }
            state["messages"].append(AIMessage(content=state["agent_output"]["output"]))
            return state

        # Normalize query for intent detection
        query_lower = query.lower()
        words = re.findall(r'\w+', query_lower)
        logger.debug(f"Normalized query: '{query_lower}', Words: {words}")

        # Expanded keyword lists with common misspellings
        styling_keywords = [
            "style", "styling", "stylng", "pair", "pairing", "match", "matching", "outfit ideas", "how to wear",
            "accessories", "accessory", "jewelry", "jewellery", "belt", "bag", "shoes", "sneakers", "boots"
        ]
        recommendation_keywords = [
            "suggest", "sugest", "recommend", "recomend", "reccomend", "show me", "showme", "find me", "findme",
            "jackets", "jacket", "jakets", "shirts", "shirt", "t-shirts", "tshirt", "tshirts", "tee", "tees",
            "hoodie", "hoodies", "corset", "corsets", "bodysuit", "bodysuits", "bottoms", "jeans", "jean",
            "pants", "trousers", "clothing", "clothes", "apparel", "outfit", "outfits", "fashion"
        ]
        sales_keywords = [
            "expensive", "costly", "trust", "reliable", "return", "returns", "refund", "shipping", "delivery",
            "about", "brand", "vasavi", "contact", "support", "help", "terms", "policy", "policies",
            "pricey", "cost", "order", "track", "tracking"
        ]
        trend_keywords = [
            "trend", "trends", "trending", "fashion trend", "latest fashion", "new fashion", "hot", "popular",
            "in style", "2025 fashion", "streetwear trends", "whats new", "what's new"
        ]
        greeting_keywords = [
            "hi", "hii", "hello", "hey", "heyy", "greetings", "good morning", "good afternoon", "good evening"
        ]
        farewell_keywords = [
            "bye", "goodbye", "see you", "seeya", "farewell", "later", "take care"
        ]

        # Keyword-based intent detection with fuzzy matching
        intent, tool, confidence = "error", "sales_agent_tool", 0.1
        reasoning = "Fallback to keyword-based intent detection"
        matched_keywords = []

        # Prioritize styling keywords to avoid misrouting
        for word in words:
            if matched := fuzzy_match(word, styling_keywords):
                intent, tool, confidence = "styling", "styling_agent_tool", 0.9
                matched_keywords.append(matched)
                break  # Exit early to prioritize styling
        if not matched_keywords:
            for word in words:
                if matched := fuzzy_match(word, recommendation_keywords):
                    intent, tool, confidence = "recommendation", "recommendation_agent_tool", 0.9
                    matched_keywords.append(matched)
                elif matched := fuzzy_match(word, sales_keywords):
                    intent, tool, confidence = "sales", "sales_agent_tool", 0.9
                    matched_keywords.append(matched)
                elif matched := fuzzy_match(word, trend_keywords):
                    intent, tool, confidence = "trend", "trend_agent_tool", 0.9
                    matched_keywords.append(matched)
                elif matched := fuzzy_match(word, greeting_keywords):
                    intent, tool, confidence = "greeting", "sales_agent_tool", 1.0
                    matched_keywords.append(matched)
                elif matched := fuzzy_match(word, farewell_keywords):
                    intent, tool, confidence = "farewell", "sales_agent_tool", 1.0
                    matched_keywords.append(matched)

        if matched_keywords:
            reasoning = f"Detected {intent} intent due to keywords: {', '.join(matched_keywords)}"

        history = "\n".join([f"{m.type}:{m.content}" for m in state["messages"][:-1][-5:]])
        context = json.dumps(state.get("context", {}))
        logger.debug(f"Input: Query='{query}', Normalized='{query_lower}', History='{history}', Context='{context}'")

        # Try LLM-based intent detection with retry
        parser = JsonOutputParser(pydantic_object=type('Decision', (), {
            "intent": str,
            "tool": Optional[str],
            "reasoning": str,
            "confidence": float
        }))
        for attempt in range(3):
            try:
                prompt = supervisor_prompt
                llm_response = await llm_with_tools.ainvoke(prompt.format(query=query, history=history, context=context))
                response_content = llm_response.content
                logger.debug(f"LLM Response (attempt {attempt + 1}): {response_content}")
                decision = parser.parse(response_content)
                intent = decision.get("intent", intent)
                tool = decision.get("tool", tool)
                reasoning = decision.get("reasoning", reasoning)
                confidence = float(decision.get("confidence", confidence))
                break
            except (json.JSONDecodeError, OutputParserException) as e:
                logger.warning(f"LLM JSON parsing error on attempt {attempt + 1}: {str(e)}. Response: {response_content}")
                if attempt == 2:
                    logger.error(f"Failed to parse JSON after 3 attempts. Using keyword-based intent: {intent}")
                    reasoning = f"LLM failed after 3 attempts: {str(e)}. Used keyword-based intent: {intent}"
            except Exception as e:
                logger.warning(f"Prompt error on attempt {attempt + 1}: {str(e)}. Trying fallback prompt.")
                prompt = fallback_prompt
                if attempt == 2:
                    logger.error(f"Failed to process prompt after 3 attempts. Using keyword-based intent: {intent}")
                    reasoning = f"Prompt failed after 3 attempts: {str(e)}. Used keyword-based intent: {intent}"

        logger.info(f"Decision: Intent={intent}, Tool={tool}, Confidence={confidence:.2f}, Reasoning={reasoning}")
        state["intent"] = intent

        # Extract products and categories for chat_log
        products = None
        categories = None
        if intent == "recommendation":
            recommendations = state.get("context", {}).get("recent_recommendations", [])
            if recommendations:
                products = ", ".join([item.get("style_name", "") for item in recommendations])
                categories = ", ".join([item.get("category", "") for item in recommendations])

        keywords_str = ", ".join(matched_keywords) if matched_keywords else None

        tool_result = None
        if tool:
            if tool == "styling_agent_tool":
                recommendations_json = json.dumps(state.get("context", {}).get("recent_recommendations", []))
                tool_result = await styling_agent_tool.ainvoke({"query": query, "recommendations": recommendations_json})
            elif tool == "recommendation_agent_tool":
                tool_result = await recommendation_agent_tool.ainvoke({"query": query})
                if tool_result and tool_result["intent"] == "recommendation" and tool_result["metadata"].get("recommendations"):
                    state["context"] = state.get("context", {})
                    state["context"]["recent_recommendations"] = tool_result["metadata"]["recommendations"]
                    products = ", ".join([item.get("style_name", "") for item in tool_result["metadata"]["recommendations"]])
                    categories = tool_result["metadata"].get("category")
            elif tool == "trend_agent_tool":
                tool_result = await trend_agent_tool.ainvoke({"query": query})
            elif tool == "sales_agent_tool":
                tool_result = await sales_agent_tool.ainvoke({"query": query})
            else:
                logger.warning(f"Unknown tool '{tool}'. Using sales agent.")
                tool_result = await sales_agent_tool.ainvoke({"query": query})
        else:
            logger.info("No tool specified. Using sales agent.")
            tool_result = await sales_agent_tool.ainvoke({"query": query})

        if tool_result:
            state["agent_output"] = tool_result
            output_content = tool_result.get("output", "No output from tool.")
            if isinstance(output_content, dict):
                output_content = json.dumps(output_content, indent=2)
            state["messages"].append(AIMessage(content=output_content))
            # Store in chat_log
            connection = get_db_connection()
            if connection:
                try:
                    store_chat_log(
                        connection=connection,
                        session_id=state["session_id"],
                        user_message=query,
                        response=tool_result,
                        intent=intent,
                        products=products,
                        categories=categories,
                        sentiment=0.5,  # Default, as sentiment analysis not implemented
                        keywords=keywords_str
                    )
                finally:
                    connection.close()
        else:
            logger.warning("Tool returned no result.")
            state["agent_output"] = {
                "output": "The tool failed to respond. Please try again or contact Support@vasavi.co! 🌟",
                "intent": "error",
                "confidence": 0.1
            }
            state["messages"].append(AIMessage(content=state["agent_output"]["output"]))
            # Store in chat_log
            connection = get_db_connection()
            if connection:
                try:
                    store_chat_log(
                        connection=connection,
                        session_id=state["session_id"],
                        user_message=query,
                        response=state["agent_output"],
                        intent="error",
                        products=None,
                        categories=None,
                        sentiment=0.5,
                        keywords=keywords_str
                    )
                finally:
                    connection.close()

        return state
    except Exception as e:
        logger.error(f"Supervisor error: {str(e)}", exc_info=True)
        state["intent"] = "error"
        state["agent_output"] = {
            "output": "Something went wrong. Please try again or contact Support@vasavi.co! 🌟",
            "intent": "error",
            "confidence": 0.1
        }
        state["messages"].append(AIMessage(content=state["agent_output"]["output"]))
        # Store in chat_log
        connection = get_db_connection()
        if connection:
            try:
                store_chat_log(
                    connection=connection,
                    session_id=state["session_id"],
                    user_message=query,
                    response=state["agent_output"],
                    intent="error",
                    products=None,
                    categories=None,
                    sentiment=0.5,
                    keywords=None
                )
            finally:
                connection.close()
        return state

async def tool_node(state: State) -> State:
    """Executes tool calls specified by the supervisor."""
    try:
        last_message = state["messages"][-1]
        if not isinstance(last_message, AIMessage):
            logger.warning("Tool node called with non-AIMessage as last message.")
            state["intent"] = "error"
            state["agent_output"] = {"output": "System error: Invalid message format.", "intent": "error", "confidence": 0.1}
            state["messages"].append(AIMessage(content=state["agent_output"]["output"]))
            # Store in chat_log
            connection = get_db_connection()
            if connection:
                try:
                    store_chat_log(
                        connection=connection,
                        session_id=state["session_id"],
                        user_message=state["messages"][-2].content if len(state["messages"]) > 1 else "Unknown",
                        response=state["agent_output"],
                        intent="error",
                        products=None,
                        categories=None,
                        sentiment=0.5,
                        keywords=None
                    )
                finally:
                    connection.close()
            return state

        tool_call = last_message.tool_calls[0] if last_message.tool_calls else None
        if not tool_call:
            logger.info("No tool calls in last message. Assuming direct supervisor response.")
            return state

        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_id = tool_call.get("id", str(uuid.uuid4()))

        logger.info(f"Executing tool: {tool_name} with args: {tool_args}")

        result = None
        products = None
        categories = None
        keywords = ", ".join([word for word in re.findall(r'\w+', tool_args.get("query", "").lower()) if fuzzy_match(word, 
            ["style", "jacket", "jeans", "trend", "shipping", "returns", "hi", "bye"], 0.8)])

        if tool_name == "sales_agent_tool":
            result = await sales_agent_tool.ainvoke(tool_args)
        elif tool_name == "recommendation_agent_tool":
            result = await recommendation_agent_tool.ainvoke(tool_args)
            if result and result.get("metadata", {}).get("recommendations"):
                state["context"] = state.get("context", {})
                state["context"]["recent_recommendations"] = result["metadata"]["recommendations"]
                products = ", ".join([item.get("style_name", "") for item in result["metadata"]["recommendations"]])
                categories = result["metadata"].get("category")
        elif tool_name == "styling_agent_tool":
            tool_args_for_styling = tool_args.copy()
            if "recommendations" not in tool_args_for_styling and state.get("context", {}).get("recent_recommendations"):
                tool_args_for_styling["recommendations"] = json.dumps(state["context"]["recent_recommendations"])
            result = await styling_agent_tool.ainvoke(tool_args_for_styling)
        elif tool_name == "trend_agent_tool":
            result = await trend_agent_tool.ainvoke(tool_args)
        else:
            logger.error(f"Unknown tool: {tool_name}")
            result = {"output": f"Unknown tool '{tool_name}'. Please try again!", "intent": "error", "confidence": 0.1}

        if result:
            state["agent_output"] = result
            tool_output_content = result.get("output", "Tool executed successfully.")
            if isinstance(tool_output_content, dict):
                tool_output_content = json.dumps(tool_output_content, indent=2)
            state["messages"].append(ToolMessage(
                content=tool_output_content,
                tool_call_id=tool_id,
                name=tool_name
            ))
            # Store tool message in chat_log
            connection = get_db_connection()
            if connection:
                try:
                    store_chat_log(
                        connection=connection,
                        session_id=state["session_id"],
                        user_message=tool_args.get("query", "Unknown"),
                        response=result,
                        intent=result.get("intent", state["intent"]),
                        products=products,
                        categories=categories,
                        sentiment=0.5,
                        keywords=keywords
                    )
                finally:
                    connection.close()
            state["intent"] = result.get("intent", state["intent"])
        else:
            logger.error(f"Tool '{tool_name}' failed silently.")
            state["agent_output"] = {"output": f"Tool '{tool_name}' failed. Please try again!", "intent": "error", "confidence": 0.1}
            state["messages"].append(AIMessage(content=state["agent_output"]["output"]))
            # Store error AI message in chat_log
            connection = get_db_connection()
            if connection:
                try:
                    store_chat_log(
                        connection=connection,
                        session_id=state["session_id"],
                        user_message=tool_args.get("query", "Unknown"),
                        response=state["agent_output"],
                        intent="error",
                        products=None,
                        categories=None,
                        sentiment=0.5,
                        keywords=keywords
                    )
                finally:
                    connection.close()
            state["intent"] = "error"

        return state
    except Exception as e:
        logger.error(f"Tool node error: {str(e)}", exc_info=True)
        state["intent"] = "error"
        state["agent_output"] = {"output": "An error occurred while processing your request. Please try again! 🌟", "intent": "error", "confidence": 0.1}
        state["messages"].append(AIMessage(content=state["agent_output"]["output"]))
        # Store error AI message in chat_log
        connection = get_db_connection()
        if connection:
            try:
                store_chat_log(
                    connection=connection,
                    session_id=state["session_id"],
                    user_message=state["messages"][-2].content if len(state["messages"]) > 1 else "Unknown",
                    response=state["agent_output"],
                    intent="error",
                    products=None,
                    categories=None,
                    sentiment=0.5,
                    keywords=None
                )
            finally:
                connection.close()
        return state

async def response_node(state: State) -> State:
    """Formats the final response to the user."""
    try:
        agent_output = state.get("agent_output", {})
        output_content = agent_output.get("output", "I'm sorry, I couldn't generate a response.")
        if isinstance(output_content, dict):
            response = json.dumps(output_content, indent=2)
        else:
            response = str(output_content)

        intent = agent_output.get("intent", state.get("intent", "error"))
        final_response = response if intent != "error" else "Sorry, something went wrong. Try again or contact Support@vasavi.co! 🌟"

        state["messages"].append(AIMessage(content=final_response))
        # Store AI message in chat_log
        connection = get_db_connection()
        if connection:
            try:
                query = state["messages"][-2].content if len(state["messages"]) > 1 else "Unknown"
                products = None
                categories = None
                if intent == "recommendation" and agent_output.get("metadata", {}).get("recommendations"):
                    products = ", ".join([item.get("style_name", "") for item in agent_output["metadata"]["recommendations"]])
                    categories = agent_output["metadata"].get("category")
                keywords = ", ".join([word for word in re.findall(r'\w+', query.lower()) if fuzzy_match(word, 
                    ["style", "jacket", "jeans", "trend", "shipping", "returns", "hi", "bye"], 0.8)])
                store_chat_log(
                    connection=connection,
                    session_id=state["session_id"],
                    user_message=query,
                    response=agent_output,
                    intent=intent,
                    products=products,
                    categories=categories,
                    sentiment=0.5,
                    keywords=keywords
                )
            finally:
                connection.close()
        state["agent_output"] = None
        return state
    except Exception as e:
        logger.error(f"Response node error: {str(e)}", exc_info=True)
        state["intent"] = "error"
        state["messages"].append(AIMessage(content="An error occurred while preparing my response. Please try again! 🌟"))
        # Store error AI message in chat_log
        connection = get_db_connection()
        if connection:
            try:
                query = state["messages"][-2].content if len(state["messages"]) > 1 else "Unknown"
                store_chat_log(
                    connection=connection,
                    session_id=state["session_id"],
                    user_message=query,
                    response={"output": "An error occurred while preparing my response. Please try again! 🌟", "intent": "error", "confidence": 0.1},
                    intent="error",
                    products=None,
                    categories=None,
                    sentiment=0.5,
                    keywords=None
                )
            finally:
                connection.close()
        state["agent_output"] = None
        return state

def router(state: State) -> str:
    """Routes to the next node based on state."""
    try:
        intent = state.get("intent", "error")
        logger.debug(f"Router intent: {intent}")

        if intent in ["greeting", "farewell", "error"]:
            logger.debug(f"Routing to END: intent={intent}")
            return END

        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            logger.debug("Routing to tool node: tool calls present.")
            return "tool"

        if state.get("agent_output"):
            logger.debug("Routing to response node: agent_output present.")
            return "response"

        logger.warning("Router fallback to END: no clear path.")
        return END
    except Exception as e:
        logger.error(f"Router error: {str(e)}", exc_info=True)
        return END

# === Build the Graph ===
graph_builder = StateGraph(State)
graph_builder.add_node("supervisor", supervisor_agent)
graph_builder.add_node("tool", tool_node)
graph_builder.add_node("response", response_node)
graph_builder.set_entry_point("supervisor")
graph_builder.add_conditional_edges(
    "supervisor",
    router,
    {"tool": "tool", "response": "response", END: END}
)
graph_builder.add_edge("tool", "response")
graph_builder.add_edge("response", END)
graph = graph_builder.compile(checkpointer=memory)

# === FastAPI Setup ===
app = FastAPI(title="Vasavi AI Salesman")

# === CORS Middleware ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://vasavi.co", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Session-ID"],
    expose_headers=["Content-Type"]
)

# === Root Endpoint ===
@app.get("/")
async def root():
    return {"message": "Welcome to Vasavi AI Salesman API. Visit /docs for Swagger UI."}

@app.get("/docs")
async def docs_redirect():
    return {"message": "Access Swagger UI at /docs or use POST /chat/stream."}

async def generate_chat_response(message: str, checkpoint_id: Optional[str] = None):
    """Streams responses from the LangGraph agent."""
    try:
        if not validate_query(message):
            yield f"data: {json.dumps({'type': 'error', 'content': 'Invalid query: Please rephrase or simplify.'})}\n\n"
            return

        current_thread_id = checkpoint_id if checkpoint_id else str(uuid.uuid4())
        if checkpoint_id:
            try:
                uuid.UUID(checkpoint_id)
                if not await memory.aget({"configurable": {"thread_id": checkpoint_id}}):
                    yield f"data: {json.dumps({'type': 'error', 'content': 'Invalid checkpoint ID: Session not found.'})}\n\n"
                    return
            except ValueError:
                yield f"data: {json.dumps({'type': 'error', 'content': 'Invalid checkpoint ID format.'})}\n\n"
                return

        # Store initial user message in chat_log
        connection = get_db_connection()
        if connection:
            try:
                store_chat_log(
                    connection=connection,
                    session_id=current_thread_id,
                    user_message=message,
                    response={"output": "Processing query..."},
                    intent="pending",
                    products=None,
                    categories=None,
                    sentiment=0.5,
                    keywords=None
                )
            finally:
                connection.close()

        config = {"configurable": {"thread_id": current_thread_id}}
        if not checkpoint_id:
            yield f"data: {json.dumps({'type': 'checkpoint', 'checkpoint_id': current_thread_id})}\n\n"
            logger.info(f"New session: {current_thread_id}")

        input_state = {
            "messages": [HumanMessage(content=message)],
            "session_id": current_thread_id,
            "context": {}
        }

        async for event in graph.astream_events(input_state, version="v1", config=config):
            event_type = event["event"]
            event_name = event.get("name", "unknown")

            if event_type == "on_chat_model_stream":
                try:
                    chunk_content = event["data"]["chunk"].content
                    if chunk_content:
                        safe_content = chunk_content.replace('"', '\\"').replace("\n", "\\n")
                        yield f"data: {json.dumps({'type': 'success', 'content': safe_content})}\n\n"
                except Exception as e:
                    logger.error(f"Error streaming chat model chunk: {str(e)}", exc_info=True)
                    continue
            elif event_type == "on_tool_start":
                try:
                    tool_name = event["name"]
                    yield f"data: {json.dumps({'type': 'tool_call', 'content': tool_name})}\n\n"
                    logger.info(f"Tool started: {tool_name}")
                except Exception as e:
                    logger.error(f"Error streaming tool start: {str(e)}", exc_info=True)
                    continue
            elif event_type == "on_tool_end":
                try:
                    output = event["data"]["output"]
                    content = output.get("output", json.dumps(output, indent=2)) if isinstance(output, dict) else str(output)
                    safe_content = content.replace('"', '\\"').replace("\n", "\\n")
                    yield f"data: {json.dumps({'type': 'content', 'content': safe_content})}\n\n"
                    logger.info(f"Tool ended: {event_name}")
                except Exception as e:
                    logger.error(f"Error streaming tool output: {str(e)}", exc_info=True)
                    continue
            elif event_type == "on_chain_end":
                logger.debug(f"Chain ended for event: {event_name}")

        yield f"data: {json.dumps({'type': 'end'})}\n\n"
        logger.info(f"Streaming completed for session: {current_thread_id}")

    except Exception as e:
        logger.error(f"Streaming error: {str(e)}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'content': 'System error. Please try again or contact Support@vasavi.co! 🌟'})}\n\n"

@app.post("/chat/stream")
async def chat_stream(
    message: str = Query(..., description="User query"),
    checkpoint_id: Optional[str] = Query(None, description="Optional checkpoint ID for session persistence")
):
    logger.info(f"Chat request: message='{message}', checkpoint_id='{checkpoint_id}'")
    return StreamingResponse(
        generate_chat_response(message, checkpoint_id),
        media_type="text/event-stream"
    )

# === Test Script ===
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
```

### Setup Instructions
1. **Replace MySQL Credentials**:
   - In `get_db_connection`, update `host`, `database`, `user`, and `password` with your actual MySQL credentials:
     ```python
     host="your_mysql_host",  # e.g., "localhost" or "mysql.example.com"
     database="chat_history",
     user="your_username",    # e.g., "root"
     password="your_password" # e.g., "mysecretpassword"
     ```

2. **Create Database and Table**:
   Run the following SQL to set up the `chat_history` database and `chat_log` table (already provided by you):
   ```sql
   CREATE DATABASE IF NOT EXISTS chat_history;
   USE chat_history;
   CREATE TABLE chat_log (
       id INT AUTO_INCREMENT PRIMARY KEY,
       timestamp DATETIME NOT NULL,
       session_id VARCHAR(36) NOT NULL,
       user_message TEXT NOT NULL,
       response JSON NOT NULL,
       intent VARCHAR(50),
       products VARCHAR(255),
       categories VARCHAR(255),
       sentiment FLOAT,
       keywords VARCHAR(255),
       returns_count INT DEFAULT 0,
       exchanges_count INT DEFAULT 0,
       INDEX idx_session_id (session_id),
       INDEX idx_intent (intent)
   );
   ```

3. **Environment Variables**:
   Ensure you have a `.env` file with the required non-MySQL variables:
   ```
   TAVILY_API_KEY=your_tavily_api_key
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   ```

4. **Dependencies**:
   Install required Python packages:
   ```bash
   pip install fastapi uvicorn mysql-connector-python langchain langchain-openai langchain-community python-dotenv
   ```
   Ensure `sales.py`, `semanticrag.py`, and `styling.py` are available and correctly implemented.

5. **Run the Application**:
   Save the code as `main.py` and run:
   ```bash
   python main.py
   ```
   The FastAPI server will start at `http://0.0.0.0:8001`. Test the `/chat/stream` endpoint:
   ```bash
   curl -X POST "http://localhost:8001/chat/stream?message=Show%20me%20some%20jackets"
   ```

6. **Test Database Connection**:
   Verify the connection with this standalone script:
   ```python
   import mysql.connector
   from mysql.connector import Error

   def test_db_connection():
       try:
           connection = mysql.connector.connect(
               host="localhost",
               database="chat_history",
               user="root",
               password="mysecretpassword"
           )
           if connection.is_connected():
               print("Connection successful!")
               cursor = connection.cursor()
               cursor.execute("SELECT DATABASE();")
               db_name = cursor.fetchone()[0]
               print(f"Connected to database: {db_name}")
               cursor.close()
               connection.close()
       except Error as e:
           print(f"Error: {e}")

   if __name__ == "__main__":
       test_db_connection()