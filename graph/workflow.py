"""
LangGraph Workflow Module
Contains the LangGraph state machine and workflow logic
"""
import os
import logging
import json
import re
from typing import List, Dict, Optional, Literal, AsyncGenerator
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph
from dotenv import load_dotenv
import asyncio

load_dotenv()

logger = logging.getLogger(__name__)

# ====================================================
# --- Initialize LLMs (Streaming and Non-Streaming) ---
# ====================================================

try:
    # Non-streaming LLM for query generation
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,
        streaming=False,
    )

    # Streaming LLM for final responses
    llm_streaming = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,
        streaming=True,
    )
    logger.info("✅ LLMs initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize LLM: {e}")
    raise


# ====================================================
# --- State Definition ---
# ====================================================

class State(TypedDict):
    input: str
    user_id: str
    messages: List[Dict[str, str]]
    user_context: Dict
    intent_type: Literal["analytics", "crm", "order_management",
                         "customer_service", "recommendation", "general", "unclear"]
    intent_details: Dict
    cypher_query: Optional[str]
    query_results: List[Dict]
    requires_action: bool
    action_type: Optional[str]
    output: str
    error: Optional[str]


# ====================================================
# --- Helper Functions ---
# ====================================================

def safe_parse_json(text: str) -> dict:
    try:
        cleaned = re.sub(r"^```json\s*|\s*```$", "",
                         text.strip(), flags=re.MULTILINE).strip()
        json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if json_match:
            cleaned = json_match.group()
        return json.loads(cleaned)
    except Exception as e:
        logger.warning(f"⚠️ JSON parse failed: {e}")
        return {}


def generate_general_response(state: State) -> str:
    """Generate response for general queries"""
    user_input = state["input"].lower()

    greetings = ["hi", "hello", "hey", "good morning", "good evening"]
    if any(g in user_input for g in greetings):
        return """👋 Hello! I'm your Restaurant CRM Assistant. I can help you with:

📊 **Analytics** - Sales reports, trends, insights
👥 **Customer Management** - Profiles, loyalty, segmentation
📦 **Order Management** - Tracking, history, status
⭐ **Customer Service** - Reviews, feedback, complaints
🍽️ **Recommendations** - Dish suggestions, popular items

What would you like to explore today?"""

    if "help" in user_input or "what can you do" in user_input:
        return """🤖 **I'm here to help with your restaurant operations!**

**Ask me things like:**
• "Show me today's top-selling dishes"
• "Which customers haven't ordered in 30 days?"
• "What are the recent negative reviews?"
• "Recommend vegan dishes for a customer"
• "Track order #12345"
• "Show me VIP customers in New York"

Just ask naturally, and I'll assist you! 😊"""

    return "I'm here to help with your restaurant CRM needs. Could you tell me more about what you'd like to know or do?"


def generate_no_results_response(state: State) -> str:
    """Generate response when no data is found"""
    intent_type = state["intent_type"]

    suggestions = {
        "analytics": "Try adjusting the time period or filters",
        "crm": "Check if the customer exists in the system",
        "order_management": "Verify the order ID or customer name",
        "customer_service": "Try broadening the search criteria",
        "recommendation": "Explore other categories or cuisines"
    }

    return f"""🔍 **No Results Found**

I searched for: {state['intent_details'].get('main_focus', 'your query')}

**Suggestions:**
• {suggestions.get(intent_type, 'Try rephrasing your question')}
• Check spelling and filters
• Use more general terms

Need help? Just ask! 😊"""


# ====================================================
# --- Node 1: Intent Classification & Routing ---
# ====================================================

def classify_intent(state: State) -> State:
    """Classify user intent into specific CRM categories"""
    logger.info("=" * 80)
    logger.info("🎯 STEP 1: CLASSIFY INTENT")
    logger.info("=" * 80)

    user_input = state["input"]
    logger.info(f"📝 User Input: {user_input}")

    recent_history = state["messages"][-4:] if state["messages"] else []
    context_str = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in recent_history])
    user_context = state["user_context"]

    logger.info(f"💬 Conversation History: {len(recent_history)} messages")
    logger.info(f"👤 User Context: {user_context}")

    prompt = f"""
You are an intelligent CRM assistant for restaurant management. Analyze the user's message and classify their intent.

**User Context:**
- Current Task: {user_context.get('current_task', 'None')}
- Last Interaction: {user_context.get('last_interaction', 'None')}
- Preferences: {json.dumps(user_context.get('preferences', {}))}

**Recent Conversation:**
{context_str}

**Current Message:**
"{user_input}"

**Intent Categories:**

1. **analytics** - Data analysis, reports, insights, trends
   Examples: "What are top dishes?", "Show revenue report", "Customer trends"

2. **crm** - Customer relationship management, loyalty, profiles
   Examples: "Who are our VIP customers?", "Find customers who haven't ordered recently", "Update customer preferences"

3. **order_management** - Orders, tracking, history, processing
   Examples: "Show recent orders", "Track order #123", "Orders for customer John"

4. **customer_service** - Complaints, feedback, reviews, support
   Examples: "Show negative reviews", "Customer complaints", "Feedback on Pizza"

5. **recommendation** - Dish suggestions, menu recommendations
   Examples: "Recommend vegan dishes", "What's popular?", "Best dishes for parties"

6. **general** - Greetings, help, capabilities, general questions
   Examples: "Hello", "What can you do?", "Help me"

7. **unclear** - Ambiguous or needs clarification

**Your Task:**
Analyze the intent and respond with JSON ONLY:

{{
  "intent_type": "<one of: analytics|crm|order_management|customer_service|recommendation|general|unclear>",
  "confidence": <0-100>,
  "intent_details": {{
    "main_focus": "<what user wants>",
    "entities": ["list", "of", "key", "entities"],
    "requires_data": <true/false>,
    "time_period": "<if mentioned: today, last_week, this_month, etc>",
    "filters": {{"type": "veg/non-veg/all", "cuisine": "if mentioned", "customer_id": "if mentioned"}}
  }},
  "suggested_clarification": "<only if unclear, what to ask user>"
}}

Be precise. Focus on the PRIMARY intent.
"""

    logger.info("🤖 Calling LLM for intent classification...")

    try:
        response = llm.invoke(prompt).content.strip()
        logger.info(f"✅ LLM Response received (length: {len(response)})")
        logger.info(f"📄 Raw LLM Response:\n{response[:500]}...")

        data = safe_parse_json(response)
        logger.info(f"✅ Parsed JSON successfully")
        logger.info(f"📊 Parsed Data: {json.dumps(data, indent=2)}")

        state["intent_type"] = data.get("intent_type", "unclear")
        state["intent_details"] = data.get("intent_details", {})

        logger.info(f"🎯 Intent Type: {state['intent_type']}")
        logger.info(
            f"🔍 Intent Details: {json.dumps(state['intent_details'], indent=2)}")
        logger.info(f"💯 Confidence: {data.get('confidence', 0)}%")

        if state["intent_type"] == "unclear":
            clarification = data.get(
                "suggested_clarification", "Could you please provide more details?")
            state["output"] = f"🤔 {clarification}"
            logger.info(f"❓ Intent unclear - asking for clarification")

    except Exception as e:
        logger.error(f"❌ Intent classification failed: {e}")
        logger.error(f"🔴 Exception type: {type(e).__name__}")
        logger.error(f"🔴 Exception details: {str(e)}")
        state["intent_type"] = "unclear"
        state["error"] = str(e)

    logger.info(f"✅ STEP 1 COMPLETED - Intent: {state['intent_type']}")
    logger.info("=" * 80)
    return state


# ====================================================
# --- Node 2: Generate Query or Action ---
# ====================================================

def generate_query_or_action(state: State) -> State:
    """Generate appropriate Cypher query or determine action based on intent"""
    logger.info("=" * 80)
    logger.info("🔧 STEP 2: GENERATE QUERY OR ACTION")
    logger.info("=" * 80)

    if state["intent_type"] in ["unclear", "general"]:
        logger.info(
            f"⏭️ Skipping query generation for intent type: {state['intent_type']}")
        logger.info("=" * 80)
        return state

    user_input = state["input"]
    intent_type = state["intent_type"]
    intent_details = state["intent_details"]

    logger.info(f"📝 User Input: {user_input}")
    logger.info(f"🎯 Intent Type: {intent_type}")
    logger.info(f"📊 Intent Details: {json.dumps(intent_details, indent=2)}")

    schema_info = """
**Neo4j Knowledge Graph Schema:**

Nodes:
- Customer: (name, id, location, loyalty_score, email, phone, join_date)
- Dish: (name, type [veg/non-veg], price, popularity_score, cuisine, category, description)
- Ingredient: (name, allergy_info, is_vegan, nutritional_info)
- Review: (id, rating, feedback_text, sentiment, timestamp)
- Order: (order_id, timestamp, total_amount, status, delivery_time)

Relationships:
- (Customer)-[:ORDERED]->(Order)
- (Order)-[:CONTAINS]->(Dish)
- (Dish)-[:CONTAINS_INGREDIENT]->(Ingredient)
- (Customer)-[:LEFT_REVIEW]->(Review)
- (Review)-[:RATES]->(Dish)
- (Customer)-[:PREFERS]->(Dish)

IMPORTANT: 
- Use CASE-INSENSITIVE matching with toLower() for name searches!
- Convert DateTime to string using toString() for JSON serialization
- Current date: 2025-01-17 (use for "this month" = January 2025)
"""

    prompt = f"""
You are generating Cypher queries for a restaurant CRM system.

**Intent Type:** {intent_type}
**Intent Details:** {json.dumps(intent_details, indent=2)}
**User Question:** "{user_input}"

{schema_info}

**Critical Query Writing Rules:**

1. **DateTime Handling:**
   - ALWAYS convert timestamp to string: toString(o.timestamp) AS timestamp
   - For date filtering: WHERE o.timestamp >= datetime('2025-01-01T00:00:00')
   - Example: RETURN toString(o.timestamp) AS date

2. **For Customer Name Searches:**
   - ALWAYS use toLower() for case-insensitive matching
   - Example: WHERE toLower(c.name) CONTAINS toLower("arjun")

3. **For Dish Name Searches:**
   - ALWAYS use toLower() for case-insensitive matching
   - Example: WHERE toLower(d.name) CONTAINS toLower("biryani")

4. **Return Complete Information:**
   - For customers: name, id, email, phone, location, loyalty_score, join_date
   - For dishes: name, type, price, cuisine, category, description, popularity_score
   - For orders: order_id, toString(timestamp) AS timestamp, total_amount, status, delivery_time

Now generate the query for the user's question. Return ONLY valid JSON:
{{
  "requires_data": <true/false>,
  "cypher_query": "<valid Cypher query or null>",
  "query_explanation": "<what this query does>",
  "expected_output": "<what kind of data we'll get>",
  "requires_action": <true if needs follow-up action>,
  "action_type": "<if action needed: update_customer, send_notification, follow_up, etc>"
}}

Remember: 
- ALWAYS use toString() for DateTime fields!
- ALWAYS use toLower() for name matching!
- For "this month", use datetime('2025-01-01T00:00:00') to datetime('2025-02-01T00:00:00')
"""

    logger.info("🤖 Calling LLM to generate Cypher query...")

    try:
        response = llm.invoke(prompt).content.strip()
        logger.info(f"✅ LLM Response received (length: {len(response)})")
        logger.info(f"📄 Raw LLM Response:\n{response}")

        data = safe_parse_json(response)
        logger.info(f"✅ Parsed JSON successfully")
        logger.info(f"📊 Parsed Data: {json.dumps(data, indent=2)}")

        state["cypher_query"] = data.get("cypher_query")
        state["requires_action"] = data.get("requires_action", False)
        state["action_type"] = data.get("action_type")

        logger.info(f"📝 Generated Cypher Query: {state['cypher_query']}")
        logger.info(f"⚡ Requires Action: {state['requires_action']}")
        logger.info(f"🎬 Action Type: {state['action_type']}")

        if state["cypher_query"]:
            logger.info(f"✅ Query generated successfully")
        else:
            logger.warning(
                f"⚠️ No query generated - cypher_query is None/empty")

    except Exception as e:
        logger.error(f"❌ Query generation failed: {e}")
        logger.error(f"🔴 Exception type: {type(e).__name__}")
        logger.error(f"🔴 Exception details: {str(e)}")
        state["error"] = str(e)

    logger.info(
        f"✅ STEP 2 COMPLETED - Query: {state['cypher_query'][:100] if state['cypher_query'] else 'None'}...")
    logger.info("=" * 80)
    return state


# ====================================================
# --- Node 3: Execute Query (Non-streaming part) ---
# ====================================================

def execute_query_only(state: State) -> State:
    """Execute query and prepare data (no response generation)"""
    logger.info("=" * 80)
    logger.info("⚙️ STEP 3A: EXECUTE QUERY")
    logger.info("=" * 80)

    intent_type = state["intent_type"]
    logger.info(f"🎯 Intent Type: {intent_type}")
    logger.info(f"📝 Cypher Query: {state.get('cypher_query')}")

    # Note: Actual query execution happens in app.py using execute_cypher_query
    # This node just prepares the state
    if not state.get("cypher_query"):
        logger.warning("⚠️ No Cypher query to execute")
        state["query_results"] = []

    logger.info(f"✅ STEP 3A COMPLETED")
    logger.info("=" * 80)
    return state


# ====================================================
# --- Streaming Response Generator ---
# ====================================================

async def generate_streaming_response(state: State) -> AsyncGenerator[str, None]:
    """Generate streaming response using LLM"""
    logger.info("=" * 80)
    logger.info("📡 STEP 3B: GENERATE STREAMING RESPONSE")
    logger.info("=" * 80)

    intent_type = state["intent_type"]
    query_results = state.get("query_results", [])
    intent_details = state["intent_details"]
    user_input = state["input"]

    # Handle general intent
    if intent_type == "general":
        logger.info("👋 Handling general intent...")
        response_text = generate_general_response(state)
        for char in response_text:
            yield json.dumps({"type": "token", "content": char}) + "\n"
            await asyncio.sleep(0.01)
        yield json.dumps({"type": "done"}) + "\n"
        return

    # Handle unclear intent
    if intent_type == "unclear" and state.get("output"):
        logger.info("❓ Intent unclear - streaming clarification")
        for char in state["output"]:
            yield json.dumps({"type": "token", "content": char}) + "\n"
            await asyncio.sleep(0.01)
        yield json.dumps({"type": "done"}) + "\n"
        return

    # Handle no results
    if state.get("cypher_query") and not query_results:
        logger.info("🔍 No results found - generating no results response")
        response_text = generate_no_results_response(state)
        for char in response_text:
            yield json.dumps({"type": "token", "content": char}) + "\n"
            await asyncio.sleep(0.01)
        yield json.dumps({"type": "done"}) + "\n"
        return

    # Generate contextual response with streaming
    prompt = f"""
You are a friendly, professional restaurant CRM assistant. Generate a natural response.

**Intent Type:** {intent_type}
**What User Asked:** "{user_input}"
**What We Found:** {intent_details.get('main_focus', '')}

**Data Retrieved:**
{json.dumps(query_results[:20], indent=2) if query_results else "No data available"}

**Response Guidelines by Intent:**

**ANALYTICS:** 
- Start with key metrics and insights
- Highlight trends and patterns
- Use numbers and percentages
- Format: "📊 Based on the data..."

**CRM:**
- Focus on customer insights
- Personalization opportunities
- Action items for engagement
- Format: "👥 Here's what I found about your customers..."

**ORDER_MANAGEMENT:**
- Order status and details
- Timeline information
- Next steps if applicable
- Format: "📦 Order Information..."

**CUSTOMER_SERVICE:**
- Sentiment and feedback summary
- Priority issues
- Actionable recommendations
- Format: "⭐ Customer Feedback Summary..."

**RECOMMENDATION:**
- Top suggestions with reasoning
- Dietary considerations
- Popularity and ratings
- Format: "🍽️ Based on your preferences..."

**Requirements:**
1. Be conversational and friendly
2. Use appropriate emojis (but not excessively)
3. Structure data clearly (bullets or numbered lists)
4. Highlight actionable insights
5. Keep it concise but informative
6. Offer to provide more details if needed

Generate the response in plain text (no JSON):
"""

    try:
        logger.info("🤖 Starting streaming LLM response...")

        full_response = ""
        async for chunk in llm_streaming.astream(prompt):
            if hasattr(chunk, 'content') and chunk.content:
                token = chunk.content
                full_response += token
                yield json.dumps({"type": "token", "content": token}) + "\n"

        state["output"] = full_response
        logger.info(
            f"✅ Streaming completed. Total length: {len(full_response)}")
        yield json.dumps({"type": "done"}) + "\n"

    except Exception as e:
        logger.error(f"❌ Streaming response generation failed: {e}")
        error_message = "⚠️ I encountered an error while generating the response. Please try again."
        yield json.dumps({"type": "token", "content": error_message}) + "\n"
        yield json.dumps({"type": "error", "message": str(e)}) + "\n"


# ====================================================
# --- Graph Definition ---
# ====================================================

def should_generate_query(state: State) -> str:
    """Decide whether to generate query or skip to execution"""
    intent_type = state["intent_type"]
    has_output = state.get("output") is not None and state.get("output") != ""

    logger.info(f"🔀 ROUTING DECISION:")
    logger.info(f"   Intent Type: {intent_type}")
    logger.info(f"   Has Output: {has_output}")
    logger.info(f"   Output Value: {state.get('output')}")

    if intent_type == "unclear" and has_output:
        logger.info(f"   ➡️ ROUTE: execute (unclear with output)")
        return "execute"

    if intent_type == "general":
        logger.info(f"   ➡️ ROUTE: execute (general intent)")
        return "execute"

    logger.info(f"   ➡️ ROUTE: generate (needs query generation)")
    return "generate"


# Build the graph
graph = StateGraph(State)
graph.add_node("classify_intent", classify_intent)
graph.add_node("generate_query_or_action", generate_query_or_action)
graph.add_node("execute_query_only", execute_query_only)

graph.set_entry_point("classify_intent")

graph.add_conditional_edges(
    "classify_intent",
    should_generate_query,
    {
        "generate": "generate_query_or_action",
        "execute": "execute_query_only"
    }
)

graph.add_edge("generate_query_or_action", "execute_query_only")

app_graph = graph.compile()

logger.info("✅ LangGraph workflow compiled successfully")
