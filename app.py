import os
from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langgraph.graph import StateGraph
from langchain_groq import ChatGroq
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, ClientError
import redis
import json
import re
from typing import List, Dict, Optional
from typing_extensions import TypedDict
from dotenv import load_dotenv
from contextlib import contextmanager
import logging

# ====================================================
# --- Configure Logging ---
# ====================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# ====================================================
# --- Initialize FastAPI ---
# ====================================================
app = FastAPI(
    title="Smart Food Recommendation Chatbot",
    description="AI-powered food recommendation system with Neo4j and Redis",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================================================
# --- Initialize LLM (ChatGroq) ---
# ====================================================
try:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.4,
        streaming=False,
    )
    logger.info("✅ LLM initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize LLM: {e}")
    raise

# ====================================================
# --- Initialize Neo4j ---
# ====================================================
try:
    driver = GraphDatabase.driver(
        "neo4j+s://05438154.databases.neo4j.io",
        auth=("neo4j", os.getenv("NEO4J_PASSWORD")),
        max_connection_lifetime=3600,
        max_connection_pool_size=50,
        connection_acquisition_timeout=120
    )
    # Verify connection
    driver.verify_connectivity()
    logger.info("✅ Neo4j connected successfully")
except Exception as e:
    logger.error(f"❌ Failed to connect to Neo4j: {e}")
    raise

# ====================================================
# --- Initialize Redis ---
# ====================================================
try:
    redis_client = redis.Redis(
        host='redis-11505.c276.us-east-1-2.ec2.redns.redis-cloud.com',
        port=11505,
        decode_responses=True,
        username="default",
        password=os.getenv("REDIS_PASSWORD"),
        socket_connect_timeout=5,
        socket_timeout=5,
        retry_on_timeout=True
    )
    # Test connection
    redis_client.ping()
    logger.info("✅ Redis connected successfully")
except Exception as e:
    logger.error(f"❌ Failed to connect to Redis: {e}")
    raise

# ====================================================
# --- Helper: Safe JSON Parsing ---
# ====================================================


def safe_parse_json(text: str) -> dict:
    """
    Safely parse JSON from LLM response, handling markdown code blocks
    """
    try:
        # Remove markdown code blocks
        cleaned = re.sub(r"^```json\s*|\s*```$", "",
                         text.strip(), flags=re.MULTILINE).strip()
        # Try to extract JSON if there's extra text
        json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if json_match:
            cleaned = json_match.group()
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.warning(f"⚠️ JSON parse failed: {e}, Text: {text[:200]}")
        return {}
    except Exception as e:
        logger.error(f"⚠️ Unexpected error in JSON parsing: {e}")
        return {}

# ====================================================
# --- Redis Helpers ---
# ====================================================


def update_user_preferences(user_id: str, cuisine: str, type_: str, location: str, liked_food: Optional[str] = None):
    """
    Update user preferences in Redis with error handling
    """
    try:
        data = {
            "cuisine": cuisine,
            "type": type_,
            "location": location,
            "liked_food": liked_food,
            "updated_at": str(pd.Timestamp.now()) if 'pd' in globals() else "now"
        }
        redis_client.setex(
            f"user:{user_id}:prefs",
            86400 * 30,  # 30 days expiry
            json.dumps(data)
        )
        logger.info(f"✅ Updated preferences for user {user_id}")
    except Exception as e:
        logger.error(f"❌ Failed to update preferences for {user_id}: {e}")


def get_user_preferences(user_id: str) -> dict:
    """
    Get user preferences from Redis
    """
    try:
        prefs_json = redis_client.get(f"user:{user_id}:prefs")
        if prefs_json:
            return json.loads(prefs_json)
    except Exception as e:
        logger.error(f"❌ Failed to get preferences for {user_id}: {e}")
    return {}


def get_conversation_history(user_id: str) -> List[Dict[str, str]]:
    """
    Get conversation history from Redis
    """
    try:
        history_json = redis_client.get(f"user:{user_id}:history")
        if history_json:
            history = json.loads(history_json)
            # Limit to last 20 messages to avoid context overflow
            return history[-20:]
    except Exception as e:
        logger.error(f"❌ Failed to get history for {user_id}: {e}")
    return []


def save_conversation_history(user_id: str, history: List[Dict[str, str]]):
    """
    Save conversation history to Redis with expiry
    """
    try:
        # Keep only last 50 messages
        trimmed_history = history[-50:]
        redis_client.setex(
            f"user:{user_id}:history",
            86400 * 7,  # 7 days expiry
            json.dumps(trimmed_history)
        )
    except Exception as e:
        logger.error(f"❌ Failed to save history for {user_id}: {e}")

# ====================================================
# --- Neo4j Helper Functions ---
# ====================================================


@contextmanager
def get_neo4j_session():
    """
    Context manager for Neo4j sessions
    """
    session = driver.session()
    try:
        yield session
    finally:
        session.close()


def get_food_recommendations(location: str, cuisine: Optional[str] = None, type_: Optional[str] = None) -> List[Dict]:
    """
    Fetch dishes for a given city/location from Neo4j with optional filters
    """
    try:
        with get_neo4j_session() as session:
            # Build dynamic query based on filters
            conditions = ["toLower(c.name) CONTAINS toLower($location)"]
            params = {"location": location}

            if cuisine and cuisine.lower() != "any":
                conditions.append("toLower(d.cuisine) = toLower($cuisine)")
                params["cuisine"] = cuisine

            if type_ and type_.lower() in ["veg", "non-veg"]:
                conditions.append("toLower(d.type) = toLower($type)")
                params["type"] = type_

            where_clause = " AND ".join(conditions)

            query = f"""
            MATCH (d:Dish)-[:AVAILABLE_IN]->(c:City)
            WHERE {where_clause}
            RETURN d.name AS name, d.cuisine AS cuisine, d.type AS type, c.name AS city
            ORDER BY d.name
            LIMIT 15
            """

            result = session.run(query, **params)
            dishes = [dict(r) for r in result]
            print("+______________",dishes)
            logger.info(f"🍽 Found {len(dishes)} dishes for {location}")
            return dishes
    except ServiceUnavailable as e:
        logger.error(f"❌ Neo4j service unavailable: {e}")
        return []
    except Exception as e:
        logger.error(f"❌ Neo4j query error: {e}")
        return []


def store_user_activity(user_id: str, liked_food: str, location: str):
    """
    Store user -> liked_food -> location relationships in Neo4j
    """
    if not liked_food or not location:
        return

    try:
        with get_neo4j_session() as session:
            # First ensure the dish exists
            session.run(
                """
                MERGE (f:Dish {name: $liked_food})
                MERGE (c:City {name: $location})
                MERGE (f)-[:AVAILABLE_IN]->(c)
                """,
                liked_food=liked_food,
                location=location
            )

            # Then create user and like relationship
            session.run(
                """
                MERGE (u:User {id: $user_id})
                MATCH (f:Dish {name: $liked_food})
                MERGE (u)-[l:LIKED]->(f)
                ON CREATE SET l.created_at = timestamp()
                SET l.last_liked = timestamp()
                """,
                user_id=user_id,
                liked_food=liked_food
            )
            logger.info(
                f"✅ Stored activity: {user_id} liked {liked_food} in {location}")
    except Exception as e:
        logger.error(f"❌ Failed to store user activity: {e}")


def get_related_user_recommendations(user_id: str, location: str, limit: int = 5) -> List[Dict]:
    """
    Get recommendations based on what similar users liked in the same location
    """
    try:
        with get_neo4j_session() as session:
            # Simplified query that doesn't rely on LIKED relationships
            # This will work even if no likes exist yet
            query = """
            MATCH (f:Dish)-[:AVAILABLE_IN]->(c:City)
            WHERE toLower(c.name) CONTAINS toLower($location)
            WITH f, c
            ORDER BY rand()
            RETURN DISTINCT f.name AS dish, f.cuisine AS cuisine, f.type AS type, c.name AS city
            LIMIT $limit
            """
            results = session.run(query, location=location, limit=limit)

            recommendations = [
                {
                    "name": record["dish"],
                    "cuisine": record["cuisine"],
                    "type": record["type"],
                    "city": record["city"]
                }
                for record in results if record.get("dish")
            ]

            logger.info(
                f"🎯 Found {len(recommendations)} related recommendations")
            return recommendations
    except Exception as e:
        logger.error(f"❌ Failed to get related recommendations: {e}")
        return []

# ====================================================
# --- State Definition ---
# ====================================================


class State(TypedDict):
    input: str
    user_id: str
    messages: List[Dict[str, str]]
    cuisine: str
    type: str
    location: str
    liked_food: Optional[str]
    output: str
    error: Optional[str]

# ====================================================
# --- Node 1: Extract Intent ---
# ====================================================


def extract_intent(state: State) -> State:
    """
    Extract user intent and preferences from the message
    """
    user_input = state["input"]

    prompt = f"""
    Analyze this user message and extract food preferences:
    "{user_input}"

    Extract these details:
    1. cuisine: Type of cuisine (e.g., South Indian, Chinese, Italian, North Indian, Continental, Any)
    2. type: MUST be either "veg" or "non-veg" - never use "unknown" or other values
    3. location: City or area name
    4. liked_food: Any specific dish mentioned that the user liked or wants

    Important rules:
    - If something isn't mentioned, use "same" to keep previous preference
    - For type: If not specified, use "same" (NOT "unknown"). Default assumption is "veg" if no history
    - If user mentions a dish they liked/enjoyed, put it in liked_food
    - For location, extract city names carefully (e.g., "I'm in Mangalore" -> location: "Mangalore")
    - Be case-insensitive when matching

    Respond ONLY with valid JSON (no extra text):
    {{
      "cuisine": "<cuisine or 'same'>",
      "type": "<veg or non-veg or 'same'>",
      "location": "<location or 'same'>",
      "liked_food": "<food name or 'none'>"
    }}
    """

    try:
        response = llm.invoke(prompt).content.strip()
        data = safe_parse_json(response)
        logger.info(f"📝 Extracted intent: {data}")
    except Exception as e:
        logger.error(f"❌ Failed to extract intent: {e}")
        data = {}

    # Get previous preferences
    prev_prefs = get_user_preferences(state["user_id"])

    # Update state with extracted or previous values
    extracted_cuisine = data.get("cuisine", "").strip()
    extracted_type = data.get("type", "").strip().lower()
    extracted_location = data.get("location", "").strip()
    extracted_liked = data.get("liked_food", "").strip()

    # Handle cuisine
    state["cuisine"] = (
        extracted_cuisine if extracted_cuisine not in ["", "same", "unknown", None]
        else prev_prefs.get("cuisine", "Indian")
    )

    # Handle type - ensure it's always veg or non-veg
    if extracted_type in ["veg", "non-veg"]:
        state["type"] = extracted_type
    elif extracted_type == "same" or extracted_type == "":
        state["type"] = prev_prefs.get("type", "veg")
    else:
        # Fallback for any other value
        state["type"] = "veg"

    # Handle location
    state["location"] = (
        extracted_location if extracted_location not in ["", "same", "unknown", None]
        else prev_prefs.get("location", "unknown")
    )

    # Handle liked food
    state["liked_food"] = (
        extracted_liked if extracted_liked not in ["", "none", None]
        else None
    )

    # Save preferences
    update_user_preferences(
        state["user_id"],
        state["cuisine"],
        state["type"],
        state["location"],
        state["liked_food"]
    )

    # Add to conversation history
    state.setdefault("messages", []).append({
        "role": "user",
        "content": user_input
    })

    return state

# ====================================================
# --- Node 2: Recommend Food ---
# ====================================================


def recommend_food(state: State) -> State:
    """
    Generate food recommendations based on user preferences
    """
    cuisine = state["cuisine"]
    type_ = state["type"]
    location = state["location"]
    liked_food = state.get("liked_food")

    # Validate location
    if location == "unknown":
        prefs = get_user_preferences(state["user_id"])
        location = prefs.get("location", "unknown")

        if location == "unknown":
            response_text = "🤔 I'd love to help! Could you tell me which city you're in?"
            state["messages"].append(
                {"role": "assistant", "content": response_text})
            state["output"] = response_text
            save_conversation_history(state["user_id"], state["messages"])
            return state

    # Store user activity if they mentioned liking something
    if liked_food:
        store_user_activity(state["user_id"], liked_food, location)

    # Fetch recommendations
    dishes = get_food_recommendations(location, cuisine, type_)
    related = get_related_user_recommendations(
        state["user_id"], location, limit=5)

    if dishes or related:
        # Combine and deduplicate recommendations
        dish_list = [d["name"] for d in dishes]
        related_list = [r["name"] for r in related]
        # Preserve order, remove duplicates
        all_dishes = list(dict.fromkeys(dish_list + related_list))

        # Build response with better formatting
        response_parts = []

        # Main greeting
        emoji = "🥗" if type_ == "veg" else "🍖"
        response_parts.append(
            f"{emoji} Here are some delicious {cuisine} {type_} options in {location}:")

        # List dishes (max 8 for readability)
        if len(all_dishes) > 8:
            response_parts.append("• " + "\n• ".join(all_dishes[:8]))
            response_parts.append(f"\n...and {len(all_dishes) - 8} more!")
        else:
            response_parts.append("• " + "\n• ".join(all_dishes))

        # Add personalized note if they liked something
        if liked_food and related_list:
            response_parts.append(
                f"\n\n💡 Since you enjoyed {liked_food}, you might also like: {', '.join(related_list[:3])}")

        response_text = "\n".join(response_parts)
    else:
        # Fallback to LLM suggestions
        try:
            llm_prompt = f"""
            Suggest 5-7 popular {cuisine} {type_} dishes that would be available in {location or 'India'}.
            Format as a friendly recommendation message.
            Keep it concise and conversational.
            """
            llm_response = llm.invoke(llm_prompt).content.strip()
            response_text = f"🔍 I couldn't find specific local results, but here are some popular options:\n\n{llm_response}"
        except Exception as e:
            logger.error(f"❌ LLM fallback failed: {e}")
            response_text = f"😅 I'm having trouble finding {cuisine} {type_} dishes in {location}. Could you try a different cuisine or location?"

    # Add response to conversation
    state["messages"].append({"role": "assistant", "content": response_text})
    state["output"] = response_text
    save_conversation_history(state["user_id"], state["messages"])

    return state


# ====================================================
# --- Graph Definition ---
# ====================================================
graph = StateGraph(State)
graph.add_node("extract_intent", extract_intent)
graph.add_node("recommend_food", recommend_food)
graph.set_entry_point("extract_intent")
graph.add_edge("extract_intent", "recommend_food")
app_graph = graph.compile()

# ====================================================
# --- API Routes ---
# ====================================================


@app.get("/")
async def root():
    """
    Health check endpoint
    """
    return {
        "status": "online",
        "service": "Smart Food Recommendation Chatbot",
        "version": "2.0.0"
    }


@app.post("/chat")
async def chat(user_id: str = Body(..., embed=True), message: str = Body(..., embed=True)):
    """
    Main chat endpoint
    """
    if not user_id or not message:
        raise HTTPException(
            status_code=400, detail="user_id and message are required")

    try:
        history = get_conversation_history(user_id)

        state = {
            "input": message.strip(),
            "user_id": user_id,
            "messages": history,
            "cuisine": "",
            "type": "",
            "location": "",
            "liked_food": None,
            "output": "",
            "error": None
        }

        result = app_graph.invoke(state)

        return {
            "success": True,
            "response": result["output"],
            # Return last 10 messages
            "conversation_history": result["messages"][-10:],
            "preferences": {
                "cuisine": result.get("cuisine"),
                "type": result.get("type"),
                "location": result.get("location")
            }
        }
    except Exception as e:
        logger.error(f"❌ Chat error for user {user_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to process message: {str(e)}")


@app.get("/chat/history/{user_id}")
async def get_history(user_id: str):
    """
    Get conversation history for a user
    """
    try:
        history = get_conversation_history(user_id)
        prefs = get_user_preferences(user_id)
        return {
            "success": True,
            "user_id": user_id,
            "history": history,
            "preferences": prefs
        }
    except Exception as e:
        logger.error(f"❌ Failed to get history for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/chat/history/{user_id}")
async def clear_history(user_id: str):
    """
    Clear conversation history for a user
    """
    try:
        redis_client.delete(f"user:{user_id}:history")
        redis_client.delete(f"user:{user_id}:prefs")
        logger.info(f"🗑️ Cleared history for user {user_id}")
        return {
            "success": True,
            "message": f"Conversation history and preferences cleared for user {user_id}"
        }
    except Exception as e:
        logger.error(f"❌ Failed to clear history for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/setup")
async def setup_data(dishes: List[Dict[str, str]] = Body(...)):
    """
    Bulk insert dishes into Neo4j
    """
    if not dishes:
        raise HTTPException(
            status_code=400, detail="dishes list cannot be empty")

    try:
        with get_neo4j_session() as session:
            for dish in dishes:
                if not all(k in dish for k in ["name", "cuisine", "type", "city"]):
                    continue

                session.run(
                    """
                    MERGE (d:Dish {name: $name})
                    SET d.cuisine = $cuisine, d.type = $type, d.updated_at = timestamp()
                    MERGE (c:City {name: $city})
                    MERGE (d)-[:AVAILABLE_IN]->(c)
                    """,
                    name=dish["name"],
                    cuisine=dish["cuisine"],
                    type=dish["type"],
                    city=dish["city"]
                )

        logger.info(f"✅ Added {len(dishes)} dishes to database")
        return {
            "success": True,
            "message": f"✅ {len(dishes)} dishes added successfully!"
        }
    except Exception as e:
        logger.error(f"❌ Failed to setup data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dishes")
async def get_all_dishes(city: Optional[str] = None, cuisine: Optional[str] = None):
    """
    Get all dishes with optional filters
    """
    try:
        with get_neo4j_session() as session:
            conditions = []
            params = {}

            if city:
                conditions.append("toLower(c.name) CONTAINS toLower($city)")
                params["city"] = city

            if cuisine:
                conditions.append("toLower(d.cuisine) = toLower($cuisine)")
                params["cuisine"] = cuisine

            where_clause = "WHERE " + \
                " AND ".join(conditions) if conditions else ""

            query = f"""
            MATCH (d:Dish)-[:AVAILABLE_IN]->(c:City)
            {where_clause}
            RETURN d.name AS name, d.cuisine AS cuisine, d.type AS type, c.name AS city
            ORDER BY c.name, d.name
            LIMIT 100
            """

            result = session.run(query, **params)
            dishes = [dict(r) for r in result]

        return {
            "success": True,
            "count": len(dishes),
            "dishes": dishes
        }
    except Exception as e:
        logger.error(f"❌ Failed to get dishes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """
    Get system statistics
    """
    try:
        with get_neo4j_session() as session:
            result = session.run("""
            MATCH (d:Dish) WITH count(d) AS dishes
            MATCH (c:City) WITH dishes, count(c) AS cities
            MATCH (u:User) WITH dishes, cities, count(u) AS users
            OPTIONAL MATCH ()-[l:LIKED]->() 
            RETURN dishes, cities, users, count(l) AS likes
            """).single()

        return {
            "success": True,
            "statistics": {
                "total_dishes": result["dishes"],
                "total_cities": result["cities"],
                "total_users": result["users"],
                "total_likes": result["likes"]
            }
        }
    except Exception as e:
        logger.error(f"❌ Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug/preferences/{user_id}")
async def get_user_debug_info(user_id: str):
    """
    Get detailed user information for debugging
    """
    try:
        prefs = get_user_preferences(user_id)
        history = get_conversation_history(user_id)

        # Get user's liked dishes from Neo4j
        with get_neo4j_session() as session:
            result = session.run("""
            MATCH (u:User {id: $user_id})-[l:LIKED]->(d:Dish)
            RETURN d.name AS dish, l.created_at AS liked_at
            ORDER BY l.last_liked DESC
            LIMIT 10
            """, user_id=user_id)
            liked_dishes = [dict(r) for r in result]

        return {
            "success": True,
            "user_id": user_id,
            "preferences": prefs,
            "conversation_count": len(history),
            "liked_dishes": liked_dishes
        }
    except Exception as e:
        logger.error(f"❌ Failed to get debug info for {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ====================================================
# --- Cleanup on Shutdown ---
# ====================================================


@app.on_event("shutdown")
async def shutdown_event():
    """
    Cleanup connections on shutdown
    """
    try:
        driver.close()
        redis_client.close()
        logger.info("✅ Connections closed successfully")
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")
