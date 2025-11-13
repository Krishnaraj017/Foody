import os
from turtle import pd
from fastapi import FastAPI, Body, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from langgraph.graph import StateGraph
from langchain_groq import ChatGroq
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, ClientError
import redis
import json
import re
import secrets
import hashlib
from datetime import datetime, timedelta
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

# Security
security = HTTPBearer()

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
# --- Authentication Helpers ---
# ====================================================


def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()


def generate_access_token(user_id: str) -> str:
    """Generate a unique access token"""
    return secrets.token_urlsafe(32)


def store_user_credentials(email: str, password: str, user_id: str):
    """Store user credentials in Redis"""
    try:
        hashed_password = hash_password(password)
        user_data = {
            "user_id": user_id,
            "email": email,
            "password": hashed_password,
            "created_at": datetime.now().isoformat()
        }
        redis_client.setex(
            f"user:email:{email}",
            86400 * 365,  # 1 year expiry
            json.dumps(user_data)
        )
        logger.info(f"✅ Stored credentials for {email}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to store credentials: {e}")
        return False


def get_user_by_email(email: str) -> Optional[Dict]:
    """Get user data by email"""
    try:
        user_json = redis_client.get(f"user:email:{email}")
        if user_json:
            return json.loads(user_json)
    except Exception as e:
        logger.error(f"❌ Failed to get user by email: {e}")
    return None


def verify_password(email: str, password: str) -> Optional[str]:
    """Verify password and return user_id if valid"""
    user_data = get_user_by_email(email)
    if not user_data:
        return None

    hashed_password = hash_password(password)
    if user_data["password"] == hashed_password:
        return user_data["user_id"]
    return None


def store_access_token(user_id: str, access_token: str):
    """Store access token in Redis with expiry"""
    try:
        token_data = {
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(days=30)).isoformat()
        }
        # Store token -> user_id mapping
        redis_client.setex(
            f"token:{access_token}",
            86400 * 30,  # 30 days expiry
            json.dumps(token_data)
        )
        # Store user_id -> token mapping (to invalidate old tokens)
        redis_client.setex(
            f"user:{user_id}:token",
            86400 * 30,
            access_token
        )
        logger.info(f"✅ Stored access token for user {user_id}")
    except Exception as e:
        logger.error(f"❌ Failed to store access token: {e}")


def verify_access_token(token: str) -> Optional[str]:
    """Verify access token and return user_id"""
    try:
        token_json = redis_client.get(f"token:{token}")
        if token_json:
            token_data = json.loads(token_json)
            # Check if token is expired
            expires_at = datetime.fromisoformat(token_data["expires_at"])
            if datetime.now() < expires_at:
                return token_data["user_id"]
    except Exception as e:
        logger.error(f"❌ Failed to verify token: {e}")
    return None


def revoke_access_token(user_id: str):
    """Revoke user's access token"""
    try:
        token = redis_client.get(f"user:{user_id}:token")
        if token:
            redis_client.delete(f"token:{token}")
            redis_client.delete(f"user:{user_id}:token")
            logger.info(f"✅ Revoked token for user {user_id}")
    except Exception as e:
        logger.error(f"❌ Failed to revoke token: {e}")


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Dependency to get current authenticated user"""
    token = credentials.credentials
    user_id = verify_access_token(token)

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user_id

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
# --- Authentication API Routes ---
# ====================================================


@app.post("/auth/register")
async def register(email: str = Body(...), password: str = Body(...)):
    """
    Register a new user
    """
    if not email or not password:
        raise HTTPException(
            status_code=400,
            detail="Email and password are required"
        )

    # Check if user already exists
    if get_user_by_email(email):
        raise HTTPException(
            status_code=400,
            detail="User with this email already exists"
        )

    # Validate password strength
    if len(password) < 6:
        raise HTTPException(
            status_code=400,
            detail="Password must be at least 6 characters long"
        )

    # Generate user_id
    user_id = f"user_{secrets.token_urlsafe(16)}"

    # Store credentials
    if not store_user_credentials(email, password, user_id):
        raise HTTPException(
            status_code=500,
            detail="Failed to register user"
        )

    # Generate access token
    access_token = generate_access_token(user_id)
    store_access_token(user_id, access_token)

    logger.info(f"✅ New user registered: {email}")

    return {
        "success": True,
        "message": "User registered successfully",
        "user_id": user_id,
        "access_token": access_token,
        "token_type": "bearer"
    }


@app.post("/auth/login")
async def login(email: str = Body(...), password: str = Body(...)):
    """
    Login user and return access token
    """
    if not email or not password:
        raise HTTPException(
            status_code=400,
            detail="Email and password are required"
        )

    # Verify credentials
    user_id = verify_password(email, password)
    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    # Generate new access token
    access_token = generate_access_token(user_id)
    store_access_token(user_id, access_token)

    logger.info(f"✅ User logged in: {email}")

    return {
        "success": True,
        "message": "Login successful",
        "user_id": user_id,
        "access_token": access_token,
        "token_type": "bearer"
    }


@app.post("/auth/logout")
async def logout(current_user: str = Depends(get_current_user)):
    """
    Logout user and revoke access token
    """
    revoke_access_token(current_user)
    logger.info(f"✅ User logged out: {current_user}")

    return {
        "success": True,
        "message": "Logged out successfully"
    }


@app.get("/auth/me")
async def get_current_user_info(current_user: str = Depends(get_current_user)):
    """
    Get current authenticated user information
    """
    prefs = get_user_preferences(current_user)

    return {
        "success": True,
        "user_id": current_user,
        "preferences": prefs
    }

# ====================================================
# --- API Routes (Protected) ---
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
async def chat(
    message: str = Body(..., embed=True),
    current_user: str = Depends(get_current_user)
):
    """
    Main chat endpoint - PROTECTED (requires authentication)
    Only logged-in users with valid access tokens can chat
    """
    if not message or not message.strip():
        raise HTTPException(
            status_code=400,
            detail="message cannot be empty"
        )

    try:
        # Get conversation history for the authenticated user
        history = get_conversation_history(current_user)

        state = {
            "input": message.strip(),
            "user_id": current_user,
            "messages": history,
            "cuisine": "",
            "type": "",
            "location": "",
            "liked_food": None,
            "output": "",
            "error": None
        }

        # Process the chat request
        result = app_graph.invoke(state)

        logger.info(f"✅ Chat processed for user {current_user}")

        return {
            "success": True,
            "response": result["output"],
            # Last 10 messages
            "conversation_history": result["messages"][-10:],
            "preferences": {
                "cuisine": result.get("cuisine"),
                "type": result.get("type"),
                "location": result.get("location")
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Chat error for user {current_user}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process message: {str(e)}"
        )


@app.get("/chat/history")
async def get_history(current_user: str = Depends(get_current_user)):
    """
    Get conversation history - PROTECTED (requires authentication)
    Returns conversation history only for the authenticated user
    """
    try:
        history = get_conversation_history(current_user)
        prefs = get_user_preferences(current_user)

        logger.info(f"✅ History retrieved for user {current_user}")

        return {
            "success": True,
            "user_id": current_user,
            "history": history,
            "history_count": len(history),
            "preferences": prefs
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get history for {current_user}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve conversation history: {str(e)}"
        )


@app.delete("/chat/history")
async def clear_history(current_user: str = Depends(get_current_user)):
    """
    Clear conversation history - PROTECTED (requires authentication)
    Clears conversation history and preferences only for the authenticated user
    """
    try:
        redis_client.delete(f"user:{current_user}:history")
        redis_client.delete(f"user:{current_user}:prefs")

        logger.info(
            f"🗑️ Cleared history and preferences for user {current_user}")

        return {
            "success": True,
            "message": "Conversation history and preferences cleared successfully",
            "user_id": current_user
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to clear history for {current_user}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear conversation history: {str(e)}"
        )


@app.post("/setup")
async def setup_data(dishes: List[Dict[str, str]] = Body(...)):
    """
    Bulk insert dishes into Neo4j (public endpoint for initial setup)
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
async def get_all_dishes(
    city: Optional[str] = None,
    cuisine: Optional[str] = None,
    current_user: str = Depends(get_current_user)
):
    """
    Get all dishes with optional filters - PROTECTED (requires authentication)
    Only authenticated users can browse dishes
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

        logger.info(f"✅ Dishes retrieved by user {current_user}")

        return {
            "success": True,
            "count": len(dishes),
            "dishes": dishes,
            "filters_applied": {
                "city": city,
                "cuisine": cuisine
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get dishes: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve dishes: {str(e)}"
        )


@app.get("/stats")
async def get_stats():
    """
    Get system statistics (public)
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


@app.get("/debug/preferences")
async def get_user_debug_info(current_user: str = Depends(get_current_user)):
    """
    Get detailed user information for debugging - PROTECTED (requires authentication)
    Returns debug information only for the authenticated user
    """
    try:
        prefs = get_user_preferences(current_user)
        history = get_conversation_history(current_user)

        # Get user's liked dishes from Neo4j
        with get_neo4j_session() as session:
            result = session.run("""
            MATCH (u:User {id: $user_id})-[l:LIKED]->(d:Dish)
            RETURN d.name AS dish, l.created_at AS liked_at
            ORDER BY l.last_liked DESC
            LIMIT 10
            """, user_id=current_user)
            liked_dishes = [dict(r) for r in result]

        logger.info(f"✅ Debug info retrieved for user {current_user}")

        return {
            "success": True,
            "user_id": current_user,
            "preferences": prefs,
            "conversation_count": len(history),
            "liked_dishes": liked_dishes,
            "liked_dishes_count": len(liked_dishes)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get debug info for {current_user}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve debug information: {str(e)}"
        )

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
