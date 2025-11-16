"""
Restaurant CRM Chatbot API
FastAPI application with authentication and chat endpoints
"""
import os
import json
import secrets
import hashlib
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from fastapi import FastAPI, Body, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
import uvicorn

# Import database connections
from db.connections import (
    initialize_connections,
    close_connections,
    execute_cypher_query,
    neo4j_connection,
    redis_connection
)

# Import LangGraph workflow
from graph.workflow import (
    app_graph,
    generate_streaming_response,
    generate_general_response,
    generate_no_results_response,
    llm
)

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
    title="Restaurant CRM Chatbot",
    description="Intelligent CRM assistant for restaurant management",
    version="4.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

# ====================================================
# --- Initialize Database Connections ---
# ====================================================


@app.on_event("startup")
async def startup_event():
    """Initialize connections on startup"""
    global neo4j_connection, redis_connection
    neo4j_connection, redis_connection = initialize_connections()
    logger.info("✅ Application started successfully")





@app.on_event("shutdown")
async def shutdown_event():
    """Close connections on shutdown"""
    try:
        close_connections()
        logger.info("🔒 Connections closed")
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}")


# ====================================================
# --- Authentication Helpers ---
# ====================================================

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def generate_access_token(user_id: str) -> str:
    return secrets.token_urlsafe(32)


def store_user_credentials(email: str, password: str, user_id: str):
    try:
        hashed_password = hash_password(password)
        user_data = {
            "user_id": user_id,
            "email": email,
            "password": hashed_password,
            "created_at": datetime.now().isoformat()
        }
        redis_connection.setex(
            f"user:email:{email}",
            86400 * 365,
            json.dumps(user_data)
        )
        return True
    except Exception as e:
        logger.error(f"❌ Failed to store credentials: {e}")
        return False


def get_user_by_email(email: str) -> Optional[Dict]:
    try:
        user_json = redis_connection.get(f"user:email:{email}")
        if user_json:
            return json.loads(user_json)
    except Exception as e:
        logger.error(f"❌ Failed to get user by email: {e}")
    return None


def verify_password(email: str, password: str) -> Optional[str]:
    user_data = get_user_by_email(email)
    if not user_data:
        return None
    hashed_password = hash_password(password)
    if user_data["password"] == hashed_password:
        return user_data["user_id"]
    return None


def store_access_token(user_id: str, access_token: str):
    try:
        token_data = {
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(days=30)).isoformat()
        }
        redis_connection.setex(
            f"token:{access_token}", 86400 * 30, json.dumps(token_data))
        redis_connection.setex(
            f"user:{user_id}:token", 86400 * 30, access_token)
    except Exception as e:
        logger.error(f"❌ Failed to store access token: {e}")


def verify_access_token(token: str) -> Optional[str]:
    try:
        token_json = redis_connection.get(f"token:{token}")
        if token_json:
            token_data = json.loads(token_json)
            expires_at = datetime.fromisoformat(token_data["expires_at"])
            if datetime.now() < expires_at:
                return token_data["user_id"]
    except Exception as e:
        logger.error(f"❌ Failed to verify token: {e}")
    return None


def revoke_access_token(user_id: str):
    try:
        token = redis_connection.get(f"user:{user_id}:token")
        if token:
            redis_connection.delete(f"token:{token}")
            redis_connection.delete(f"user:{user_id}:token")
    except Exception as e:
        logger.error(f"❌ Failed to revoke token: {e}")


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
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
# --- Conversation History Helpers ---
# ====================================================

def get_conversation_history(user_id: str) -> List[Dict[str, str]]:
    try:
        history_json = redis_connection.get(f"user:{user_id}:history")
        if history_json:
            return json.loads(history_json)[-20:]
    except Exception as e:
        logger.error(f"❌ Failed to get history: {e}")
    return []


def save_conversation_history(user_id: str, history: List[Dict[str, str]]):
    try:
        trimmed_history = history[-50:]
        redis_connection.setex(
            f"user:{user_id}:history", 86400 * 7, json.dumps(trimmed_history))
    except Exception as e:
        logger.error(f"❌ Failed to save history: {e}")


def get_user_context(user_id: str) -> Dict:
    """Get user's contextual information"""
    try:
        context_json = redis_connection.get(f"user:{user_id}:context")
        if context_json:
            return json.loads(context_json)
    except Exception as e:
        logger.error(f"❌ Failed to get user context: {e}")
    return {"preferences": {}, "last_interaction": None, "current_task": None}


def save_user_context(user_id: str, context: Dict):
    """Save user's contextual information"""
    try:
        redis_connection.setex(
            f"user:{user_id}:context", 86400 * 30, json.dumps(context))
    except Exception as e:
        logger.error(f"❌ Failed to save user context: {e}")


# ====================================================
# --- Authentication API Routes ---
# ====================================================

@app.post("/auth/register")
async def register(email: str = Body(...), password: str = Body(...)):
    if not email or not password:
        raise HTTPException(
            status_code=400, detail="Email and password are required")
    if get_user_by_email(email):
        raise HTTPException(status_code=400, detail="User already exists")
    if len(password) < 6:
        raise HTTPException(
            status_code=400, detail="Password must be at least 6 characters")

    user_id = f"user_{secrets.token_urlsafe(16)}"
    if not store_user_credentials(email, password, user_id):
        raise HTTPException(status_code=500, detail="Failed to register user")

    access_token = generate_access_token(user_id)
    store_access_token(user_id, access_token)

    return {
        "success": True,
        "message": "User registered successfully",
        "user_id": user_id,
        "access_token": access_token,
        "token_type": "bearer"
    }


@app.post("/auth/login")
async def login(email: str = Body(...), password: str = Body(...)):
    if not email or not password:
        raise HTTPException(
            status_code=400, detail="Email and password required")

    user_id = verify_password(email, password)
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token = generate_access_token(user_id)
    store_access_token(user_id, access_token)

    return {
        "success": True,
        "user_id": user_id,
        "access_token": access_token,
        "token_type": "bearer"
    }


@app.post("/auth/logout")
async def logout(current_user: str = Depends(get_current_user)):
    revoke_access_token(current_user)
    return {"success": True, "message": "Logged out successfully"}


# ====================================================
# --- Main Chat Endpoints (Streaming & Non-Streaming) ---
# ====================================================

@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "Restaurant CRM Chatbot",
        "version": "4.2.0",
        "description": "Intelligent CRM assistant with streaming support",
    }


@app.post("/chat/stream")
async def chat_stream(
    message: str = Body(..., embed=True),
    current_user: str = Depends(get_current_user)
):
    """Streaming chat endpoint - returns Server-Sent Events (SSE) format"""
    logger.info("\n" + "=" * 100)
    logger.info("🌊 NEW STREAMING CHAT REQUEST")
    logger.info("=" * 100)
    logger.info(f"👤 User ID: {current_user}")
    logger.info(f"💬 Message: {message}")

    if not message or not message.strip():
        logger.error("❌ Empty message received")
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    async def event_stream():
        try:
            # Get conversation context
            history = get_conversation_history(current_user)
            user_context = get_user_context(current_user)

            # Initialize state
            state = {
                "input": message.strip(),
                "user_id": current_user,
                "messages": history,
                "user_context": user_context,
                "intent_type": "unclear",
                "intent_details": {},
                "cypher_query": None,
                "query_results": [],
                "requires_action": False,
                "action_type": None,
                "output": "",
                "error": None
            }

            # Execute non-streaming parts (intent classification & query execution)
            logger.info("⚙️ Executing non-streaming workflow steps...")
            result = app_graph.invoke(state)

            # Execute Cypher query if present
            if result.get("cypher_query"):
                logger.info("🔄 Executing Cypher query...")
                query_results = execute_cypher_query(
                    neo4j_connection, result["cypher_query"])
                result["query_results"] = query_results if query_results else []
                logger.info(
                    f"📊 Query Results Count: {len(result['query_results'])}")

            # Update context
            user_context["last_interaction"] = message
            user_context["current_task"] = result["intent_type"]
            result["user_context"] = user_context
            save_user_context(current_user, user_context)

            # Send metadata first
            metadata = {
                "type": "metadata",
                "intent_type": result["intent_type"],
                "intent_details": result.get("intent_details", {}),
                "cypher_query": result.get("cypher_query"),
                "results_count": len(result.get("query_results", [])),
                "requires_action": result.get("requires_action", False),
                "action_type": result.get("action_type")
            }
            yield json.dumps(metadata) + "\n"

            # Stream the response
            logger.info("📡 Starting response streaming...")
            async for chunk in generate_streaming_response(result):
                yield chunk

            # Save to history
            result["messages"].append({"role": "user", "content": message})
            result["messages"].append(
                {"role": "assistant", "content": result.get("output", "")})
            save_conversation_history(current_user, result["messages"])

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"❌ Streaming failed: {e}")
            error_data = {"type": "error", "message": str(e)}
            yield json.dumps(error_data) + "\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/chat")
async def chat(
    message: str = Body(..., embed=True),
    current_user: str = Depends(get_current_user)
):
    """Non-streaming chat endpoint (original functionality preserved)"""
    logger.info("\n" + "=" * 100)
    logger.info("💬 NEW CHAT REQUEST (NON-STREAMING)")
    logger.info("=" * 100)
    logger.info(f"👤 User ID: {current_user}")
    logger.info(f"📝 Message: {message}")

    if not message or not message.strip():
        logger.error("❌ Empty message received")
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        history = get_conversation_history(current_user)
        user_context = get_user_context(current_user)

        state = {
            "input": message.strip(),
            "user_id": current_user,
            "messages": history,
            "user_context": user_context,
            "intent_type": "unclear",
            "intent_details": {},
            "cypher_query": None,
            "query_results": [],
            "requires_action": False,
            "action_type": None,
            "output": "",
            "error": None
        }

        # Execute workflow
        result = app_graph.invoke(state)

        # Execute Cypher query if present
        if result.get("cypher_query"):
            logger.info("🔄 Executing Cypher query...")
            query_results = execute_cypher_query(
                neo4j_connection, result["cypher_query"])
            result["query_results"] = query_results if query_results else []

        # Update context
        user_context["last_interaction"] = message
        user_context["current_task"] = result["intent_type"]
        result["user_context"] = user_context
        save_user_context(current_user, user_context)

        # Generate non-streaming response if not already set
        if not result.get("output"):
            intent_type = result["intent_type"]
            if intent_type == "general":
                result["output"] = generate_general_response(result)
            elif intent_type == "unclear":
                pass  # Already set in classify_intent
            elif result.get("cypher_query") and not result.get("query_results"):
                result["output"] = generate_no_results_response(result)
            else:
                # Generate contextual response (non-streaming)
                prompt = f"""
You are a friendly, professional restaurant CRM assistant. Generate a natural response.

**Intent Type:** {result["intent_type"]}
**What User Asked:** "{message}"
**Data Retrieved:**
{json.dumps(result.get("query_results", [])[:20], indent=2)}

Generate a concise, helpful response.
"""
                result["output"] = llm.invoke(prompt).content.strip()

        # Save to history
        result["messages"].append({"role": "user", "content": message})
        result["messages"].append(
            {"role": "assistant", "content": result["output"]})
        save_conversation_history(current_user, result["messages"])

        query_results = result.get("query_results") or []

        response = {
            "success": True,
            "response": result["output"],
            "intent_type": result["intent_type"],
            "intent_details": result.get("intent_details", {}),
            "cypher_query": result.get("cypher_query"),
            "results_count": len(query_results),
            "requires_action": result.get("requires_action", False),
            "action_type": result.get("action_type")
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ CHAT REQUEST FAILED: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/history")
async def get_history(current_user: str = Depends(get_current_user)):
    history = get_conversation_history(current_user)
    return {"success": True, "history": history, "count": len(history)}


@app.delete("/chat/history")
async def clear_history(current_user: str = Depends(get_current_user)):
    redis_connection.delete(f"user:{current_user}:history")
    redis_connection.delete(f"user:{current_user}:context")
    return {"success": True, "message": "History cleared"}


# ====================================================
# --- Stats API Endpoints (for Dashboard) ---
# ====================================================

@app.get("/stats/overview")
async def get_stats_overview(current_user: str = Depends(get_current_user)):
    """Get overall restaurant statistics"""
    try:
        with neo4j_connection.get_session() as session:
            result = session.run("""
                MATCH (d:Dish) WITH count(d) AS dishes
                MATCH (c:Customer) WITH dishes, count(c) AS customers
                MATCH (o:Order) WITH dishes, customers, count(o) AS orders
                MATCH (r:Review) WITH dishes, customers, orders, count(r) AS reviews
                MATCH (i:Ingredient) WITH dishes, customers, orders, reviews, count(i) AS ingredients
                RETURN dishes, customers, orders, reviews, ingredients
            """).single()

            revenue_result = session.run("""
                MATCH (o:Order)
                RETURN SUM(o.total_amount) AS total_revenue,
                       AVG(o.total_amount) AS avg_order_value,
                       COUNT(o) AS total_orders
            """).single()

            rating_result = session.run("""
                MATCH (r:Review)
                RETURN AVG(r.rating) AS avg_rating,
                       COUNT(r) AS total_reviews
            """).single()

        return {
            "success": True,
            "statistics": {
                "total_dishes": result["dishes"] if result else 0,
                "total_customers": result["customers"] if result else 0,
                "total_orders": result["orders"] if result else 0,
                "total_reviews": result["reviews"] if result else 0,
                "total_ingredients": result["ingredients"] if result else 0,
                "total_revenue": float(revenue_result["total_revenue"] or 0),
                "avg_order_value": float(revenue_result["avg_order_value"] or 0),
                "avg_rating": float(rating_result["avg_rating"] or 0)
            }
        }
    except Exception as e:
        logger.error(f"❌ Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats/revenue-trend")
async def get_revenue_trend(days: int = 7, current_user: str = Depends(get_current_user)):
    """Get revenue trend for the specified number of days"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        with neo4j_connection.get_session() as session:
            result = session.run("""
                MATCH (o:Order)
                WHERE o.timestamp >= datetime($start_date) 
                  AND o.timestamp <= datetime($end_date)
                WITH date(o.timestamp) AS order_date, 
                     SUM(o.total_amount) AS daily_revenue
                RETURN toString(order_date) AS date, 
                       daily_revenue
                ORDER BY order_date ASC
            """, start_date=start_date.isoformat(), end_date=end_date.isoformat())

            records = [dict(record) for record in result]

            formatted_data = []
            total_revenue = 0

            for record in records:
                date_obj = datetime.fromisoformat(record['date'])
                day_name = date_obj.strftime('%a')
                revenue = float(record['daily_revenue'])

                formatted_data.append({
                    "date": record['date'],
                    "day": day_name,
                    "revenue": revenue
                })
                total_revenue += revenue

            average_daily_revenue = total_revenue / \
                len(formatted_data) if formatted_data else 0

            return {
                "success": True,
                "data": formatted_data,
                "total_revenue": round(total_revenue, 2),
                "average_daily_revenue": round(average_daily_revenue, 2),
                "days_count": len(formatted_data)
            }

    except Exception as e:
        logger.error(f"❌ Failed to get revenue trend: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats/top-dishes")
async def get_top_dishes(limit: int = 5, current_user: str = Depends(get_current_user)):
    """Get top dishes by order count"""
    try:
        with neo4j_connection.get_session() as session:
            dishes_result = session.run("""
                MATCH (o:Order)-[:CONTAINS]->(d:Dish)
                WITH d, 
                     COUNT(o) AS order_count,
                     d.price * COUNT(o) AS estimated_revenue
                RETURN d.name AS name,
                       d.type AS type,
                       d.cuisine AS cuisine,
                       d.price AS price,
                       order_count,
                       estimated_revenue,
                       d.popularity_score AS popularity
                ORDER BY order_count DESC
                LIMIT $limit
            """, limit=limit)

            dishes = [dict(record) for record in dishes_result]
            formatted_data = []
            total_dishes_sold = 0

            for dish in dishes:
                rating_result = session.run("""
                    MATCH (r:Review)-[:RATES]->(d:Dish {name: $dish_name})
                    RETURN AVG(r.rating) AS avg_rating, COUNT(r) AS review_count
                """, dish_name=dish['name']).single()

                avg_rating = float(
                    rating_result['avg_rating'] or 0) if rating_result else 0
                order_count = int(dish['order_count'])
                total_dishes_sold += order_count

                formatted_data.append({
                    "name": dish['name'],
                    "type": dish['type'],
                    "cuisine": dish['cuisine'],
                    "price": float(dish['price']),
                    "orders": order_count,
                    "revenue": round(float(dish['estimated_revenue']), 2),
                    "rating": round(avg_rating, 1),
                    "popularity_score": int(dish['popularity']) if dish['popularity'] else 0
                })

            return {
                "success": True,
                "data": formatted_data,
                "total_dishes_sold": total_dishes_sold,
                "limit": limit
            }

    except Exception as e:
        logger.error(f"❌ Failed to get top dishes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats/order-status")
async def get_order_status_distribution(current_user: str = Depends(get_current_user)):
    """Get order status distribution"""
    try:
        with neo4j_connection.get_session() as session:
            result = session.run("""
                MATCH (o:Order)
                WITH o.status AS status, COUNT(o) AS count
                WITH collect({status: status, count: count}) AS status_data,
                     SUM(count) AS total
                UNWIND status_data AS sd
                RETURN sd.status AS status,
                       sd.count AS count,
                       total,
                       round(toFloat(sd.count) / toFloat(total) * 100, 1) AS percentage
                ORDER BY count DESC
            """)

            records = [dict(record) for record in result]
            data = {}
            total_orders = 0

            for record in records:
                status = record['status'] or 'unknown'
                count = int(record['count'])
                percentage = float(record['percentage'])

                data[status] = {
                    "count": count,
                    "percentage": percentage
                }
                total_orders = int(record['total'])

            common_statuses = ['delivered',
                               'in_progress', 'cancelled', 'pending']
            for status in common_statuses:
                if status not in data:
                    data[status] = {"count": 0, "percentage": 0.0}

            return {
                "success": True,
                "data": data,
                "total_orders": total_orders
            }

    except Exception as e:
        logger.error(f"❌ Failed to get order status distribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ====================================================
# --- Frontend Data API Endpoints ---
# ====================================================

@app.get("/orders")
async def get_orders(
    status: Optional[str] = None,
    current_user: str = Depends(get_current_user)
):
    """Get orders list with optional status filter"""
    try:
        with neo4j_connection.get_session() as session:
            query = """
                MATCH (c:Customer)-[:ORDERED]->(o:Order)
                OPTIONAL MATCH (o)-[:CONTAINS]->(d:Dish)
            """

            params = {}
            if status:
                query += " WHERE o.status = $status"
                params["status"] = status

            query += """
                WITH c, o, COLLECT({
                    dish_name: d.name,
                    quantity: 1,
                    price: d.price
                }) AS items
                RETURN o.order_id AS order_id,
                       c.name AS customer_name,
                       items,
                       o.total_amount AS total_amount,
                       o.status AS status,
                       toString(o.timestamp) AS timestamp,
                       CASE 
                           WHEN o.delivery_time IS NOT NULL 
                           THEN toString(o.delivery_time) + ' mins'
                           ELSE 'Pending'
                       END AS delivery_time,
                       c.location AS location
                ORDER BY o.timestamp DESC
            """

            result = session.run(query, **params)
            orders = []

            for record in result:
                order = {
                    "order_id": record["order_id"],
                    "customer_name": record["customer_name"],
                    "items": [item for item in record["items"] if item["dish_name"]],
                    "total_amount": float(record["total_amount"]),
                    "status": record["status"],
                    "timestamp": record["timestamp"],
                    "delivery_time": record["delivery_time"],
                    "location": record["location"]
                }
                orders.append(order)

            return {
                "success": True,
                "orders": orders,
                "total_count": len(orders)
            }

    except Exception as e:
        logger.error(f"❌ Failed to get orders: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/dishes")
async def get_dishes(
    cuisine: Optional[str] = None,
    type: Optional[str] = None,
    sort: Optional[str] = None,
    current_user: str = Depends(get_current_user)
):
    """Get dishes catalog with filters and sorting"""
    try:
        with neo4j_connection.get_session() as session:
            query = """
                MATCH (d:Dish)
                OPTIONAL MATCH (o:Order)-[:CONTAINS]->(d)
                OPTIONAL MATCH (r:Review)-[:RATES]->(d)
                OPTIONAL MATCH (d)-[:CONTAINS_INGREDIENT]->(i:Ingredient)
            """

            where_clauses = []
            params = {}

            if cuisine:
                where_clauses.append("toLower(d.cuisine) = toLower($cuisine)")
                params["cuisine"] = cuisine

            if type:
                where_clauses.append("toLower(d.type) = toLower($type)")
                params["type"] = type

            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)

            query += """
                WITH d,
                     COUNT(DISTINCT o) AS order_count,
                     COUNT(DISTINCT r) AS review_count,
                     AVG(r.rating) AS avg_rating,
                     COLLECT(DISTINCT i.name) AS ingredients
                RETURN d.name AS name,
                       d.cuisine AS cuisine,
                       d.type AS type,
                       d.price AS price,
                       COALESCE(avg_rating, d.popularity_score / 20.0, 0) AS popularity,
                       order_count AS orders,
                       review_count AS reviews,
                       ingredients
            """

            if sort == "popularity":
                query += " ORDER BY popularity DESC"
            elif sort == "price-low":
                query += " ORDER BY price ASC"
            elif sort == "price-high":
                query += " ORDER BY price DESC"
            elif sort == "orders":
                query += " ORDER BY orders DESC"
            else:
                query += " ORDER BY popularity DESC"

            result = session.run(query, **params)
            dishes = []

            for record in result:
                dish = {
                    "dish_id": f"D{len(dishes) + 1:03d}",
                    "name": record["name"],
                    "cuisine": record["cuisine"],
                    "type": record["type"],
                    "price": float(record["price"]),
                    "popularity": round(float(record["popularity"]), 1),
                    "orders": int(record["orders"]),
                    "reviews": int(record["reviews"]),
                    "ingredients": record["ingredients"]
                }
                dishes.append(dish)

            return {
                "success": True,
                "dishes": dishes,
                "total_count": len(dishes)
            }

    except Exception as e:
        logger.error(f"❌ Failed to get dishes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/reviews")
async def get_reviews(
    sentiment: Optional[str] = None,
    current_user: str = Depends(get_current_user)
):
    """Get customer reviews with optional sentiment filter"""
    try:
        with neo4j_connection.get_session() as session:
            query = """
                MATCH (c:Customer)-[:LEFT_REVIEW]->(r:Review)-[:RATES]->(d:Dish)
            """

            params = {}
            if sentiment:
                query += " WHERE toLower(r.sentiment) = toLower($sentiment)"
                params["sentiment"] = sentiment

            query += """
                RETURN r.id AS review_id,
                       c.name AS customer_name,
                       d.name AS dish_name,
                       r.rating AS rating,
                       r.sentiment AS sentiment,
                       r.feedback_text AS feedback,
                       toString(r.timestamp) AS timestamp
                ORDER BY r.timestamp DESC
            """

            result = session.run(query, **params)
            reviews = []

            for record in result:
                review = {
                    "review_id": record["review_id"],
                    "customer_name": record["customer_name"],
                    "dish_name": record["dish_name"],
                    "rating": int(record["rating"]),
                    "sentiment": record["sentiment"],
                    "feedback": record["feedback"],
                    "timestamp": record["timestamp"]
                }
                reviews.append(review)

            # Get statistics
            stats_query = """
                MATCH (r:Review)
                WITH AVG(r.rating) AS avg_rating,
                     COUNT(r) AS total,
                     SUM(CASE WHEN toLower(r.sentiment) = 'positive' THEN 1 ELSE 0 END) AS positive,
                     SUM(CASE WHEN toLower(r.sentiment) = 'neutral' THEN 1 ELSE 0 END) AS neutral,
                     SUM(CASE WHEN toLower(r.sentiment) = 'negative' THEN 1 ELSE 0 END) AS negative
                RETURN avg_rating, total, positive, neutral, negative
            """
            stats = session.run(stats_query).single()

            return {
                "success": True,
                "reviews": reviews,
                "average_rating": round(float(stats["avg_rating"] or 0), 1),
                "sentiment_counts": {
                    "positive": int(stats["positive"] or 0),
                    "neutral": int(stats["neutral"] or 0),
                    "negative": int(stats["negative"] or 0)
                },
                "total_count": int(stats["total"] or 0)
            }

    except Exception as e:
        logger.error(f"❌ Failed to get reviews: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ====================================================
# --- Analytics API Endpoints ---
# ====================================================

@app.get("/analytics/customer-growth")
async def get_customer_growth(
    months: int = 6,
    current_user: str = Depends(get_current_user)
):
    """Get customer growth trend over specified months"""
    try:
        with neo4j_connection.get_session() as session:
            query = """
                MATCH (c:Customer)
                WHERE c.join_date IS NOT NULL
                WITH c, date(c.join_date) AS join_date
                WHERE join_date >= date() - duration({months: $months})
                WITH date.truncate('month', join_date) AS month_date, COUNT(c) AS new_customers
                WITH month_date, new_customers
                ORDER BY month_date
                WITH COLLECT({month: month_date, new: new_customers}) AS monthly_data
                UNWIND range(0, size(monthly_data)-1) AS idx
                WITH monthly_data[idx].month AS month_date,
                     REDUCE(total = 0, i IN range(0, idx) | 
                         total + monthly_data[i].new) + monthly_data[idx].new AS cumulative
                RETURN month_date.year AS year,
                       month_date.month AS month_num,
                       cumulative AS customers
                ORDER BY year, month_num
            """

            result = session.run(query, months=months)
            data = []

            month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

            for record in result:
                month_num = int(record["month_num"])
                data.append({
                    "month": month_names[month_num - 1],
                    "customers": int(record["customers"])
                })

            # If no data, generate sample data
            if not data:
                base = 1000
                current_month = datetime.now().month
                for i in range(months):
                    month_idx = (current_month - months + i - 1) % 12
                    data.append({
                        "month": month_names[month_idx],
                        "customers": base + (i * 150)
                    })

            return {
                "success": True,
                "data": data
            }

    except Exception as e:
        logger.error(f"❌ Failed to get customer growth: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/order-heatmap")
async def get_order_heatmap(current_user: str = Depends(get_current_user)):
    """Get order heatmap by hour of day"""
    try:
        with neo4j_connection.get_session() as session:
            query = """
                MATCH (o:Order)
                WHERE o.timestamp IS NOT NULL
                WITH o.timestamp.hour AS hour, COUNT(o) AS order_count
                RETURN hour, order_count
                ORDER BY hour
            """

            result = session.run(query)
            hour_data = {i: 0 for i in range(24)}

            for record in result:
                hour_data[int(record["hour"])] = int(record["order_count"])

            data = []
            for hour in range(24):
                if hour == 0:
                    hour_label = "12 AM"
                elif hour < 12:
                    hour_label = f"{hour} AM"
                elif hour == 12:
                    hour_label = "12 PM"
                else:
                    hour_label = f"{hour - 12} PM"

                data.append({
                    "hour": hour_label,
                    "orders": hour_data.get(hour, 0)
                })

            return {
                "success": True,
                "data": data
            }

    except Exception as e:
        logger.error(f"❌ Failed to get order heatmap: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/top-customers")
async def get_top_customers(
    limit: int = 10,
    current_user: str = Depends(get_current_user)
):
    """Get top customers by orders and spending"""
    try:
        with neo4j_connection.get_session() as session:
            query = """
                MATCH (c:Customer)-[:ORDERED]->(o:Order)
                WITH c,
                     COUNT(o) AS order_count,
                     SUM(o.total_amount) AS total_spent
                RETURN c.name AS name,
                       order_count AS orders,
                       total_spent AS spent,
                       c.loyalty_score AS loyalty_score
                ORDER BY total_spent DESC
                LIMIT $limit
            """

            result = session.run(query, limit=limit)
            customers = []

            for record in result:
                customer = {
                    "name": record["name"],
                    "orders": int(record["orders"]),
                    "spent": round(float(record["spent"]), 2),
                    "loyalty_score": int(record["loyalty_score"] or 0)
                }
                customers.append(customer)

            return {
                "success": True,
                "customers": customers
            }

    except Exception as e:
        logger.error(f"❌ Failed to get top customers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/dish-performance")
async def get_dish_performance(current_user: str = Depends(get_current_user)):
    """Get dish performance (revenue vs rating)"""
    try:
        with neo4j_connection.get_session() as session:
            query = """
                MATCH (d:Dish)
                OPTIONAL MATCH (o:Order)-[:CONTAINS]->(d)
                OPTIONAL MATCH (r:Review)-[:RATES]->(d)
                WITH d,
                     COUNT(DISTINCT o) AS order_count,
                     AVG(r.rating) AS avg_rating
                WHERE order_count > 0
                RETURN d.name AS name,
                       (order_count * d.price) AS revenue,
                       COALESCE(avg_rating, 0) AS rating
                ORDER BY revenue DESC
                LIMIT 20
            """

            result = session.run(query)
            dishes = []

            for record in result:
                dish = {
                    "name": record["name"],
                    "revenue": round(float(record["revenue"]), 2),
                    "rating": round(float(record["rating"]), 1)
                }
                dishes.append(dish)

            return {
                "success": True,
                "dishes": dishes
            }

    except Exception as e:
        logger.error(f"❌ Failed to get dish performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analytics/cuisine-distribution")
async def get_cuisine_distribution(current_user: str = Depends(get_current_user)):
    """Get cuisine distribution by orders"""
    try:
        with neo4j_connection.get_session() as session:
            query = """
                MATCH (o:Order)-[:CONTAINS]->(d:Dish)
                WITH d.cuisine AS cuisine, COUNT(o) AS order_count
                WITH COLLECT({cuisine: cuisine, orders: order_count}) AS cuisine_data,
                     SUM(order_count) AS total_orders
                UNWIND cuisine_data AS cd
                RETURN cd.cuisine AS name,
                       round(toFloat(cd.orders) / toFloat(total_orders) * 100, 1) AS percentage,
                       cd.orders AS orders
                ORDER BY orders DESC
            """

            result = session.run(query)
            cuisines = []

            for record in result:
                cuisine = {
                    "name": record["name"],
                    "percentage": float(record["percentage"]),
                    "orders": int(record["orders"])
                }
                cuisines.append(cuisine)

            return {
                "success": True,
                "cuisines": cuisines
            }

    except Exception as e:
        logger.error(f"❌ Failed to get cuisine distribution: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ====================================================
# --- Setup & Bulk Data Upload Endpoints ---
# ====================================================

@app.post("/setup/initialize-schema")
async def initialize_schema(current_user: str = Depends(get_current_user)):
    """Initialize Neo4j database schema with constraints and indexes"""
    try:
        with neo4j_connection.get_session() as session:
            # Create constraints
            constraints = [
                "CREATE CONSTRAINT customer_id IF NOT EXISTS FOR (c:Customer) REQUIRE c.id IS UNIQUE",
                "CREATE CONSTRAINT dish_name IF NOT EXISTS FOR (d:Dish) REQUIRE d.name IS UNIQUE",
                "CREATE CONSTRAINT order_id IF NOT EXISTS FOR (o:Order) REQUIRE o.order_id IS UNIQUE",
                "CREATE CONSTRAINT ingredient_name IF NOT EXISTS FOR (i:Ingredient) REQUIRE i.name IS UNIQUE",
            ]

            # Create indexes for performance
            indexes = [
                "CREATE INDEX customer_email IF NOT EXISTS FOR (c:Customer) ON (c.email)",
                "CREATE INDEX customer_loyalty IF NOT EXISTS FOR (c:Customer) ON (c.loyalty_score)",
                "CREATE INDEX dish_type IF NOT EXISTS FOR (d:Dish) ON (d.type)",
                "CREATE INDEX dish_cuisine IF NOT EXISTS FOR (d:Dish) ON (d.cuisine)",
                "CREATE INDEX order_timestamp IF NOT EXISTS FOR (o:Order) ON (o.timestamp)",
                "CREATE INDEX review_rating IF NOT EXISTS FOR (r:Review) ON (r.rating)",
                "CREATE INDEX review_sentiment IF NOT EXISTS FOR (r:Review) ON (r.sentiment)",
            ]

            for constraint in constraints:
                try:
                    session.run(constraint)
                    logger.info(f"✅ Created: {constraint[:50]}")
                except Exception as e:
                    logger.warning(
                        f"⚠️ Constraint already exists or error: {e}")

            for index in indexes:
                try:
                    session.run(index)
                    logger.info(f"✅ Created: {index[:50]}")
                except Exception as e:
                    logger.warning(f"⚠️ Index already exists or error: {e}")

        return {
            "success": True,
            "message": "Schema initialized successfully",
            "constraints_created": len(constraints),
            "indexes_created": len(indexes)
        }

    except Exception as e:
        logger.error(f"❌ Schema initialization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/setup/bulk-upload")
async def bulk_upload_data(
    customers: Optional[List[Dict]] = Body(default=[]),
    dishes: Optional[List[Dict]] = Body(default=[]),
    ingredients: Optional[List[Dict]] = Body(default=[]),
    orders: Optional[List[Dict]] = Body(default=[]),
    reviews: Optional[List[Dict]] = Body(default=[]),
    dish_ingredients: Optional[List[Dict]] = Body(default=[]),
    customer_preferences: Optional[List[Dict]] = Body(default=[]),
    current_user: str = Depends(get_current_user)
):
    """Bulk upload restaurant data to Neo4j"""

    results = {
        "customers_added": 0,
        "dishes_added": 0,
        "ingredients_added": 0,
        "orders_added": 0,
        "reviews_added": 0,
        "relationships_created": 0,
        "errors": []
    }

    try:
        with neo4j_connection.get_session() as session:
            # 1. Add Customers
            logger.info("📝 Adding customers...")
            for customer in customers:
                try:
                    session.run("""
                        MERGE (c:Customer {id: $id})
                        SET c.name = $name,
                            c.email = $email,
                            c.phone = $phone,
                            c.location = $location,
                            c.loyalty_score = $loyalty_score,
                            c.join_date = $join_date,
                            c.updated_at = datetime()
                    """, **customer)
                    results["customers_added"] += 1
                except Exception as e:
                    results["errors"].append(
                        f"Customer {customer.get('id')}: {str(e)}")

            # 2. Add Dishes
            logger.info("🍽️ Adding dishes...")
            for dish in dishes:
                try:
                    session.run("""
                        MERGE (d:Dish {name: $name})
                        SET d.type = $type,
                            d.price = $price,
                            d.popularity_score = $popularity_score,
                            d.cuisine = $cuisine,
                            d.category = $category,
                            d.description = $description,
                            d.updated_at = datetime()
                    """, **dish)
                    results["dishes_added"] += 1
                except Exception as e:
                    results["errors"].append(
                        f"Dish {dish.get('name')}: {str(e)}")

            # 3. Add Ingredients
            logger.info("🥗 Adding ingredients...")
            for ingredient in ingredients:
                try:
                    session.run("""
                        MERGE (i:Ingredient {name: $name})
                        SET i.allergy_info = $allergy_info,
                            i.is_vegan = $is_vegan,
                            i.nutritional_info = $nutritional_info,
                            i.updated_at = datetime()
                    """, **ingredient)
                    results["ingredients_added"] += 1
                except Exception as e:
                    results["errors"].append(
                        f"Ingredient {ingredient.get('name')}: {str(e)}")

            # 4. Add Orders and create relationships
            logger.info("📦 Adding orders...")
            for order in orders:
                try:
                    # Create order node
                    session.run("""
                        MERGE (o:Order {order_id: $order_id})
                        SET o.timestamp = datetime($timestamp),
                            o.total_amount = $total_amount,
                            o.status = $status,
                            o.delivery_time = $delivery_time,
                            o.updated_at = datetime()
                    """,
                                order_id=order["order_id"],
                                timestamp=order["timestamp"],
                                total_amount=order["total_amount"],
                                status=order.get("status", "completed"),
                                delivery_time=order.get("delivery_time")
                                )

                    # Link customer to order
                    session.run("""
                        MATCH (c:Customer {id: $customer_id})
                        MATCH (o:Order {order_id: $order_id})
                        MERGE (c)-[:ORDERED]->(o)
                    """, customer_id=order["customer_id"], order_id=order["order_id"])

                    # Link dishes to order
                    for dish_name in order.get("dishes", []):
                        session.run("""
                            MATCH (o:Order {order_id: $order_id})
                            MATCH (d:Dish {name: $dish_name})
                            MERGE (o)-[:CONTAINS]->(d)
                        """, order_id=order["order_id"], dish_name=dish_name)
                        results["relationships_created"] += 1

                    results["orders_added"] += 1
                    results["relationships_created"] += 1

                except Exception as e:
                    results["errors"].append(
                        f"Order {order.get('order_id')}: {str(e)}")

            # 5. Add Reviews and create relationships
            logger.info("⭐ Adding reviews...")
            for review in reviews:
                try:
                    review_id = f"R_{review['customer_id']}_{review['dish_name']}_{review['timestamp']}"

                    # Create review node
                    session.run("""
                        MERGE (r:Review {id: $review_id})
                        SET r.rating = $rating,
                            r.feedback_text = $feedback_text,
                            r.sentiment = $sentiment,
                            r.timestamp = datetime($timestamp),
                            r.updated_at = datetime()
                    """,
                                review_id=review_id,
                                rating=review["rating"],
                                feedback_text=review.get("feedback_text", ""),
                                sentiment=review.get("sentiment", "neutral"),
                                timestamp=review["timestamp"]
                                )

                    # Link customer to review
                    session.run("""
                        MATCH (c:Customer {id: $customer_id})
                        MATCH (r:Review {id: $review_id})
                        MERGE (c)-[:LEFT_REVIEW]->(r)
                    """, customer_id=review["customer_id"], review_id=review_id)

                    # Link review to dish
                    session.run("""
                        MATCH (r:Review {id: $review_id})
                        MATCH (d:Dish {name: $dish_name})
                        MERGE (r)-[:RATES]->(d)
                    """, review_id=review_id, dish_name=review["dish_name"])

                    results["reviews_added"] += 1
                    results["relationships_created"] += 2

                except Exception as e:
                    results["errors"].append(f"Review: {str(e)}")

            # 6. Link Dishes to Ingredients
            logger.info("🔗 Linking dishes to ingredients...")
            for link in dish_ingredients:
                try:
                    session.run("""
                        MATCH (d:Dish {name: $dish_name})
                        MATCH (i:Ingredient {name: $ingredient_name})
                        MERGE (d)-[:CONTAINS_INGREDIENT]->(i)
                    """, dish_name=link["dish_name"], ingredient_name=link["ingredient_name"])
                    results["relationships_created"] += 1
                except Exception as e:
                    results["errors"].append(f"Dish-Ingredient link: {str(e)}")

            # 7. Link Customer Preferences
            logger.info("❤️ Creating customer preferences...")
            for pref in customer_preferences:
                try:
                    session.run("""
                        MATCH (c:Customer {id: $customer_id})
                        MATCH (d:Dish {name: $dish_name})
                        MERGE (c)-[:PREFERS]->(d)
                    """, customer_id=pref["customer_id"], dish_name=pref["dish_name"])
                    results["relationships_created"] += 1
                except Exception as e:
                    results["errors"].append(f"Customer preference: {str(e)}")

        logger.info(f"✅ Bulk upload completed: {results}")

        return {
            "success": True,
            "message": "Bulk upload completed successfully",
            "summary": {
                "customers_added": results["customers_added"],
                "dishes_added": results["dishes_added"],
                "ingredients_added": results["ingredients_added"],
                "orders_added": results["orders_added"],
                "reviews_added": results["reviews_added"],
                "relationships_created": results["relationships_created"],
                "total_errors": len(results["errors"])
            },
            "errors": results["errors"][:10] if results["errors"] else []
        }

    except Exception as e:
        logger.error(f"❌ Bulk upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/setup/sample-data")
async def load_sample_data(current_user: str = Depends(get_current_user)):
    """Load sample restaurant data for testing"""
    sample_customers = [
        {"id": "C001", "name": "Alice Johnson", "email": "alice@email.com", "phone": "+1234567890",
         "location": "New York", "loyalty_score": 95, "join_date": "2024-01-15"},
        {"id": "C002", "name": "Bob Smith", "email": "bob@email.com", "phone": "+1234567891",
         "location": "Los Angeles", "loyalty_score": 78, "join_date": "2024-02-20"},
        {"id": "C003", "name": "Carol White", "email": "carol@email.com", "phone": "+1234567892",
         "location": "Chicago", "loyalty_score": 85, "join_date": "2024-03-10"},
        {"id": "C004", "name": "David Brown", "email": "david@email.com", "phone": "+1234567893",
         "location": "Houston", "loyalty_score": 62, "join_date": "2024-04-05"},
        {"id": "C005", "name": "Emma Davis", "email": "emma@email.com", "phone": "+1234567894",
         "location": "New York", "loyalty_score": 90, "join_date": "2024-01-25"}
    ]

    sample_dishes = [
        {"name": "Margherita Pizza", "type": "veg", "price": 12.99, "popularity_score": 92,
         "cuisine": "Italian", "category": "Main Course", "description": "Classic tomato and mozzarella"},
        {"name": "Chicken Tikka Masala", "type": "non-veg", "price": 15.99, "popularity_score": 88,
         "cuisine": "Indian", "category": "Main Course", "description": "Spicy chicken in creamy sauce"},
        {"name": "Caesar Salad", "type": "veg", "price": 8.99, "popularity_score": 75,
         "cuisine": "American", "category": "Appetizer", "description": "Fresh romaine with parmesan"},
        {"name": "Vegan Buddha Bowl", "type": "veg", "price": 13.99, "popularity_score": 82,
         "cuisine": "Fusion", "category": "Main Course", "description": "Quinoa, vegetables, tahini"},
        {"name": "Beef Burger", "type": "non-veg", "price": 11.99, "popularity_score": 90,
         "cuisine": "American", "category": "Main Course", "description": "Juicy beef patty with fries"},
        {"name": "Pad Thai", "type": "veg", "price": 12.49, "popularity_score": 85,
         "cuisine": "Thai", "category": "Main Course", "description": "Stir-fried rice noodles"},
        {"name": "Chocolate Lava Cake", "type": "veg", "price": 6.99, "popularity_score": 95,
         "cuisine": "Dessert", "category": "Dessert", "description": "Warm chocolate with molten center"},
        {"name": "Grilled Salmon", "type": "non-veg", "price": 18.99, "popularity_score": 87,
         "cuisine": "Seafood", "category": "Main Course", "description": "Fresh Atlantic salmon"}
    ]

    sample_ingredients = [
        {"name": "Tomato", "allergy_info": "none",
            "is_vegan": True, "nutritional_info": "Vitamin C"},
        {"name": "Mozzarella", "allergy_info": "dairy",
            "is_vegan": False, "nutritional_info": "Calcium, Protein"},
        {"name": "Chicken", "allergy_info": "none",
            "is_vegan": False, "nutritional_info": "High Protein"},
        {"name": "Peanuts", "allergy_info": "nuts", "is_vegan": True,
            "nutritional_info": "Protein, Healthy Fats"},
        {"name": "Quinoa", "allergy_info": "none", "is_vegan": True,
            "nutritional_info": "Complete Protein"},
        {"name": "Beef", "allergy_info": "none",
            "is_vegan": False, "nutritional_info": "Iron, B12"},
        {"name": "Rice Noodles", "allergy_info": "gluten-free",
            "is_vegan": True, "nutritional_info": "Carbs"},
        {"name": "Chocolate", "allergy_info": "may contain dairy",
            "is_vegan": False, "nutritional_info": "Antioxidants"},
        {"name": "Salmon", "allergy_info": "fish", "is_vegan": False,
            "nutritional_info": "Omega-3, Protein"}
    ]

    sample_orders = [
        {"order_id": "O001", "customer_id": "C001", "timestamp": "2025-01-15T18:30:00",
         "total_amount": 21.98, "status": "delivered", "delivery_time": 35, "dishes": ["Margherita Pizza", "Caesar Salad"]},
        {"order_id": "O002", "customer_id": "C002", "timestamp": "2025-01-16T19:00:00",
         "total_amount": 27.98, "status": "delivered", "delivery_time": 40, "dishes": ["Chicken Tikka Masala", "Beef Burger"]},
        {"order_id": "O003", "customer_id": "C001", "timestamp": "2025-01-17T20:15:00",
         "total_amount": 20.98, "status": "delivered", "delivery_time": 30, "dishes": ["Vegan Buddha Bowl", "Chocolate Lava Cake"]},
        {"order_id": "O004", "customer_id": "C003", "timestamp": "2025-01-18T18:45:00",
         "total_amount": 31.48, "status": "delivered", "delivery_time": 45, "dishes": ["Grilled Salmon", "Pad Thai"]},
        {"order_id": "O005", "customer_id": "C004", "timestamp": "2025-01-19T19:30:00",
         "total_amount": 18.98, "status": "delivered", "delivery_time": 35, "dishes": ["Beef Burger", "Chocolate Lava Cake"]},
        {"order_id": "O006", "customer_id": "C005", "timestamp": "2025-01-20T17:30:00",
         "total_amount": 12.99, "status": "in_progress", "delivery_time": None, "dishes": ["Margherita Pizza"]}
    ]

    sample_reviews = [
        {"customer_id": "C001", "dish_name": "Margherita Pizza", "rating": 5,
         "feedback_text": "Best pizza in town! Perfectly cooked.", "sentiment": "positive", "timestamp": "2025-01-15T20:00:00"},
        {"customer_id": "C002", "dish_name": "Chicken Tikka Masala", "rating": 4,
         "feedback_text": "Good flavor but a bit too spicy for me.", "sentiment": "positive", "timestamp": "2025-01-16T20:30:00"},
        {"customer_id": "C001", "dish_name": "Vegan Buddha Bowl", "rating": 5,
         "feedback_text": "Healthy and delicious! Will order again.", "sentiment": "positive", "timestamp": "2025-01-17T21:00:00"},
        {"customer_id": "C003", "dish_name": "Grilled Salmon", "rating": 5,
         "feedback_text": "Fresh salmon, cooked to perfection!", "sentiment": "positive", "timestamp": "2025-01-18T20:00:00"},
        {"customer_id": "C004", "dish_name": "Beef Burger", "rating": 3,
         "feedback_text": "Burger was okay, fries were cold.", "sentiment": "neutral", "timestamp": "2025-01-19T21:00:00"},
        {"customer_id": "C001", "dish_name": "Chocolate Lava Cake", "rating": 5,
         "feedback_text": "Amazing dessert! Highly recommend.", "sentiment": "positive", "timestamp": "2025-01-17T21:30:00"}
    ]

    dish_ingredients = [
        {"dish_name": "Margherita Pizza", "ingredient_name": "Tomato"},
        {"dish_name": "Margherita Pizza", "ingredient_name": "Mozzarella"},
        {"dish_name": "Chicken Tikka Masala", "ingredient_name": "Chicken"},
        {"dish_name": "Vegan Buddha Bowl", "ingredient_name": "Quinoa"},
        {"dish_name": "Beef Burger", "ingredient_name": "Beef"},
        {"dish_name": "Pad Thai", "ingredient_name": "Rice Noodles"},
        {"dish_name": "Pad Thai", "ingredient_name": "Peanuts"},
        {"dish_name": "Chocolate Lava Cake", "ingredient_name": "Chocolate"},
        {"dish_name": "Grilled Salmon", "ingredient_name": "Salmon"}
    ]

    customer_preferences = [
        {"customer_id": "C001", "dish_name": "Margherita Pizza"},
        {"customer_id": "C001", "dish_name": "Vegan Buddha Bowl"},
        {"customer_id": "C002", "dish_name": "Chicken Tikka Masala"},
        {"customer_id": "C003", "dish_name": "Grilled Salmon"},
        {"customer_id": "C005", "dish_name": "Margherita Pizza"}
    ]

    # Call bulk upload with sample data
    return await bulk_upload_data(
        customers=sample_customers,
        dishes=sample_dishes,
        ingredients=sample_ingredients,
        orders=sample_orders,
        reviews=sample_reviews,
        dish_ingredients=dish_ingredients,
        customer_preferences=customer_preferences,
        current_user=current_user
    )


@app.delete("/setup/clear-all-data")
async def clear_all_data(
    confirm: bool = Body(..., embed=True),
    current_user: str = Depends(get_current_user)
):
    """Clear all data from Neo4j database (USE WITH CAUTION!)"""
    if not confirm:
        raise HTTPException(
            status_code=400, detail="Must confirm deletion by setting confirm=true")

    try:
        with neo4j_connection.get_session() as session:
            # Delete all nodes and relationships
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("🗑️ All data cleared from Neo4j")

        return {
            "success": True,
            "message": "All data cleared successfully",
            "warning": "This action cannot be undone"
        }

    except Exception as e:
        logger.error(f"❌ Failed to clear data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/setup/data-stats")
async def get_data_stats(current_user: str = Depends(get_current_user)):
    """Get statistics about the data in Neo4j"""
    try:
        with neo4j_connection.get_session() as session:
            stats = session.run("""
                MATCH (c:Customer) WITH count(c) AS customers
                MATCH (d:Dish) WITH customers, count(d) AS dishes
                MATCH (i:Ingredient) WITH customers, dishes, count(i) AS ingredients
                MATCH (o:Order) WITH customers, dishes, ingredients, count(o) AS orders
                MATCH (r:Review) WITH customers, dishes, ingredients, orders, count(r) AS reviews
                MATCH ()-[rel]->() WITH customers, dishes, ingredients, orders, reviews, count(rel) AS relationships
                RETURN customers, dishes, ingredients, orders, reviews, relationships
            """).single()

        return {
            "success": True,
            "statistics": {
                "customers": stats["customers"] if stats else 0,
                "dishes": stats["dishes"] if stats else 0,
                "ingredients": stats["ingredients"] if stats else 0,
                "orders": stats["orders"] if stats else 0,
                "reviews": stats["reviews"] if stats else 0,
                "relationships": stats["relationships"] if stats else 0
            }
        }

    except Exception as e:
        logger.error(f"❌ Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ====================================================
# --- Debug Endpoints ---
# ====================================================

@app.get("/debug/customers")
async def debug_customers(current_user: str = Depends(get_current_user)):
    """Debug endpoint to see all customers in Neo4j"""
    try:
        with neo4j_connection.get_session() as session:
            result = session.run("""
                MATCH (c:Customer)
                OPTIONAL MATCH (c)-[:ORDERED]->(o:Order)
                OPTIONAL MATCH (c)-[:PREFERS]->(d:Dish)
                WITH c, 
                     COUNT(DISTINCT o) AS total_orders, 
                     SUM(o.total_amount) AS total_spent,
                     COLLECT(DISTINCT d.name) AS preferred_dishes
                RETURN c.id AS id,
                       c.name AS name, 
                       c.email AS email, 
                       c.phone AS phone,
                       c.location AS location, 
                       c.loyalty_score AS loyalty_score,
                       c.join_date AS join_date,
                       total_orders,
                       total_spent,
                       preferred_dishes
                ORDER BY c.name
            """)
            customers = [dict(record) for record in result]

            count_result = session.run(
                "MATCH (c:Customer) RETURN count(c) AS total").single()

        return {
            "success": True,
            "total_customers": count_result["total"] if count_result else 0,
            "customers": customers
        }
    except Exception as e:
        logger.error(f"❌ Debug query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug/customer/{customer_id}")
async def debug_customer_by_id(
    customer_id: str,
    current_user: str = Depends(get_current_user)
):
    """Get detailed information about a specific customer by ID"""
    try:
        with neo4j_connection.get_session() as session:
            result = session.run("""
                MATCH (c:Customer {id: $customer_id})
                OPTIONAL MATCH (c)-[:ORDERED]->(o:Order)
                OPTIONAL MATCH (o)-[:CONTAINS]->(d:Dish)
                OPTIONAL MATCH (c)-[:LEFT_REVIEW]->(r:Review)-[:RATES]->(rd:Dish)
                OPTIONAL MATCH (c)-[:PREFERS]->(pd:Dish)
                WITH c,
                     COUNT(DISTINCT o) AS total_orders,
                     SUM(o.total_amount) AS total_spent,
                     COLLECT(DISTINCT {order_id: o.order_id, date: toString(o.timestamp), amount: o.total_amount, status: o.status}) AS orders,
                     COLLECT(DISTINCT d.name) AS ordered_dishes,
                     COLLECT(DISTINCT {dish: rd.name, rating: r.rating, feedback: r.feedback_text}) AS reviews,
                     COLLECT(DISTINCT pd.name) AS preferred_dishes
                RETURN c.id AS id,
                       c.name AS name,
                       c.email AS email,
                       c.phone AS phone,
                       c.location AS location,
                       c.loyalty_score AS loyalty_score,
                       c.join_date AS join_date,
                       total_orders,
                       total_spent,
                       orders,
                       ordered_dishes,
                       reviews,
                       preferred_dishes
            """, customer_id=customer_id).single()

            if not result:
                raise HTTPException(
                    status_code=404, detail=f"Customer {customer_id} not found")

        return {
            "success": True,
            "customer": dict(result)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Debug query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug/customer-by-name/{name}")
async def debug_customer_by_name(
    name: str,
    current_user: str = Depends(get_current_user)
):
    """Search for customers by name (case-insensitive partial match)"""
    try:
        with neo4j_connection.get_session() as session:
            result = session.run("""
                MATCH (c:Customer)
                WHERE toLower(c.name) CONTAINS toLower($name)
                OPTIONAL MATCH (c)-[:ORDERED]->(o:Order)
                OPTIONAL MATCH (c)-[:PREFERS]->(d:Dish)
                WITH c,
                     COUNT(DISTINCT o) AS total_orders,
                     SUM(o.total_amount) AS total_spent,
                     COLLECT(DISTINCT d.name) AS preferred_dishes
                RETURN c.id AS id,
                       c.name AS name,
                       c.email AS email,
                       c.phone AS phone,
                       c.location AS location,
                       c.loyalty_score AS loyalty_score,
                       c.join_date AS join_date,
                       total_orders,
                       total_spent,
                       preferred_dishes
                ORDER BY c.name
            """, name=name)

            customers = [dict(record) for record in result]

        return {
            "success": True,
            "search_term": name,
            "found": len(customers),
            "customers": customers
        }
    except Exception as e:
        logger.error(f"❌ Debug query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug/test-query")
async def debug_test_query(
    name: str = "Arjun",
    current_user: str = Depends(get_current_user)
):
    """Test a specific customer query"""
    try:
        with neo4j_connection.get_session() as session:
            logger.info(f"Testing query for name containing: {name}")

            result = session.run("""
                MATCH (c:Customer) 
                WHERE toLower(c.name) CONTAINS toLower($name)
                RETURN c.name AS name, c.id AS id, c.email AS email, 
                       c.location AS location, c.loyalty_score AS loyalty_score
            """, name=name)

            customers = [dict(record) for record in result]
            logger.info(f"Found {len(customers)} customers")

        return {
            "success": True,
            "search_term": name,
            "found": len(customers),
            "customers": customers
        }
    except Exception as e:
        logger.error(f"❌ Test query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/debug/all-nodes")
async def debug_all_nodes(current_user: str = Depends(get_current_user)):
    """Check what nodes exist in the database"""
    try:
        with neo4j_connection.get_session() as session:
            result = session.run("""
                CALL db.labels() YIELD label
                CALL {
                    WITH label
                    MATCH (n)
                    WHERE label IN labels(n)
                    RETURN count(n) AS count
                }
                RETURN label, count
                ORDER BY label
            """)

            labels = [dict(record) for record in result]

        return {
            "success": True,
            "node_labels": labels
        }
    except Exception as e:
        logger.error(f"❌ Debug query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
