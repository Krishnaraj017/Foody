"""
Database Connections Module
Handles Neo4j and Redis connections
"""
import os
import logging
from neo4j import GraphDatabase
from neo4j.time import DateTime
import redis
from contextlib import contextmanager
from datetime import datetime
from typing import List, Dict, Optional
import json
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ====================================================
# --- Neo4j Connection ---
# ====================================================


class Neo4jConnection:
    def __init__(self):
        try:
            self.driver = GraphDatabase.driver(
                "neo4j+s://05438154.databases.neo4j.io",
                auth=("neo4j", os.getenv("NEO4J_PASSWORD")),
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_acquisition_timeout=120
            )
            self.driver.verify_connectivity()
            logger.info("✅ Neo4j connected successfully")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Neo4j: {e}")
            raise

    @contextmanager
    def get_session(self):
        session = self.driver.session()
        try:
            yield session
        finally:
            session.close()

    def close(self):
        self.driver.close()
        logger.info("🔒 Neo4j connection closed")


# ====================================================
# --- Redis Connection ---
# ====================================================

class RedisConnection:
    def __init__(self):
        try:
            self.client = redis.Redis(
                host='redis-11505.c276.us-east-1-2.ec2.redns.redis-cloud.com',
                port=11505,
                decode_responses=True,
                username="default",
                password=os.getenv("REDIS_PASSWORD"),
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
            self.client.ping()
            logger.info("✅ Redis connected successfully")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            raise

    def get(self, key: str) -> Optional[str]:
        try:
            return self.client.get(key)
        except Exception as e:
            logger.error(f"❌ Redis GET failed for {key}: {e}")
            return None

    def set(self, key: str, value: str, ttl: int = None):
        try:
            if ttl:
                self.client.setex(key, ttl, value)
            else:
                self.client.set(key, value)
        except Exception as e:
            logger.error(f"❌ Redis SET failed for {key}: {e}")

    def setex(self, key: str, ttl: int, value: str):
        try:
            self.client.setex(key, ttl, value)
        except Exception as e:
            logger.error(f"❌ Redis SETEX failed for {key}: {e}")

    def delete(self, key: str):
        try:
            self.client.delete(key)
        except Exception as e:
            logger.error(f"❌ Redis DELETE failed for {key}: {e}")

    def close(self):
        self.client.close()
        logger.info("🔒 Redis connection closed")


# ====================================================
# --- Helper Functions ---
# ====================================================

def neo4j_to_json_serializable(obj):
    """Convert Neo4j types to JSON serializable types"""
    if isinstance(obj, DateTime):
        return obj.iso_format()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: neo4j_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [neo4j_to_json_serializable(item) for item in obj]
    return obj


def execute_cypher_query(neo4j_conn: Neo4jConnection, cypher_query: str, params: dict = None) -> List[Dict]:
    """Execute a Cypher query and return results"""
    logger.info("🔄 EXECUTING CYPHER QUERY")
    logger.info(f"📝 Query: {cypher_query}")
    logger.info(f"📊 Params: {params}")

    try:
        with neo4j_conn.get_session() as session:
            logger.info("🔓 Neo4j session opened")
            result = session.run(cypher_query, **(params or {}))
            logger.info("✅ Query executed successfully")

            records = []
            for record in result:
                record_dict = dict(record)
                serializable_record = neo4j_to_json_serializable(record_dict)
                records.append(serializable_record)

            logger.info(f"📈 Query returned {len(records)} records")

            if records:
                logger.info(
                    f"🔍 First record: {json.dumps(records[0], indent=2)}")
            else:
                logger.warning("⚠️ Query returned 0 records")

            return records
    except Exception as e:
        logger.error(f"❌ Query execution failed!")
        logger.error(f"🔴 Exception type: {type(e).__name__}")
        logger.error(f"🔴 Exception message: {str(e)}")
        logger.error(f"🔴 Failed query was: {cypher_query}")
        return []


# ====================================================
# --- Initialize Connections (Singleton Pattern) ---
# ====================================================

neo4j_connection = None
redis_connection = None


def initialize_connections():
    """Initialize database connections"""
    global neo4j_connection, redis_connection

    neo4j_connection = Neo4jConnection()
    redis_connection = RedisConnection()

    return neo4j_connection, redis_connection


def close_connections():
    """Close all database connections"""
    global neo4j_connection, redis_connection

    if neo4j_connection:
        neo4j_connection.close()
    if redis_connection:
        redis_connection.close()
