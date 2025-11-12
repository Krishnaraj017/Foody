from neo4j import GraphDatabase

# Secure AuraDB connection
URI = "neo4j+s://05438154.databases.neo4j.io"
AUTH = ("neo4j", "_DWkybznFMKhgSFfabr1UYf6Wxx5z0eFSPmxcXsk5x4")

# Initialize driver (don't close yet)
driver = GraphDatabase.driver(URI, auth=AUTH)

# Test connection
driver.verify_connectivity()
print("✅ Connected securely to Neo4j AuraDB Cloud")


# --- Function to create data ---
def create_food_graph(driver):
    with driver.session() as session:
        # Create Dish nodes
        session.run("""
            MERGE (:Dish {name:'Idli', cuisine:'South Indian', type:'veg'})
            MERGE (:Dish {name:'Butter Chicken', cuisine:'North Indian', type:'non-veg'})
        """)
        # Create User and relationship
        session.run("""
            MERGE (u:User {name:'Krishnaraj'})
            WITH u
            MATCH (d:Dish {name:'Idli'})
            MERGE (u)-[:LIKES]->(d)
        """)
    print("✅ Sample graph created successfully")


# --- Function to recommend for user ---
def recommend_for_user(driver, user_name):
    with driver.session() as session:
        query = """
        MATCH (u:User {name: $name})-[:LIKES]->(d:Dish)
        RETURN d.name AS dish, d.cuisine AS cuisine
        """
        result = session.run(query, name=user_name)
        print(f"🍽 {user_name} likes:")
        for record in result:
            print(f" - {record['dish']} ({record['cuisine']})")


# --- Function to find similar users ---
def find_similar_users(driver, user_name):
    query = """
    MATCH (u1:User {name:$name})-[:LIKES]->(d:Dish)<-[:LIKES]-(u2:User)
    RETURN DISTINCT u2.name AS similar_user
    """
    with driver.session() as session:
        result = session.run(query, name=user_name)
        print(f"👥 Users similar to {user_name}:")
        for record in result:
            print(" -", record["similar_user"] or "None found")


# --- Run the flow ---
create_food_graph(driver)
recommend_for_user(driver, "Krishnaraj")
find_similar_users(driver, "Krishnaraj")

# Close driver properly
driver.close()
