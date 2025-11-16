# **Foody AI CRM**  
### *A GenAI-powered restaurant CRM with conversational intelligence and real-time graph analytics.*

---

## 🧠 **Overview**

Foody AI CRM transforms raw restaurant data into **actionable insights** using a conversational AI assistant, a Neo4j knowledge graph, and an intelligent analytics dashboard.

Instead of digging through spreadsheets or BI tools, managers can simply ask:

- **“Show today’s top-selling dishes.”**  
- **“Who are my most loyal customers?”**  
- **“Summarize negative reviews from this week.”**  
- **“What’s the revenue trend for the last 7 days?”**

A combination of **LLM reasoning + graph analytics** enables deep insights into customers, orders, dishes, reviews, and performance — all through natural language.

---

## 🌐 **Live Dashboard**

Explore visual analytics, charts, customer insights, and the AI assistant:

👉 **https://v0-restaurant-crm-dashboard.vercel.app**

Includes:

- Revenue insights  
- Dish performance analytics  
- Customer segmentation  
- Order trends  
- Cuisine distribution  
- Heatmaps  
- Real-time chat assistant  
- Review sentiment intelligence  

---

## 🏗️ **Architecture Overview**

### **High-Level System Diagram**

```
                 ┌──────────────────────────────┐
                 │        Frontend (Next.js)     │
                 │  - Dashboard UI               │
                 │  - Charts, metrics, insights  │
                 │  - Chat interface             │
                 └──────────────┬───────────────┘
                                │
                                ▼
                 ┌──────────────────────────────┐
                 │        FastAPI Backend        │
                 │  - Chat endpoints             │
                 │  - Streaming responses        │
                 │  - Analytics API              │
                 │  - Auth + user context        │
                 └──────────────┬───────────────┘
                                │
                                ▼
                  ┌────────────────────────────┐
                  │       LangGraph Engine      │
                  │  - Intent classification    │
                  │  - Query generation logic   │
                  │  - Workflow management      │
                  └─────────────┬──────────────┘
                                │
                                ▼
                  ┌────────────────────────────┐
                  │    Groq Llama 3.3 Model     │
                  │  - Natural language reply   │
                  │  - Structured reasoning     │
                  │  - Cypher generation        │
                  └─────────────┬──────────────┘
                                │
                                ▼
    ┌────────────────────────────────────────────────────────┐
    │                       Datastores                        │
    │                                                        │
    │   ┌────────────────────────┐   ┌────────────────────┐  │
    │   │     Neo4j Graph DB     │   │     Redis Cache    │  │
    │   │ - Orders               │   │ - Auth tokens      │  │
    │   │ - Customers            │   │ - Session history  │  │
    │   │ - Dishes               │   │ - User context     │  │
    │   │ - Reviews              │   └────────────────────┘  │
    │   └────────────────────────┘                            │
    └────────────────────────────────────────────────────────┘
```

**Flow:**  
Frontend → FastAPI → LangGraph → LLM → Neo4j/Redis → Back to user.

---

## ✨ **Key Features**

### 🤖 **AI Chat Assistant**
Ask anything about:
- Orders  
- Customers  
- Reviews  
- Dish performance  
- Revenue trends  
- Analytics  

### 🕸️ **Graph-Based Analytics**
Neo4j relationships enable:
- VIP customer detection  
- Purchase patterns  
- Dish co-occurrence  
- Trend analysis  
- Review intelligence  

### ⚡ **Streaming Responses**
Powered by Groq’s ultra-fast Llama 3.3 models.

### 📊 **Modern Dashboard**
Includes:
- Metrics  
- Charts  
- Heatmaps  
- Segmentation  
- Real-time chat  

### 🧠 **Contextual Memory**
Redis stores:
- Chat history  
- User preferences  
- Conversation context  

### 🔌 **Plug-and-Play Integration**
Can connect to **any restaurant database**.

---

## 📈 **Why Foody AI CRM Matters**

Restaurant owners rarely have time to read dashboards or analyze data.

Foody AI CRM gives them:

- Instant data-driven insights  
- Natural language decision support  
- Automated summaries and recommendations  
- A complete 360° view of customers, orders, and performance  

It’s like having a **data analyst + CRM expert + operations manager** — available 24/7.

---

## 🔄 **Recommended Sync Methods (for Restaurants Integrating Their DB)**

If a restaurant wants to plug its existing database into Foody AI CRM, here are the **three practical ways** to keep Neo4j updated.

These are the same patterns used by modern SaaS CRMs.

---

### **1️⃣ Real-Time Event Sync (Most Practical for Restaurants)**  
Whenever the restaurant backend updates something (orders/customers/dishes), it sends a **simple webhook** to the Foody Sync API.

**Why it works great:**
- Real-time  
- Easy to implement  
- Just a POST request  
- Works with PHP, Node, Python, Java, Go — anything  
- Most restaurants can integrate in minutes  

---

### **2️⃣ Change Data Capture (CDC — Debezium/Kafka)**  
For medium/large restaurants or chains.

Reads DB logs directly → streams changes automatically.

**Why it’s powerful:**
- Zero code in restaurant backend  
- Fully real-time  
- Enterprise-grade reliability  
- Great for high-volume traffic  

---

### **3️⃣ Scheduled ETL Pull (No-Code Option)**  
Foody AI CRM periodically pulls new data from the restaurant database.

**Why restaurants love it:**
- They only share DB credentials  
- No dev work needed  
- Ideal for small/medium outlets  
- Syncing every 5 minutes is more than enough  

---

## ❤️ **Final Notes**

Foody AI CRM combines **GenAI**, **graph intelligence**, and **real-time analytics** into a unified platform — designed to give restaurants superpowers through data.

The goal is simple:

### **Make restaurant intelligence effortless.  
Make insights conversational.  
Make data accessible to everyone — instantly.**

---

## 🏢 **Foody AI CRM – Multi-Restaurant SaaS Architecture**

```
                       ┌────────────────────────────────────────┐
                       │          SaaS Admin Platform            │
                       │  - Restaurant onboarding UI             │
                       │  - Connect DB credentials               │
                       │  - Manage sync methods (Event/CDC/ETL)  │
                       └───────────────────────┬─────────────────┘
                                               │
                     ┌─────────────────────────┼────────────────────────┐
                     │                         │                        │
                     ▼                         ▼                        ▼

   ┌────────────────────────┐    ┌────────────────────────┐   ┌────────────────────────┐
   │   Restaurant A          │    │   Restaurant B          │   │   Restaurant C          │
   │   (MySQL / Postgres)   │    │   (MongoDB)             │   │   (Firestore / Other)   │
   └───────┬────────────────┘    └──────────┬─────────────┘   └──────────┬─────────────┘
           │                                 │                           │
           │   ┌──────────────────────────┐  │                           │
           │   │  Sync Method Options     │  │                           │
           │   │  1. Real-time Webhooks   │  │                           │
           ├──▶│  2. CDC (Debezium/Kafka) │  ├──────────────────────────▶│
           │   │  3. Scheduled ETL Pull   │  │                           │
           │   └──────────────────────────┘  │                           │
           │                                 │                           │
           ▼                                 ▼                           ▼

                   ┌────────────────────────────────────────────────────┐
                   │      Foody Sync Processor (Multi-Tenant)           │
                   │  - Normalizes data from each restaurant            │
                   │  - Converts rows/changes → Graph format            │
                   │  - Applies tenant isolation rules                  │
                   │  - Writes to the correct Neo4j subgraph           │
                   └───────────────────────┬────────────────────────────┘
                                           │
                                           ▼

            ┌───────────────────────────────────────────────────────────┐
            │                     Neo4j Multi-Tenant Graph               │
            │  - Each restaurant has its own namespace/subgraph          │
            │  - Data stored: Customers, Orders, Dishes, Reviews, etc.   │
            │  - Shared schema, isolated data                            │
            └─────────────────────────────────┬──────────────────────────┘
                                              │
                                              ▼

            ┌───────────────────────────────────────────────────────────┐
            │                      FastAPI Backend                       │
            │  - Multi-tenant auth + routing                             │
            │  - Chat and analytics APIs                                  │
            │  - Streams responses to frontend                            │
            └─────────────────────────────────┬──────────────────────────┘
                                              │
                                              ▼

            ┌───────────────────────────────────────────────────────────┐
            │                    LangGraph Orchestrator                  │
            │  - Intent classification                                   │
            │  - Cypher query generation                                 │
            │  - Workflow logic                                          │
            └─────────────────────────────────┬──────────────────────────┘
                                              │
                                              ▼

            ┌───────────────────────────────────────────────────────────┐
            │                  LLM (Groq Llama 3.3)                      │
            │  - Natural language generation                              │
            │  - Structured reasoning                                     │
            │  - Tenant-aware responses                                   │
            └─────────────────────────────────┬──────────────────────────┘
                                              │
                                              ▼

                       ┌────────────────────────────────────────────┐
                       │          Frontend (Next.js)                │
                       │  - Restaurant Dashboard                    │
                       │  - Real-time chat                          │
                       │  - Visual analytics                        │
                       └────────────────────────────────────────────┘


