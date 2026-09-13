# <div align="center"><img src="rahin_front_ai/public/assets/logo-mark.png" width="80" alt="Rahin Logo"></div>

# 🎯 AI Customer Behavior Analytics Platform

<div align="center">

![Rahin Analytics](rahin_front_ai/public/assets/logo-mark.png)

**Transform Customer Data Into Actionable Intelligence**

[![Python](https://img.shields.io/badge/Python-90.2%25-3776ab?logo=python&logoColor=white&style=for-the-badge)](https://python.org)
[![JavaScript](https://img.shields.io/badge/JavaScript-2.3%25-f7df1e?logo=javascript&logoColor=black&style=for-the-badge)](https://javascript.com)
[![CSS](https://img.shields.io/badge/CSS-5.6%25-1572b6?logo=css3&logoColor=white&style=for-the-badge)](https://www.w3.org/Style/CSS/)
[![HTML](https://img.shields.io/badge/HTML-1.9%25-e34c26?logo=html5&logoColor=white&style=for-the-badge)](https://html.spec.whatwg.org/)

<br/>

A sophisticated AI-powered analytics platform combining intelligent agents, semantic search, SQL analytics, and real-time dashboards to unlock deep insights from customer behavior data.

[![GitHub Stars](https://img.shields.io/github/stars/Rainsh724/ai-customer-behavior-analytics-platform?style=social)](https://github.com/Rainsh724/ai-customer-behavior-analytics-platform)
[![GitHub Forks](https://img.shields.io/github/forks/Rainsh724/ai-customer-behavior-analytics-platform?style=social)](https://github.com/Rainsh724/ai-customer-behavior-analytics-platform)

[📚 Features](#-key-features) • [🏗️ Architecture](#-architecture) • [🚀 Quick Start](#-quick-start) • [💻 Tech Stack](#-tech-stack) • [📖 Documentation](#-documentation)

</div>

---

## ✨ Key Features

### 🤖 **Intelligent Conversational Agent**
```
💬 Natural Language Understanding
├─ Persian & English query support
├─ Context-aware follow-up detection
├─ Multi-turn conversation memory
└─ Smart tool orchestration
```

- 🧠 **Advanced Understanding**: Comprehends complex business questions
- 💾 **Memory Management**: Preserves context across conversations
- 🔄 **Smart Follow-ups**: "Why?" questions automatically reference previous data
- 🎯 **Automatic Tool Selection**: Chooses SQL, RAG, or charts based on intent

### 📊 **Multi-Modal Analytics**
```
Data Analysis Toolkit
├─ SQL Analytics → Direct PostgreSQL queries
├─ RAG Search → Semantic customer review analysis  
├─ Chart Generation → Interactive visualizations
└─ Comparative Analysis → Period-over-period insights
```

| Feature | Capability |
|---------|-----------|
| **SQL Analytics** | Complex aggregations, rankings, time-series |
| **Semantic Search** | Customer review analysis with embeddings |
| **Time-Series** | 30-day rolling windows & custom periods |
| **Comparisons** | Month-over-month and year-over-year trends |

### 📈 **Executive Dashboards**
```
Real-Time Intelligence
┌─────────────────────────────────────────┐
│ 📊 Main Dashboard                       │
│ ├─ 30-day trend analysis                │
│ ├─ Top brands & products                │
│ └─ Customer segment distribution        │
│                                         │
│ 🏢 Brand Intelligence                   │
│ ├─ Performance metrics                  │
│ ├─ Conversion rates                     │
│ └─ Sentiment scores                     │
│                                         │
│ 🏷️ Category Analysis                    │
│ ├─ Sales tier classification            │
│ ├─ Customer satisfaction                │
│ └─ Conversion benchmarks                │
│                                         │
│ 👥 Customer Segmentation                │
│ ├─ VIP Champions (5%)                   │
│ ├─ Returning Customers (25%)            │
│ ├─ One-Time Buyers (35%)                │
│ ├─ Low Engagement (20%)                 │
│ └─ Window Shoppers (15%)                │
└─────────────────────────────────────────┘
```

### 💾 **Persistent Conversation Memory**
- 🗄️ **PostgreSQL Backend**: Reliable, scalable storage
- 🤖 **LLM-Powered Compaction**: Intelligent summarization of old turns
- 📊 **Evaluation Logging**: Track confidence, faithfulness, relevance
- 📈 **Calibration Monitoring**: Understand model reliability over time

---

## 🏗️ Architecture

### System Overview
```
┌──────────────────────────────────────────────────────────┐
│           🖥️  Frontend (React + Vite)                   │
│          rahin_front_ai/                                 │
│  ┌──────────────────────────────────────────┐           │
│  │ • Dashboards    • Chat Interface        │           │
│  │ • Charts        • Real-time Updates     │           │
│  └──────────────────────────────────────────┘           │
└────────────────────┬─────────────────────────────────────┘
                     │ HTTP/WebSocket
                     ▼
┌──────────────────────────────────────────────────────────┐
│        🚀 FastAPI Backend (api.py)                      │
│  ┌──────────────────────────────────────────┐           │
│  │ POST /api/chat          (Main endpoint)  │           │
│  │ GET  /api/dashboard     (Dashboard data) │           │
│  │ GET  /api/brand-intel   (Brand metrics)  │           │
│  │ GET  /api/category-intel (Categories)    │           │
│  │ GET  /api/segments      (Customer segs)  │           │
│  └──────────────────────────────────────────┘           │
│                   ▲                                       │
│         ┌─────────┼─────────┐                            │
│         │         │         │                            │
└─────────┼─────────┼─────────┼────────────────────────────┘
          │         │         │
          ▼         ▼         ▼
    ┌─────────┐ ┌──────────┐ ┌──────────────────┐
    │ 🧠 Main │ │ 💾 Memory│ │ ⚙️  Agentic     │
    │ Agent   │ │ Store    │ │    Graph        │
    │ (main)  │ │ (memory) │ │ (LangGraph)     │
    └────┬────┘ └──────────┘ └────────┬────────┘
         │                             │
         └──────────────┬──────────────┘
                        │
        ┌───────────────┼───────────────┐
        │               │               │
        ▼               ▼               ▼
    ┌────────┐     ┌────────┐    ┌──────────┐
    │📊 SQL  │     │🔍 RAG  │    │📈 Chart  │
    │Tool    │     │Tool    │    │ Tool     │
    └───┬────┘     └───┬────┘    └──────────┘
        │              │
        └──────────────┼──────────────┐
                       │              │
                ┌──────▼─────┐  ┌────▼───────┐
                │ PostgreSQL │  │ Embeddings │
                │(Analytics) │  │  Model     │
                └────────────┘  └────────────┘
```

### Component Breakdown

| Layer | Component | Technology |
|-------|-----------|-----------|
| **Presentation** | React Frontend | React 18, Vite, JavaScript |
| **API** | FastAPI Server | FastAPI, Python 3.10+, AsyncIO |
| **Agent** | LangGraph Engine | LangGraph, LangChain, OpenAI |
| **Memory** | Chat Storage | PostgreSQL, JSONB |
| **Analytics** | Tools | SQL, RAG, Chart Generation |
| **Data** | Databases | PostgreSQL (Read + Write Pools) |

---

## 🚀 Quick Start

### 📋 Prerequisites

```bash
✅ Python 3.10 or higher
✅ Node.js 18 or higher  
✅ PostgreSQL 13 or higher
✅ OpenAI API Key
```

### 1️⃣ Clone & Setup Environment

```bash
# Clone the repository
git clone https://github.com/Rainsh724/ai-customer-behavior-analytics-platform.git
cd ai-customer-behavior-analytics-platform

# Create environment file
cp .env.example .env
```

### 2️⃣ Configure `.env` File

```env
# 🗄️ Analytics Database (Read-Only)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=analytics_db
DB_USER=analytics_reader
DB_PASSWORD=your_secure_password

# 💬 Chat Memory Database (Write)
CHAT_DB_HOST=localhost
CHAT_DB_PORT=5432
CHAT_DB_NAME=postgres
CHAT_DB_USER=app_chat_writer
CHAT_DB_PASSWORD=your_secure_password

# 🤖 LLM Configuration
OPENAI_API_KEY=sk-your-key-here
LLM_MODEL=gpt-4-turbo-preview
EMBEDDING_MODEL=text-embedding-3-small

# ⚙️ Cache & Memory
DASHBOARD_CACHE_TTL_SECONDS=300
CHAT_MEMORY_MAX_RAW_TURNS=3
CHAT_DB_CONNECT_TIMEOUT=10
```

### 3️⃣ Install Backend Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt

# Initialize database schema
python -c "from main import main; main()"
```

### 4️⃣ Start Backend Server

```bash
# Run with auto-reload for development
uvicorn api:app --reload --port 8000

# Server runs at: http://localhost:8000
```

### 5️⃣ Install Frontend Dependencies

```bash
# Install Node packages
npm install

# Start development server
npm run dev

# Frontend runs at: http://localhost:5173
```

### 6️⃣ Verify Everything Works

```bash
# ✅ Check API health
curl http://localhost:8000/api/health

# ✅ Test chat endpoint
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "کدام محصولات بیشترین فروش را دارند؟",
    "session_id": "test-session-001"
  }'

# ✅ Load dashboard data
curl http://localhost:8000/api/dashboard
```

---

## 💻 Tech Stack

### 🐍 Backend Python Stack
```
FastAPI          → Modern, async web framework
LangGraph        → Agent orchestration & state management
LangChain        → LLM tooling & chains
OpenAI           → GPT-4 Turbo & text-embedding-3-small
PostgreSQL       → Persistent storage (dual connection pools)
Psycopg2         → Python PostgreSQL adapter
Pandas           → Data manipulation & export
```

### 🎨 Frontend Stack
```
React 18         → UI framework
Vite             → Fast build tool
JavaScript       → Dynamic interactions
CSS3             → Responsive styling
Chart.js         → Data visualization
Axios            → HTTP client
```

### 📊 Data & Analytics Stack
```
PostgreSQL       → Relational database
Vector Embeddings → Semantic search
LLM Models       → OpenAI API
SQL Queries      → Complex aggregations
Window Functions → Time-series analysis
```

---

## 📚 Documentation

### 💬 How to Ask Questions

The AI assistant understands various query types:

```
✅ Metrics Queries
"Show me top 10 products in the last 30 days"

✅ Causal Questions  
"Why did sales drop for brand X?"

✅ Comparative Analysis
"Which category has the best conversion rate?"

✅ Smart Follow-ups
User: "What products sold best?"
Agent: [Returns top products]
User: "Why?" ← Automatically knows you mean "Why did these products sell best?"

✅ Visualization Requests
"Show me a chart of monthly sales trends"

✅ Recommendations
"What should we do to improve customer retention?"
```

### 🔄 Follow-Up Context Preservation

The system automatically remembers:

```
📊 Product Info
├─ Product ID and name
├─ Category
└─ Brand

📈 Metrics
├─ Units sold
├─ Revenue
├─ Conversion rate
└─ Satisfaction score

⏰ Time Periods
├─ Last 7 days
├─ Last 30 days
├─ Custom date range
└─ Month-over-month

💾 Previous Results
├─ Ranking positions
├─ Numerical values
└─ Trend direction
```

### 📊 Dashboard Pages

#### 1. **Main Dashboard** 📊
- 30-day trend visualization
- Top 10 brands by sales
- Top 10 products by purchases
- Customer segment breakdown

#### 2. **Brand Intelligence** 🏢
- Brand performance metrics
- 30-day sales trends
- Conversion rates
- Customer sentiment scores
- Revenue tracking

#### 3. **Category Analysis** 🏷️
- 6 main categories analyzed
- Sales tier classification (weak/medium/strong)
- Conversion performance
- Customer satisfaction
- Comparative benchmarks

#### 4. **Customer Segments** 👥
```
VIP Champions           → 5%   (High value, loyal)
Returning Customers     → 25%  (Regular buyers)
One-Time Buyers        → 35%  (Occasional purchases)
Low Engagement         → 20%  (Minimal activity)
Window Shoppers        → 15%  (Browse only)

⬇️ Export customer lists to Excel for targeted campaigns
```

### 🔌 API Endpoints

#### Chat Operations
```http
POST /api/chat
Content-Type: application/json

{
  "message": "سوال کاربر / User question",
  "session_id": "unique-session-id"
}

Response (200 OK):
{
  "answer": "پاسخ دستیار / Assistant response",
  "chart": { /* optional chart configuration */ }
}
```

```http
DELETE /api/chat/{session_id}

Response (200 OK):
{
  "deleted": true
}
```

#### Data Endpoints
```http
GET /api/dashboard
→ Returns: trend data, top brands, top products, segments

GET /api/brand-intelligence
→ Returns: brand metrics, sentiment, conversion rates

GET /api/category-intelligence
→ Returns: category performance, tier classifications

GET /api/customer-segments
→ Returns: segment distribution & counts

GET /api/customer-segments/{cluster_name}/export
→ Returns: Excel file with customer IDs for segment

GET /api/health
→ Returns: {"status": "ok"}
```

### 🗄️ Database Schema

#### Chat Memory Tables
```sql
-- Conversation history storage
CREATE TABLE public.chat_memory (
  chat_id TEXT PRIMARY KEY,
  messages JSONB NOT NULL,
  updated_at TIMESTAMPTZ DEFAULT now()
);

-- Evaluation metrics & calibration
CREATE TABLE public.eval_log (
  id BIGSERIAL PRIMARY KEY,
  chat_id TEXT,
  question TEXT,
  faithfulness_score INTEGER,    -- 0-100
  relevance_score INTEGER,       -- 0-100
  confidence_score INTEGER,      -- 0-100
  grounded BOOLEAN,              -- Backed by data?
  turn_index INTEGER,            -- Which turn?
  created_at TIMESTAMPTZ DEFAULT now()
);
```

#### Analytics Database (Read-Only)
```sql
-- Core tables you query through the platform
user_behavior_logs  → Customer actions (views, purchases, cart)
products            → Product catalog with prices & metadata
categories          → Product categorization
brands              → Brand information
comments            → Customer reviews & ratings
kpi.ml_user_clusters → Behavioral segments (ML-generated)
```

---

## 🎯 Example Conversations

### 📊 Example 1: Simple Query + Follow-up

```
🧑 User: "کدام برند بیشترین درآمد داشت؟"
          (Which brand had the highest revenue?)

🤖 Agent: "برند X در 30 روز اخیر بیشترین درآمد را 
          با ۲,۵۰۰,۰۰۰ تومان داشت"
          (Brand X had the highest revenue with 2.5M in last 30 days)

🧑 User: "چرا؟"
         (Why?)

🤖 Agent: [Automatically knows you mean "Why did Brand X earn the most?"]
         [Runs RAG on Brand X reviews to find satisfaction signals]
         "برند X به دلیل کیفیت بالا و سرویس خوب مشتری..."
         (Brand X succeeded due to high quality and good customer service...)
```

### 🎨 Example 2: Visualization Request

```
🧑 User: "نمودار فروش محصولات دسته بندی آرایشی را نشون بده"
         (Show me a chart of cosmetic product sales)

🤖 Agent: [Runs SQL to get cosmetics category data]
         [Generates chart configuration]
         [Frontend renders interactive chart]
         
         📈 Provides trend analysis:
         "فروش دسته آرایشی در هفته اخیر ۱۵% رشد داشت"
         (Cosmetics sales grew 15% last week)
```

### 💡 Example 3: Strategic Question

```
🧑 User: "برای بازگرداندن مشتریان Window Shopper چه کار کنیم؟"
         (How do we bring back Window Shopper customers?)

🤖 Agent: [Calls tool_knowledge_base for best practices]
         [Combines with actual data from segment analysis]
         
         ✅ Recommendation:
         "با توجه به رفتار این گروه و تجربیات موفق:
          1. تخفیف 20% روی خرید اول
          2. ایمیل مارکتینگ هفتگی
          3. توصیه‌های شخصی‌شده
          ..."
```

---

## 🔒 Security Architecture

### Database Security
```
┌─────────────────────────────────────────────┐
│      PostgreSQL Database Server             │
│                                             │
│  🔐 Analytics Role (Read-Only)              │
│     └─ SELECT only on data tables           │
│     └─ Cannot modify data                   │
│                                             │
│  🔓 Chat Role (Write-Capable)               │
│     └─ SELECT/INSERT on chat_memory         │
│     └─ SELECT/INSERT on eval_log            │
│     └─ No access to analytics data          │
│                                             │
└─────────────────────────────────────────────┘
```

### Application Security
- ✅ **Input Validation**: Strict validation on all user inputs
- ✅ **CORS Protection**: Configurable origin whitelist
- ✅ **Environment Variables**: Secrets never in code
- ✅ **Connection Pooling**: Efficient resource management
- ✅ **SQL Injection Prevention**: Parameterized queries throughout

---

## ⚡ Performance Optimization

### Caching Strategy
```
Dashboard Cache (5-min TTL)
├─ Pre-warmed on startup
├─ Background refresh loop
├─ Never blocks requests
└─ Thread-safe with locks

Query Optimization
├─ SQL tie-breakers for deterministic results
├─ Minimum sample filters (view_cnt >= 30)
├─ Pre-aggregation before JOINs
└─ Strategic indexing on hot columns

Memory Management
├─ Automatic conversation compaction
├─ LLM-powered summarization
├─ JSONB efficient storage
└─ Configurable max raw turns (default: 3)
```

---

## 📂 Project Structure

```
ai-customer-behavior-analytics-platform/
│
├── 🐍 Backend (Python)
│   ├── main.py                    # Agent orchestration & follow-up logic
│   ├── api.py                     # FastAPI endpoints & caching layer
│   ├── memory_store.py            # PostgreSQL-backed conversation memory
│   │
│   └── Graph/                     # LangGraph agent implementation
│       ├── graph.py               # State machine definition
│       ├── nodes.py               # Decision nodes
│       ├── sql_agent.py           # 📊 SQL tool
│       ├── rag_agent.py           # 🔍 RAG tool
│       ├── chart_agent.py         # 📈 Chart tool
│       ├── llm_client.py          # OpenAI API wrapper
│       └── db.py                  # Database connections
│
├── 🎨 Frontend (React)
│   └── rahin_front_ai/
│       ├── public/assets/
│       │   └── logo-mark.png      # Rahin Logo
│       ├── src/
│       │   ├── components/
│       │   ├── pages/
│       │   └── main.jsx
│       └── vite.config.js
│
├── ⚙️ Configuration
│   ├── .env.example
│   ├── requirements.txt
│   └── package.json
│
└── 📚 Documentation
    └── README.md
```

---

## 🐛 Troubleshooting

### API Not Responding
```bash
# Check if server is running
curl http://localhost:8000/api/health

# Verify PostgreSQL connection
psql -U analytics_reader -d analytics_db -c "SELECT 1;"
```

### Chat History Lost
```bash
# Verify chat_memory table
psql -U app_chat_writer -d postgres -c \
  "SELECT COUNT(*) FROM public.chat_memory;"
```

### Slow Queries
```sql
-- Monitor slow queries
SELECT mean, calls, query FROM pg_stat_statements 
WHERE mean > 1000 ORDER BY mean DESC;
```

---

## 📝 Contributing

We welcome contributions! 🎉

```bash
# 1. Fork the repository
# 2. Create feature branch
git checkout -b feature/amazing-feature

# 3. Make changes & test
# 4. Commit with clear message
git commit -m "✨ Add amazing feature"

# 5. Push & open Pull Request
git push origin feature/amazing-feature
```

---

## 📄 License

This project is licensed under the **MIT License**.

---

<div align="center">

## ⭐ If you find this helpful, please give it a star!

[![GitHub stars](https://img.shields.io/github/stars/Rainsh724/ai-customer-behavior-analytics-platform?style=social)](https://github.com/Rainsh724/ai-customer-behavior-analytics-platform)

**Made with 💚 by Rahin Analytics Team**

[🔝 Back to Top](#-ai-customer-behavior-analytics-platform)

![Last Updated](https://img.shields.io/badge/Last%20Updated-2025-green?style=flat-square)
![Maintained](https://img.shields.io/badge/Maintained%3F-Yes-green?style=flat-square)

</div>
