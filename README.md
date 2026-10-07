# <div align="center"><img src="rahin_front_ai/public/assets/logo-mark.png" width="80" alt="Rahin Logo"></div>

# 🎯 AI Customer Behavior Analytics Platform

<div align="center">

**Transform Customer Data Into Actionable Intelligence**

[![Python](https://img.shields.io/badge/Python-90.2%25-3776ab?logo=python&logoColor=white&style=for-the-badge)](https://python.org)
[![JavaScript](https://img.shields.io/badge/JavaScript-2.3%25-f7df1e?logo=javascript&logoColor=black&style=for-the-badge)](https://javascript.com)
[![CSS](https://img.shields.io/badge/CSS-5.6%25-1572b6?logo=css3&logoColor=white&style=for-the-badge)](https://www.w3.org/Style/CSS/)
[![HTML](https://img.shields.io/badge/HTML-1.9%25-e34c26?logo=html5&logoColor=white&style=for-the-badge)](https://html.spec.whatwg.org/)

<br/>

A production-oriented, ReAct-based analytics platform that combines an LLM agent, validated Text-to-SQL, semantic search over Persian customer reviews (pgvector), validated analytics, and real-time dashboards to turn customer behavior data into evidence-backed insights.

[![GitHub Stars](https://img.shields.io/github/stars/Rainsh724/ai-customer-behavior-analytics-platform?style=social)](https://github.com/Rainsh724/ai-customer-behavior-analytics-platform)
[![GitHub Forks](https://img.shields.io/github/forks/Rainsh724/ai-customer-behavior-analytics-platform?style=social)](https://github.com/Rainsh724/ai-customer-behavior-analytics-platform)

<br/>

<img src="rahin_front_ai/public/assets/logo-mark.png" width="120" alt="Rahin Logo">

### 🧠 From Customer Behavior → Evidence → Decision

<!-- Replace this file with your real product/demo GIF when available. -->

[📚 Features](#-key-features) • [🏗️ Architecture](#-architecture) • [🚀 Quick Start](#-quick-start) • [🛡️ Reliability](#️-reliability--guardrails) • [💻 Tech Stack](#-tech-stack) • [📖 Documentation](#-documentation)

</div>

---

## ✨ Key Features
## 🚀 What's New

This README reflects the current architecture and recent platform evolution:

- 🧩 **Three production tools**: `tool_sql`, `tool_rag`, and `tool_chart`
- 🧠 **Multi-question orchestration** with independent sub-question execution and validation
- 🔁 **Follow-up understanding** with Active Context preservation
- 🛡️ **Answer audit + retry + correction** before returning low-confidence results
- 🔐 **Defense-in-depth SQL controls**: AST validation, schema introspection, read-only execution, timeout, and row limits
- 💾 **Persistent conversation memory** with structured compaction and evaluation logging
- 📉 **Token/context optimization** through result compaction and non-LLM RAG summarization
- 📊 **Dashboard and business-intelligence views** for trends, brands, categories, and customer segments
- ⚡ **Caching and startup preloading** for frequently requested analytical data and local embeddings

### 🤖 **ReAct Conversational Agent**
```
💬 Natural Language Understanding
├─ Persian & English query support
├─ Tool-calling agent (Reason → Act → Observe loop)
├─ Context-aware follow-up detection ("Why?" keeps product / metric / period)
├─ Multi-question support (each sub-answer validated separately)
└─ Persistent multi-turn memory
```

- 🧠 **Agent-driven routing**: the LLM selects the appropriate analytics capability from the available tools
- 🔄 **Smart follow-ups**: an *Active Context* (product, `product_id`, metric, period, previous result) is injected only for follow-up questions
- 🎯 **Evidence first**: business metrics come from SQL, customer voice comes from RAG, and visualizations are generated from validated analytical results
- 🕒 **Deterministic time semantics**: all relative periods ("last 30 days") are anchored to a configurable *dataset reference date* instead of `NOW()`

### 🧰 **Agent Toolkit**
```
tools.py  (Tool Calling gateway)
├─ tool_sql   → validated, read-only PostgreSQL analytics
├─ tool_rag   → semantic search over customer reviews (pgvector)
└─ tool_chart → charts built on top of validated SQL results
```

| Tool | Answers the question | Source of truth |
|------|----------------------|-----------------|
| **tool_sql** | *What happened?* | PostgreSQL analytics DB |
| **tool_rag** | *What do customers say?* | `comments` + `comments_embedding` (pgvector) |
| **tool_chart** | *Show me visually* | Result of a validated SQL query |

The agent can call one or several tools in a single reasoning cycle. Raw tool output is compacted before it re-enters the LLM context, while the underlying evidence remains available for audit.

### 🛡️ **Answer Audit & Self-Correction**
- ✅ Every final answer is scored for **grounded / faithfulness / relevance / confidence** against the real tool trace
- 🔁 Score below threshold (`CORRECTION_THRESHOLD = 70`) → *retry* (re-reason with tools) or *correction* (rewrite the text only)
- 🔍 A corrected answer is **validated again** before being returned; correction is capped at one attempt
- 📊 Every evaluation is logged for later calibration

### 📊 **Multi-Modal Analytics**
```
Data Analysis Toolkit
├─ SQL Analytics → rankings, aggregations, time-series (Production SQL Validator)
├─ RAG Search → semantic customer review analysis (multilingual-e5-base, 768-d)
├─ Chart Generation → interactive visualizations on validated data
```

| Feature | Capability |
|---------|-----------|
| **SQL Analytics** | Complex aggregations, rankings with deterministic tie-breakers, time-series |
| **Semantic Search** | Persian review retrieval, Top-K = 20, compact keyword + representative-comment summary |
| **Time-Series** | Rolling windows anchored to the dataset reference date (`2023-03-01`) |
| **Comparisons** | Period-over-period and month-over-month trends |

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
- 🗄️ **PostgreSQL backend**: `chat_memory` (JSONB) in a separate database/role from the read-only analytics DB
- 🤖 **LLM-powered compaction**: only the last `CHAT_MEMORY_MAX_RAW_TURNS` turns stay raw; older turns become a structured summary that preserves `product_id`, metric and period
- ⚛️ **Atomic saves**: messages are appended/merged, so concurrent requests cannot overwrite each other
- 📊 **Evaluation logging**: confidence, faithfulness and relevance per turn (`eval_log`)
- 🧯 **Failure isolation**: a failed summarization or evaluation log never breaks the user's answer

---

## 🏗️ Architecture

### System Overview
```
┌──────────────────────────────────────────────────────────┐
│           🖥️  Frontend (React + Vite)                   │
│          rahin_front_ai/                                 │
│   • Dashboards   • Chat Interface   • Charts             │
└────────────────────┬─────────────────────────────────────┘
                     │ HTTP
                     ▼
┌──────────────────────────────────────────────────────────┐
│        🚀 FastAPI Backend (api.py)                      │
│   /api/chat  /api/dashboard  /api/brand-intelligence     │
│   /api/category-intelligence  /api/customer-segments     │
└────────────────────┬─────────────────────────────────────┘
                     ▼
┌──────────────────────────────────────────────────────────┐
│   main.py  →  Startup + per-request lifecycle            │
│   Load memory → Active Context → Graph → Save → Evaluate │
└───────┬──────────────────────────────────┬───────────────┘
        │                                  │
        ▼                                  ▼
┌──────────────────┐            ┌────────────────────────────┐
│ 💾 memory_store  │            │ ⚙️  LangGraph (Graph/)     │
│ chat_memory      │            │                            │
│ eval_log         │            │  agent ⇄ tools (ReAct)     │
│ (write role)     │            │     ↓                      │
└──────────────────┘            │  finalize → validate       │
                                │     ↓ (retry / correct)    │
                                │    END                     │
                                └─────────────┬──────────────┘
                                              │ tools.py
              ┌───────────────┬───────────────┼────────────────┐
              ▼               ▼               ▼                ▼
        ┌──────────┐   ┌────────────┐  ┌───────────┐   ┌────────────┐
        │📊 SQL    │   │🔍 Vector   │  │📈 Chart   │   │📚 Knowledge│
        │sql_agent │   │retriever   │  │chart_agent│   │Base agent  │
        └────┬─────┘   └─────┬──────┘  └─────┬─────┘   └────────────┘
             ▼               │               │
   ┌──────────────────┐      │               │
   │ Production SQL   │      │               │
   │ Validator (AST)  │      │               │
   └────────┬─────────┘      │               │
            ▼                ▼               │
      ┌─────────────────────────────┐        │
      │ db.py (read-only pool)      │◄───────┘
      │ PostgreSQL + pgvector       │
      └─────────────────────────────┘
```

### Request Lifecycle
```
Question
  ↓
main.run(): load memory → build messages (system prompt once) → extract Active Context
  ↓                                  → add question → compact history if needed
LangGraph ReAct loop  (max 6 iterations, max 3 consecutive tool errors)
  agent ──tool calls──► tools_node ──compact result──► agent ...
  ↓ (no more tool calls)
finalize → validate (audit)
  ├─ OK ─────────────────────────────► answer
  ├─ low score → retry (re-reason) ──► loop
  └─ low score → correct (rewrite) ──► validate again ──► answer
  ↓
save_messages → log_evaluation (best-effort)
```

### Component Breakdown

| Layer | Component | Technology |
|-------|-----------|-----------|
| **Presentation** | React Frontend | React 18, Vite, JavaScript |
| **API** | FastAPI Server | FastAPI, Python 3.10+ |
| **Agent** | LangGraph ReAct Engine | LangGraph, OpenAI-compatible chat/tool-calling API |
| **LLM** | Configurable provider | `AGENT_LLM_MODEL` (default `openai/gpt-oss-120b`) |
| **Embeddings** | Local model | `intfloat/multilingual-e5-base` (sentence-transformers, 768-d) |
| **SQL Safety** | Production SQL Validator | sqlglot AST analysis, dynamic schema introspection |
| **Memory** | Chat storage | PostgreSQL, JSONB, separate write-capable role |
| **Data** | Analytics DB | PostgreSQL + pgvector (read-only pool) |

---

## 🛡️ Reliability & Guardrails

### SQL Safety (defense in depth)
```
LLM-generated SQL
  ↓
1. ProductionSQLValidator (sqlglot AST)
   ├─ single SELECT / WITH…SELECT only, no multi-statement
   ├─ allowed tables + real schema (introspected from PostgreSQL)
   ├─ allowed join pairs & keys, no comma joins
   ├─ fan-out protection (parent/child joins, multi-child joins)
   ├─ no direct SUM/AVG on ratio columns (weighted average hint)
   ├─ division-by-zero safety (NULLIF)
   └─ deterministic Top-N (stable tie-breaker such as product_id)
  ↓
2. Read-only connection (default_transaction_read_only)
3. statement_timeout  (STATEMENT_TIMEOUT_MS, default 60000 ms)
4. HARD_MAX_ROWS = 200  → result marked truncated + total row count reported
```
Validator errors explain both the problem and the fix, so the agent can self-correct in the next ReAct round.

### Agent Loop Protection
| Guard | Value | Purpose |
|-------|-------|---------|
| `MAX_ITERATIONS` | 6 | Stops unbounded reasoning / token burn |
| `MAX_CONSECUTIVE_TOOL_ERRORS` | 3 | Circuit breaker when every tool call keeps failing |
| `MAX_CORRECTION_RETRIES` | 1 | One rewrite, then re-validation |
| `CORRECTION_THRESHOLD` | 70 | Minimum audit score to accept an answer |

### Context & Cost Control
- 📉 Raw tool results are **compacted** before entering the LLM context (raw evidence is stored separately)
- 🧾 The DB schema lives only in `tool_sql`'s definition (not repeated per tool)
- 💬 RAG output is compressed without an LLM (keywords + representative comments with `comment_id`)
- 🗜️ Old conversation turns are summarized, never blindly resent
- ⏱️ 429 rate limits: exponential backoff (2s, 4s, 8s, 16s) via `LLM_RATE_LIMIT_MAX_RETRIES` / `LLM_RATE_LIMIT_BASE_DELAY`

### Time Semantics
The dataset is a historical snapshot, so the reference date is configuration, not `NOW()`:
`DATASET_REFERENCE_DATE` (default **2023-03-01**). Ranges use the half-open convention `[start, end)`; the reference day is included by using `< reference + INTERVAL '1 day'`.

### Correlation vs. Causation
The system prompt requires evidence-based explanations: SQL numbers and retrieved customer comments are reported as *observed signals*, and the answer must not invent causes that the evidence does not support. The audit layer flags answers that are not grounded in the tool trace.

---

## 🚀 Quick Start

### 📋 Prerequisites

```bash
✅ Python 3.10 or higher
✅ Node.js 18 or higher
✅ PostgreSQL 13 or higher with the pgvector extension
✅ An API key for an OpenAI-compatible LLM provider (tool calling required)
```

### 1️⃣ Clone & Setup Environment

```bash
git clone https://github.com/Rainsh724/ai-customer-behavior-analytics-platform.git
cd ai-customer-behavior-analytics-platform
cp .env.example .env
```

### 2️⃣ Configure `.env` File

```env
# 🗄️ Analytics Database (Read-Only role)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=analytics_db
DB_USER=analytics_reader
DB_PASSWORD=your_secure_password
STATEMENT_TIMEOUT_MS=60000

# 💬 Chat Memory Database (Write-capable role)
CHAT_DB_HOST=localhost
CHAT_DB_PORT=5432
CHAT_DB_NAME=postgres
CHAT_DB_USER=app_chat_writer
CHAT_DB_PASSWORD=your_secure_password
CHAT_DB_CONNECT_TIMEOUT=10

# 🤖 LLM (OpenAI-compatible provider: API key / base URL variable names as in .env.example)
AGENT_LLM_MODEL=openai/gpt-oss-120b
LLM_RATE_LIMIT_MAX_RETRIES=4
LLM_RATE_LIMIT_BASE_DELAY=2

# 🔎 Local embedding model (intfloat/multilingual-e5-base, 768-d)
# After the first successful download you can run fully offline:
# HF_HUB_OFFLINE=1
# HF_TOKEN=your_hf_token   # optional, avoids Hub rate limits on first download

# 🕒 Dataset time
DATASET_REFERENCE_DATE=2023-03-01

# ⚙️ Cache & Memory
DASHBOARD_CACHE_TTL_SECONDS=300
CHAT_MEMORY_MAX_RAW_TURNS=3
```

### 3️⃣ Prepare the Databases

- The analytics database must already contain the data tables and the `comments_embedding` table (768-d vectors created with the same e5 model used at query time).
- At runtime the application **does not create tables**; it only verifies that `public.chat_memory` exists. Create it once with an admin role (SQL in [Database Schema](#️-database-schema)).

### 4️⃣ Install Backend Dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 5️⃣ Start Backend Server

```bash
uvicorn api:app --reload --port 8000
# Startup preloads the embedding model, checks chat_memory, ensures the eval schema
# and refreshes the SQL validator from the real database schema.
# Server runs at: http://localhost:8000
```

### 6️⃣ Install & Start Frontend

```bash
cd rahin_front_ai
npm install
npm run dev
# Frontend runs at: http://localhost:5173
```

### 7️⃣ Verify Everything Works

```bash
curl http://localhost:8000/api/health

curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "کدام محصولات بیشترین فروش را دارند؟", "session_id": "test-session-001"}'

curl http://localhost:8000/api/dashboard
```

---

## 💻 Tech Stack

### 🐍 Backend
```
FastAPI          → API layer
LangGraph        → ReAct agent orchestration & state management
OpenAI-compatible client → chat completions + tool calling (provider is swappable via config)
sentence-transformers    → local multilingual-e5-base embeddings (768-d)
sqlglot          → AST-based SQL validation
PostgreSQL       → analytics data, chat memory, evaluation log
pgvector         → cosine similarity search over review embeddings
Psycopg2         → PostgreSQL adapter (pooled connections)
Pandas           → Data manipulation & export
```

### 🎨 Frontend
```
React 18 · Vite · JavaScript · CSS3 · Chart.js · Axios
```

### 📊 Data & Analytics
```
PostgreSQL       → Relational analytics + kpi schema
pgvector         → Semantic search on Persian comments
Window Functions → Time-series analysis
```

---

## 🖼️ Product Showcase

> Keep screenshots and GIFs in `docs/assets/` so the README remains easy to maintain.

| View | What it shows |
|---|---|
| 🤖 **AI Assistant** | Conversational analytics, follow-ups, and multi-question reasoning |
| 📊 **Dashboard** | Trends, rankings, KPIs, and customer segments |
| 🏢 **Brand Intelligence** | Brand-level performance and customer feedback signals |
| 🏷️ **Category Analysis** | Category comparison and conversion/satisfaction analytics |
| 👥 **Customer Segments** | Behavioral clusters and exportable customer lists |

## 📚 Documentation

### 💬 How to Ask Questions

```
✅ Metrics Queries
"Show me top 10 products in the last 30 days"

✅ Explanatory Questions (evidence-based, not causal claims)
"Why did sales drop for brand X?"  → SQL trend + customer comments

✅ Smart Follow-ups
User: "What products sold best?"
User: "Why?"  ← keeps the same products, metric and period

✅ Visualization Requests (only when a chart is explicitly requested)
"Show me a chart of monthly sales trends"

"What should we do to improve customer retention?"
```

### 🔄 Follow-Up Context Preservation
The Active Context extracted from the last *successful* turn contains: product name / `product_id`, metric, time period and the previous result (ranking positions, values, trend). It is injected only when the new question is detected as a follow-up, which keeps prompts small.

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
-- Conversation history storage (create once with an admin role)
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
-- Core tables queried through tool_sql / tool_rag
products, brands, categories, sellers, cities   → catalog & dimensions
users, sessions, user_behavior_logs              → customer actions (views, purchases, cart)
comments, comment_aspects                        → customer reviews & extracted aspects
comments_embedding                               → 768-d e5 vectors for pgvector search
analytics / kpi schemas (e.g. kpi.ml_user_clusters) → aggregates & ML behavioral segments
```
The validator loads the real schema and foreign keys from PostgreSQL at startup (`schema_introspector.py`), with architecture-specific join rules kept alongside them.

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

### 💡 Example 3: Multi-Question Analytics

```text
🧑 User:
"کدام برند بیشترین فروش را دارد و مشتریان درباره محصولاتش بیشتر از چه موضوعی صحبت می‌کنند؟"

🤖 Agent:
[Splits the request into analytical sub-questions]
[Runs SQL for sales ranking]
[Runs RAG for customer-review evidence]
[Validates each sub-answer]
[Combines the results]

✅ Final answer:
A structured comparison of sales performance and customer feedback,
with the underlying evidence kept in the tool trace.
```

---

## 🔒 Security Architecture

### Database Security
```
┌─────────────────────────────────────────────┐
│      PostgreSQL Database Server             │
│                                             │
│  🔐 Analytics Role (Read-Only)              │
│     └─ SELECT only, read-only transactions  │
│     └─ statement_timeout enforced           │
│     └─ used by SQL tool & vector search     │
│                                             │
│  🔓 Chat Role (Write-Capable, separate DB)  │
│     └─ SELECT/INSERT/UPDATE on chat_memory  │
│     └─ INSERT on eval_log                   │
│     └─ No access to analytics data          │
│                                             │
└─────────────────────────────────────────────┘
```
The application follows least privilege: it never issues `CREATE TABLE` at runtime.

### Application Security
- ✅ **AST-based SQL validation** before any LLM-generated query is executed
- ✅ **Read-only execution + timeout + row cap** independent of the prompt
- ✅ **Parameterized queries** for internal filters (e.g. `product_id` in vector search)
- ✅ **CORS Protection**: Configurable origin whitelist
- ✅ **Environment Variables**: Secrets never in code
- ✅ **Import without side effects**: no DB connection is opened at import time; pools are lazy

---

## ⚡ Performance Optimization

```
Dashboard Cache (5-min TTL)
├─ Pre-warmed on startup
├─ Background refresh loop
├─ Never blocks requests
└─ Thread-safe with locks

Query Optimization
├─ Deterministic tie-breakers for Top-N
├─ Minimum sample filters (view_cnt >= 30)
├─ Pre-aggregation of child tables before JOINs
└─ Strategic indexing on hot columns

LLM / Token Optimization
├─ Compact tool results, raw evidence kept out of context
├─ Schema defined once (tool_sql only)
├─ Non-LLM RAG summarization
├─ Conversation compaction (last N raw turns + summary)
└─ Exponential backoff on 429 rate limits

Startup vs. Hot Path
├─ Embedding model preloaded at startup
└─ SQL validator refreshed from the real schema at startup
```

---

## 📂 Project Structure

```
ai-customer-behavior-analytics-platform/
│
├── 🐍 Backend (Python)
│   ├── main.py                    # Startup, per-request lifecycle, system prompt, Active Context
│   ├── api.py                     # FastAPI endpoints & caching layer
│   ├── memory_store.py            # PostgreSQL-backed conversation memory + eval log
│   │
│   └── Graph/                     # LangGraph ReAct agent implementation
│       ├── state.py               # GraphState shared between nodes
│       ├── nodes.py               # agent / tools / finalize / validate / safe nodes
│       ├── graph.py               # Graph wiring & routing (loop guards, retry/correct)
│       ├── tools.py               # Tool definitions + dispatcher (tool calling gateway)
│       ├── sql_agent.py           # 📊 SQL execution boundary (validate → run)
│       ├── production_validator.py# 🛡️ AST-based SQL validator
│       ├── schema_introspector.py # Real schema / FK extraction from PostgreSQL
│       ├── vector_retriever.py    # 🔍 RAG: e5 embedding + pgvector Top-K
│       ├── text_summary.py        # Non-LLM compression of retrieved comments
│       ├── chart_agent.py         # 📈 Chart tool
│       ├── audit.py               # Answer validation & correction
│       ├── dataset_time.py        # Dataset reference date & time contract
│       ├── llm_client.py          # LLM calls, rate-limit retry, local embeddings
│       └── db.py                  # Read-only pool, timeout, vector search
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

### LLM Errors (429 / 413)
- **429 RateLimitError**: the client retries with exponential backoff; daily token limits (TPD) may require waiting or switching provider/model via `AGENT_LLM_MODEL`.
- **413 Request too large**: means the LLM context grew too much (very large tool results or history). Results are compacted and history is summarized by default; check `CHAT_MEMORY_MAX_RAW_TURNS`.

### Embedding Model Fails to Load
```bash
# After one successful download, run offline from the local cache
export HF_HUB_OFFLINE=1
```
Query embeddings must come from the same model as the stored `comments_embedding` vectors (multilingual-e5-base, 768-d, `query:` prefix).

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

## 🎨 Repository Branding

The Rahin logo is already referenced from the frontend asset path:

```html
<div align="center">
  <img src="rahin_front_ai/public/assets/logo-mark.png" width="120" alt="Rahin Logo">
</div>
```

For GitHub, there are three different branding surfaces:

1. **README logo** — the image above, stored in the repository.
2. **Repository social preview** — set a Rahin-branded banner in **GitHub → Settings → Social preview**.
3. **GitHub repository icon** — GitHub does not provide a normal per-repository custom icon setting. The closest equivalents are the README logo, social preview, and the profile/organization avatar.

For the best result, use a clean Rahin logo on a transparent background for the README and a wider 1280×640-style branded banner for the social preview.

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

![Last Updated](https://img.shields.io/badge/Last%20Updated-2026-green?style=flat-square)
![Maintained](https://img.shields.io/badge/Maintained%3F-Yes-green?style=flat-square)

</div>
