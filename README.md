<div align="center">

<img src="rahin_front_ai/public/assets/logo-mark.png" width="110" alt="Rahin Logo">

# 🎯 Rahin — AI Customer Behavior Analytics Platform

### Intelligent Customer & Business Analytics Assistant

**From Raw Data → Evidence → Intelligence → Management Action**

<br/>

[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white&style=for-the-badge)](https://www.python.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?logo=postgresql&logoColor=white&style=for-the-badge)](https://www.postgresql.org/)
[![pgvector](https://img.shields.io/badge/pgvector-Vector_Search-336791?style=for-the-badge)](https://github.com/pgvector/pgvector)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agent_Orchestration-1C1C1C?style=for-the-badge)](https://www.langchain.com/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white&style=for-the-badge)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-20232A?logo=react&logoColor=61DAFB&style=for-the-badge)](https://react.dev/)

<br/>

[![GitHub Stars](https://img.shields.io/github/stars/Rainsh724/ai-customer-behavior-analytics-platform?style=social)](https://github.com/Rainsh724/ai-customer-behavior-analytics-platform)
[![GitHub Forks](https://img.shields.io/github/forks/Rainsh724/ai-customer-behavior-analytics-platform?style=social)](https://github.com/Rainsh724/ai-customer-behavior-analytics-platform)

</div>

---

## 🧠 What is Rahin?

**Rahin** is an AI-powered customer and business analytics platform designed to reduce the gap between **raw customer data** and **managerial decision-making**.

Instead of requiring a user to manually select dashboards, write SQL queries, inspect customer reviews, and combine multiple analytical results, Rahin turns a natural-language business question into a controlled analytical workflow:

```text
Business Question
       ↓
   AI Agent
       ↓
 ┌───────────────┬────────────────────┐
 │ Structured    │ Customer Voice     │
 │ Analytics     │ Semantic Search    │
 │   tool_sql    │     tool_rag       │
 └───────┬───────┴──────────┬─────────┘
         ↓                   ↓
       Evidence + Context + Validation
                    ↓
              Insight / Answer
                    ↓
          Visualization when needed
```

The platform combines a substantial **data engineering pipeline**, a PostgreSQL-based analytical foundation, **feature engineering and KPI generation**, **ABSA for customer reviews**, vector retrieval with pgvector, behavioral customer segmentation, and a **LangGraph-based agentic layer**.

---

# 📌 From Data to Decision

Rahin is built as a complete analytical pipeline rather than only an LLM chatbot:

```text
Raw Datasets
     ↓
Metadata & Data Understanding
     ↓
Data Cleaning + Quality Validation
     ↓
Text Preprocessing
     ↓
Chunked Processing / Parquet
     ↓
ABSA + Sentiment + Aspect Extraction
     ↓
Embeddings + Vector Indexing
     ↓
PostgreSQL Data Foundation
     ↓
Feature Engineering
     ↓
KPI Generation
     ↓
RFM + Behavioral Clustering
     ↓
Analytics / RAG / Visualization
     ↓
LangGraph Agent
     ↓
Validation + Retry / Correction
     ↓
Evidence-Grounded Business Insight
```

---

# ✨ Core Capabilities

## 1. 🧹 Data Engineering & Data Quality

The project started from large, heterogeneous raw datasets and built a controlled data foundation before the AI layer was introduced.

### Dataset scale

| Dataset / Output | Records |
|---|---:|
| Products | **948,352** |
| Behavioral Logs | **3,750,416+** |
| Comments / Reviews | **6,153,060** |
| Comment Embeddings | **6,153,060** |
| Extracted Aspects | **4,032,541** |
| Sessions | **774,037** |
| Total tracked records / outputs | **21,821,169** |

### Structured Data Cleaning

The cleaning pipeline included:

- String standardization
- Duplicate removal based on primary keys
- Removal of records without required primary keys
- Numeric and currency cleaning
- Null-value handling / imputation
- Boolean standardization
- ID normalization to integer types
- Date normalization
- Persian/Jalali date conversion to Gregorian
- Data-type consistency checks

### Text Preprocessing

Customer review text was normalized through a dedicated preprocessing pipeline:

```text
Multiple Text Columns
        ↓
Text Consolidation
        ↓
Persian / Arabic Character Normalization
        ↓
HTML Removal
        ↓
URL / Email Removal
        ↓
Punctuation Handling
        ↓
Character-Repetition Reduction
        ↓
Whitespace Normalization
        ↓
Clean Review Text
```

### Data Quality Validation

A dedicated quality-control process was used to detect issues such as:

- Missing values
- Infinite numeric values
- Duplicate records
- Invalid behavioral events
- Inconsistent data types
- Integrity problems before downstream modeling

---

## 2. ⚙️ Scalable Data Processing

The dataset size made single-pass processing inefficient.

The pipeline therefore introduced **chunk-based processing**:

```text
Large Dataset
     ↓
50,000-row Chunks
     ↓
Clean
     ↓
Normalize
     ↓
Write Parquet
     ↓
Parallel / Multiprocessing Processing
```

Parquet was used as an intermediate representation because it provides:

- Columnar storage
- Better suitability for large analytical datasets
- Stronger schema/type preservation than CSV
- Efficient processing with DuckDB
- Reduced dependency on a single massive file

**DuckDB** was also used as part of the high-volume ETL architecture, including attachment to PostgreSQL for large-scale analytical processing.

---

# 🗄️ Data Foundation & Database Architecture

The database was **not copied directly from the raw dataset structure**.

Instead, the data model was redesigned around real entities, relationships, keys, and analytical requirements.

This resulted in a relational PostgreSQL foundation with core entities such as:

```text
users
cities
brands
categories
sellers
sessions
products
user_behavior_logs
comments
comment_aspects
comments_embedding
```

The architecture separates:

- **Core data**
- **Feature layers**
- **KPI / analytical views**
- **Vectorized customer-review data**

The database therefore became the central analytical foundation instead of maintaining disconnected feature and KPI files.

### Why this matters

The final architecture evolved from:

```text
Raw Files
   ↓
Feature Files
   ↓
KPI Files
   ↓
Database
```

to:

```text
Core Database Tables
        ↓
SQL Feature Engineering
        ↓
KPI Views / Analytics Layer
        ↓
AI + BI + Agentic Analytics
```

This reduced unnecessary duplication and made analytical features and KPIs directly reproducible from the underlying database.

---

# 🧩 Feature Engineering

Feature engineering was implemented at multiple analytical levels.

### Feature families

```text
Behavior Features
User Features
Product Features
City Features
Category Features
Brand Features
User × Product
User × Category
Sentiment Features
Aspect Features
Product Sentiment
Product × Aspect
Brand Sentiment
Category Sentiment
Global Aspect
Time Features
```

### Examples of behavioral features

- Views
- Carts
- Purchases
- Removes
- Conversion rates
- Engagement
- Session dynamics
- Weekend activity
- Night activity
- View-to-purchase behavior
- Cart / remove relationships

### User-level behavioral profile

The customer clustering layer uses behavioral variables including:

```text
recency_days
total_spend
purchase_frequency
category_diversity
avg_session_duration
night_activity_ratio
weekend_activity_ratio
view_to_purchase_ratio
```

These features form the bridge between raw events and higher-level customer intelligence.

---

# 📊 KPI & Analytics Layer

The project does not stop at raw features.

A dedicated **KPI Layer** transforms behavioral and transactional data into business-level indicators.

## Global Executive Funnel

Tracks the customer journey through:

```text
Views
  ↓
Add to Cart
  ↓
Purchase
  ↓
Conversion
  ↓
Remove / Drop-off
```

## Product 360

Combines multiple evidence types around products:

- Behavioral performance
- Purchases
- Revenue
- Ratings
- Customer sentiment
- Aspect-level signals
- Managerial classification tags

## User Segmentation

The KPI layer identifies behavioral groups such as:

- VIP Customer
- Returning Customer
- One-Time Buyer
- Window Shopper
- Low Engagement

## Brand Diagnostics

Brand-level analytics combine behavioral and customer-voice signals:

- Views
- Purchases
- Comments
- Average rating
- Brand sentiment score

## Aspect Diagnostics

Aspect-level analytics support root-cause analysis using:

- Total mentions
- Positive mentions
- Negative mentions
- Negative impact percentage

Aspects can also receive managerial classifications such as:

- **Critical Weakness**
- **Key Strength**
- **Neutral**

## RFM Segmentation

The system calculates:

```text
R = Recency
F = Frequency
M = Monetary
```

R, F, and M scores are created using **NTILE-based scoring from 1 to 5**, followed by RFM classification.

Customer groups include:

- `vip`
- `promising`
- `at_risk`
- `lost`
- `regular`

## ML User Clusters

Behavioral clustering adds a second, model-driven segmentation layer.

The project identifies **five behavioral clusters**, including patterns corresponding to:

- High-value / VIP customers
- Returning customers
- One-time buyers
- Window shoppers
- Low-engagement customers

The purpose is to transform scattered behavioral events into interpretable customer segments that can be used by analytics and decision-support components.

---

# 💬 ABSA — Aspect-Based Sentiment Analysis

One of the major analytical components of Rahin is **Aspect-Based Sentiment Analysis (ABSA)**.

Traditional sentiment analysis answers:

> Is this review positive or negative?

ABSA answers a more useful business question:

> **What does the customer feel about which aspect of the product?**

For example:

```text
Review
  ↓
Aspect Detection
  ↓
Aspect-level Sentiment
  ↓
Product / Brand / Category Diagnostics
```

## Two-Stage ABSA Pipeline

### Stage 1 — Aspect Extraction

The aspect extraction model uses:

```text
ParsBERT
   ↓
BiLSTM
   ↓
CRF
   ↓
BIO Sequence Labels
```

**Base encoder:** `HooshvareLab/bert-base-parsbert-uncased`

The BiLSTM captures contextual sequence dependencies, while CRF provides structured decoding for BIO labels such as:

```text
B-ASP
I-ASP
O
```

This is particularly important because aspect extraction is a **sequence-labeling problem**, where neighboring token labels are not independent.

### Training considerations

The aspect pipeline addresses:

- Class imbalance
- Multi-aspect reviews
- Review-level aggregation
- Weighted loss
- Focal-loss-based training stabilization

The documented training configuration includes **Weighted Focal Loss with γ = 1.5**.

### Stage 2 — Aspect-level Sentiment Classification

After aspects are extracted, the sentiment stage determines the polarity associated with each aspect.

Production inference produces structured output similar to:

```json
{
  "term": "battery",
  "sentiment": "negative",
  "negative_pct": 0.82,
  "neutral_pct": 0.10,
  "positive_pct": 0.08
}
```

This structured output can then feed product, brand, category, and aspect diagnostics.

---

# 🔎 Semantic Search & Customer Voice

Customer reviews are transformed into vector representations to enable semantic retrieval.

### Embedding pipeline

```text
Customer Review
      ↓
Text Preprocessing
      ↓
Embedding Model
      ↓
Normalized Vector
      ↓
pgvector
      ↓
HNSW Index
      ↓
Top-K Semantic Retrieval
```

The project uses multilingual embeddings with a **768-dimensional vector representation** and a consistent document/query prefix strategy:

```text
Documents → passage: <comment>
Queries   → query: <user query>
```

### Retrieval

The RAG layer uses vector similarity search with:

```text
TOP_K = 20
```

Retrieved results include information such as:

- `comment_id`
- Similarity / distance
- Relevant review text
- Representative customer comments

The retrieval output is also compressed before being passed to the LLM to control context size without requiring another LLM call.

---

# 🤖 Agentic Intelligence

After building the data and analytics foundation, Rahin adds a **LangGraph-based agentic layer**.

The agent is not responsible for inventing business numbers.

Instead:

```text
Agent
  ↓
Select analytical capability
  ↓
Retrieve real evidence
  ↓
Validate execution
  ↓
Synthesize grounded answer
```

## Current Agent Tools

### `tool_sql`
**Structured business analytics**

Uses validated SQL against PostgreSQL for:

- Aggregations
- Rankings
- Time-series analysis
- Comparisons
- Business KPIs
- Customer / product / brand analytics

### `tool_rag`
**Semantic search on customer reviews**

Uses pgvector-based retrieval to answer questions about:

- Customer opinions
- Product aspects
- Sentiment
- Review evidence
- Qualitative customer signals

### `tool_chart`
**SQL-driven visualization**

Generates chart configurations from validated analytical results when visualization is explicitly requested.

---

# 🔄 LangGraph Orchestration

The agentic layer uses a controlled graph rather than an unrestricted LLM loop.

```text
                    ┌──────────────┐
                    │    User      │
                    └──────┬───────┘
                           ↓
                  Detect Question Type
                           ↓
              ┌────────────┴────────────┐
              ↓                         ↓
        Single Question            Multi Question
              ↓                         ↓
            Agent                  Sub-question Loop
              ↓                         ↓
           Tools                  Sub-agent / Tools
              ↓                         ↓
           Finalize              Combine Sub-answers
              ↓                         ↓
           Validate ←───────────────┬───┘
              ↓                     │
       ┌──────┴───────┐             │
       ↓              ↓             │
      Accept         Retry / Correct
                       ↓
                    Validate
```

### Multi-question orchestration

Complex questions can be decomposed into sub-questions.

Each sub-question has its own execution and validation state before the final answers are combined.

### Follow-up understanding

The system also detects follow-up questions and extracts an **Active Context** containing relevant information such as:

- Product / product ID
- Metric
- Time period
- Previous analytical result

This allows a question such as:

```text
User: Which products sold best?

User: Why?
```

to retain the analytical context instead of treating `"Why?"` as an unrelated question.

---

# 🛡️ Answer Validation & Self-Correction

A major reliability layer evaluates the generated answer against the actual execution trace.

The system can assess:

- Groundedness
- Faithfulness
- Relevance
- Confidence

When an answer falls below the configured correction threshold:

```text
Low Score
   ↓
Retry / Re-reason with tools
   ↓
Validate
   ↓
If necessary
   ↓
Correct the answer text
   ↓
Validate Again
```

The graph also enforces bounded recovery using:

```text
MAX_ITERATIONS
MAX_CONSECUTIVE_TOOL_ERRORS
MAX_CORRECTION_RETRIES
CORRECTION_THRESHOLD
```

This prevents uncontrolled agent loops and limits unnecessary LLM/token usage.

---

# 🔐 SQL Safety & Production Guardrails

LLM-generated SQL is treated as **untrusted input**.

The project therefore uses defense-in-depth controls.

```text
LLM-generated SQL
       ↓
ProductionSQLValidator
       ↓
SQL AST Analysis
       ↓
Schema / Join Validation
       ↓
Read-only PostgreSQL
       ↓
Statement Timeout
       ↓
Hard Row Limit
       ↓
Result
```

## AST-based validation

`sqlglot` is used to parse SQL structurally rather than relying on simple regular expressions.

The validator can inspect:

- SELECT / WITH structure
- JOINs
- ORDER BY
- Aggregations
- CTEs
- Division
- LIMIT
- Multiple statements
- Allowed tables
- Join relationships
- Ratio-sensitive columns

### Database-level controls

The analytical database connection is protected by:

- Read-only execution
- Statement timeout
- Hard result-row limit
- Real schema introspection
- Controlled join rules

The validator also considers parent/child relationships to reduce accidental **fan-out joins** and inflated aggregations.

---

# ⏱️ Deterministic Time Semantics

The dataset is a historical snapshot rather than a continuously updated live database.

Therefore, relative periods are anchored to a configurable:

```text
DATASET_REFERENCE_DATE
```

instead of relying blindly on:

```sql
NOW()
```

The documented reference date is:

```text
2023-03-01
```

Time ranges follow a half-open convention:

```text
[start, end)
```

This prevents queries such as "last 30 days" from changing meaning merely because the system clock changes.

---

# 💾 Persistent Conversation Memory

Rahin maintains conversation context through PostgreSQL-backed storage.

The memory layer supports:

- Multi-turn conversations
- Structured history
- Active-context preservation
- Conversation compaction
- Evaluation logging
- Failure isolation

Older conversation turns can be summarized rather than repeatedly sending the entire history to the LLM.

This reduces:

- Prompt size
- Token consumption
- Context degradation
- Repeated information

---

# 📈 Executive Analytics & Dashboards

The platform exposes the analytical foundation through an interactive React dashboard.

### Main Dashboard

- Daily / recent trends
- Top products
- Top brands
- Customer segments
- Funnel analytics

### Brand Intelligence

- Brand performance
- Sales trends
- Conversion indicators
- Customer sentiment
- Revenue-related metrics

### Category Analysis

- Category performance
- Sales tiers
- Conversion performance
- Customer satisfaction
- Comparative analysis

### Customer Segmentation

Behavioral segments can be inspected and customer lists can be exported for targeted analysis and campaigns.

---

# 🏗️ End-to-End Architecture

```text
┌─────────────────────────────────────────────────────────┐
│                    Rahin Frontend                       │
│             React + Vite + Dashboards                  │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Backend                      │
│            API + Dashboard + Chat + Export              │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                 LangGraph Agent Layer                   │
│                                                         │
│     Agent → Tools → Evidence → Validate → Answer       │
└───────────────┬───────────────────────┬─────────────────┘
                │                       │
        ┌───────▼────────┐      ┌──────▼─────────┐
        │   tool_sql     │      │    tool_rag    │
        │ Structured     │      │ Customer Voice │
        │ Analytics      │      │ Semantic Search│
        └───────┬────────┘      └──────┬─────────┘
                │                       │
                └──────────┬────────────┘
                           ▼
┌─────────────────────────────────────────────────────────┐
│              PostgreSQL + pgvector                      │
│                                                         │
│ Core Tables → Feature Layer → KPI Layer → Vectors      │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                 Data Engineering                        │
│ Cleaning → Text Processing → ETL → ABSA → Features     │
└─────────────────────────────────────────────────────────┘
```

---

# 📊 Evaluation

The project includes quantitative evaluation across the data, NLP, retrieval, agent, and system layers.

## ABSA

| Metric | Result |
|---|---:|
| Sentiment Accuracy | **86%** |
| Sentiment Macro-F1 | **0.84** |
| Aspect Recall | **0.45** |
| Aspect F1 | **0.34** |
| Aspect Precision | **0.27** |

## RAG / Retrieval

| Metric | Result |
|---|---:|
| Retrieval Quality | **81.2%** |
| Precision@5 | **82.0%** |
| Recall@5 | **74.5%** |
| F1@5 | **78.0%** |

`@5` denotes evaluation over the **top five retrieved results**.

## Agent & Application

| Metric | Result |
|---|---:|
| Text-to-SQL Accuracy | **86.5%** |
| Tool Routing Accuracy | **88.0%** |
| Task Completion | **83.0%** |
| HTTP Request Success Rate | **94.5%** |
| Faithfulness | **74.7%** |
| Faithfulness Pass Rate (>70) | **79.8%** |
| Relevance | **80.4%** |
| Insight Quality | **82.0%** |
| Manager Satisfaction | **81.5%** |
| Report Coverage | **8 domains** |

## Performance

| Metric | Result |
|---|---:|
| Average API Response Time | **6.8 s** |
| API Response Range | **1.2–9.5 s** |
| Agent Decision Time | **1.25 s** |
| Complex PostgreSQL Query | **360 ms** |
| Complex Query Range | **234–480 ms** |
| Customer Clusters | **5** |

---

# 🧰 Technology Stack

## Data Engineering

- Python
- Pandas
- PyArrow
- DuckDB
- Parquet
- Multiprocessing / chunked processing

## NLP & ABSA

- PyTorch
- Transformers
- ParsBERT
- BiLSTM
- CRF
- Weighted Focal Loss
- Sentence-level / aspect-level sentiment analysis

## Data & Database

- PostgreSQL
- SQL
- pgvector
- HNSW
- PostgreSQL views
- SQL-based Feature Engineering
- KPI schemas / views

## AI & Agentic Layer

- LLM
- LangChain
- LangGraph
- ReAct-style tool orchestration
- Tool calling
- Text-to-SQL
- RAG / semantic retrieval

## Backend

- FastAPI
- Uvicorn
- PostgreSQL connection management
- API endpoints
- Caching
- Evaluation logging

## Frontend

- React
- Vite
- JavaScript
- Interactive dashboards
- Data visualization
- Chat interface

---

# 📂 Project Structure

```text
ai-customer-behavior-analytics-platform/
│
├── Graph/
│   ├── main.py
│   ├── graph.py
│   ├── state.py
│   ├── nodes.py
│   ├── tools.py
│   ├── db.py
│   ├── sql_agent.py
│   ├── production_validator.py
│   ├── schema_introspector.py
│   ├── vector_retriever.py
│   ├── chart_agent.py
│   ├── dataset_time.py
│   ├── memory_store.py
│   └── ...
│
├── rahin_front_ai/
│   ├── public/
│   │   └── assets/
│   │       └── logo-mark.png
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   └── ...
│   └── vite.config.js
│
├── requirements.txt
├── .env.example
└── README.md
```

The repository is organized around the same conceptual layers described above: data foundation, analytical/AI components, agent orchestration, API services, and frontend presentation.

---

# 🚀 Quick Start

## Prerequisites

- Python 3.10+
- Node.js 18+
- PostgreSQL
- pgvector extension
- An OpenAI-compatible LLM provider with tool-calling support

## Clone

```bash
git clone https://github.com/Rainsh724/ai-customer-behavior-analytics-platform.git
cd ai-customer-behavior-analytics-platform
```

## Backend

```bash
python -m venv venv
```

Windows:

```powershell
venv\Scripts\activate
```

Linux / macOS:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure environment variables using:

```text
.env.example
```

## Frontend

```bash
cd rahin_front_ai
npm install
npm run dev
```

---

# 🔌 Analytical Interaction Examples

### Structured Analytics

```text
User:
Which products had the highest sales?

Rahin:
→ Generates / validates SQL
→ Executes against PostgreSQL
→ Returns the ranking with evidence
```

### Customer Voice

```text
User:
What are customers complaining about regarding this product?

Rahin:
→ Generates semantic query
→ Retrieves relevant reviews through pgvector
→ Summarizes representative evidence
→ Returns aspect-level customer signals
```

### Combined Question

```text
User:
Which products sell well but receive negative feedback about quality?

Rahin:
→ SQL: identifies high-sales products
→ RAG: retrieves relevant customer reviews
→ ABSA / aspect evidence: identifies quality-related signals
→ Combines the evidence into one analytical answer
```

### Multi-question

```text
User:
Which brand has the highest sales, which product drives it,
and what do customers say about its quality?

Rahin:
→ Decomposes the request
→ Executes sub-questions
→ Validates each result
→ Combines the sub-answers
→ Produces one grounded response
```

### Visualization

```text
User:
Show me the monthly sales trend.

Rahin:
→ SQL analytics
→ tool_chart
→ Frontend renders the visualization
```

---

# 🔒 Security & Reliability

Rahin uses multiple layers of protection rather than trusting the LLM prompt alone.

### SQL Security

- AST-based SQL validation
- SELECT / WITH-SELECT restrictions
- Controlled tables
- Real schema introspection
- Join validation
- Read-only database execution
- Statement timeout
- Hard result-row limit
- Parameterized internal filters

### Agent Reliability

- Bounded graph iterations
- Consecutive tool-error budget
- Validation before final response
- Retry / correction path
- Re-validation after correction
- Evaluation logging

### Data Integrity

- Data cleaning before modeling
- Automated quality checks
- Duplicate detection
- Missing-value checks
- Behavioral integrity rules
- Structured database relationships
- Reproducible SQL-based feature/KPI generation

---

# ⚡ Performance & Context Optimization

The project includes several optimizations designed around the practical constraints of large datasets and LLM context windows.

### Data Processing

- Chunked processing
- Parquet intermediate storage
- DuckDB for analytical ETL
- PostgreSQL-centered feature/KPI processing

### Retrieval

- Vector normalization
- HNSW indexing
- Top-K retrieval
- Non-LLM result compression
- Representative comment selection

### Agent / LLM

- Tool-result compaction
- Conversation summarization
- Active-context injection only when relevant
- Bounded iterations
- Rate-limit retry with exponential backoff
- Schema reuse instead of repeatedly injecting large schema descriptions

### Database

- Strategic indexes
- Pre-aggregation before risky joins
- Deterministic Top-N ordering
- Ratio-aware aggregation rules
- Query timeout and row limits

---

# 🎯 Why Rahin?

Rahin is designed around a simple principle:

> **A useful business answer must be grounded in evidence.**

The system therefore separates the responsibilities of each layer:

```text
Data Engineering
      ↓
Makes data trustworthy

Database + Feature Engineering
      ↓
Makes behavior analyzable

KPI Layer
      ↓
Makes analytics meaningful for business

ABSA + RAG
      ↓
Makes customer voice searchable and interpretable

Agent
      ↓
Makes analysis conversational

Validation
      ↓
Checks whether the answer is grounded

Dashboard
      ↓
Makes insights visible and actionable
```

The result is not simply a chatbot over a database.

It is an integrated **customer intelligence and business analytics system** that connects:

**Customer Behavior + Customer Voice + Business KPIs + AI Reasoning + Visualization**

into a single workflow.

---

# 📚 Documentation Map

For deeper technical understanding, the project documentation covers:

- Dataset metadata extraction
- Data cleaning and quality validation
- Text preprocessing
- ABSA model training and inference
- Embedding generation
- HNSW vector indexing
- PostgreSQL schema design
- Feature Engineering
- KPI Generation
- RFM segmentation
- ML customer clustering
- RAG retrieval
- Text-to-SQL
- SQL validation
- LangGraph state and routing
- Multi-question orchestration
- Follow-up / Active Context
- Answer validation and correction
- FastAPI backend
- React dashboard
- Evaluation and performance testing

---

# 📄 License

This project is licensed under the **MIT License**.

---

<div align="center">

<img src="rahin_front_ai/public/assets/logo-mark.png" width="70" alt="Rahin">

### Rahin — Intelligent Customer & Business Analytics

**From Data → Evidence → Intelligence → Decision**

⭐ If you find the project useful, consider giving the repository a star.

[Back to Top](#-rahin--ai-customer-behavior-analytics-platform)

</div>
