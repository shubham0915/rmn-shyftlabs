

# 🛒 Privacy-Preserving Agentic Retail Media Network (RMN) Engine

> **Built for ShyftLabs AdTech**  
> Real-time on-site advertising with Differential Privacy (DuckDB), Vector Search (ChromaDB), **XGBoost ML Ranking**, and **Agentic LLM Copywriting** (Gemini Flash).

[![Live Demo](https://img.shields.io/badge/Demo-Railway-0B0D0E?logo=railway)](https://rmn-engine.up.railway.app)
[![Privacy](https://img.shields.io/badge/Privacy-ε≤0.9-065f46)](https://diffprivlib.readthedocs.io)
[![Latency](https://img.shields.io/badge/Latency-<150ms-0284c7)](http://localhost:8000/docs)
[![Python](https://img.shields.io/badge/Python-3.11-3776ab?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi)](https://fastapi.tiangolo.com)

---

## 🎯 Business Problem & ShyftLabs Alignment

Retailers (Myntra, Nykaa, etc.) want to show **personalised ads on their own website** without sending raw user data to advertisers. ShyftLabs solves this with a **Retail Media Network (RMN)** architecture — a Data Clean Room that keeps first-party data private while still enabling smart ad targeting.

| ShyftLabs Real World | This Project Architecture |
|---|---|
| First-party Data Clean Room | **DuckDB** Async Background Worker (Scalable Flush) |
| Mathematical privacy guarantee | **diffprivlib Laplace** mechanism (Persistent DuckDB tracking) |
| Contextual / Semantic match | **ChromaDB** Vector Engine (`all-MiniLM-L6-v2`) |
| Real-time CTR Prediction | **XGBoost** Learning-to-Rank (LTR) |
| Ad Copy Personalization | **Gemini 1.5 Flash** Agentic generation |

---

## 🗺️ Diagram 1 — Master System Architecture

**The complete end-to-end picture: every component and how they connect.**

```mermaid
flowchart TB
    subgraph USER["👤 User / Browser"]
        UI["🖥️ Streamlit UI\n(port 7860)"]
    end

    subgraph API["⚡ FastAPI Backend (port 8000)"]
        direction TB
        EP_AD["/get_ad\n(GET)"]
        EP_TRACK["/track_event\n(POST)"]
        EP_METRICS["/metrics\n(GET)"]
        WORKER["🔄 Async Background\nEvent Worker\n(every 5s)"]
    end

    subgraph AI["🤖 AI / ML Layer"]
        EMBED["🧠 SentenceTransformer\nall-MiniLM-L6-v2\n(CPU Encoder)"]
        CHROMA["📦 ChromaDB\nVector Store\n(HNSW Cosine)"]
        XGB["📈 XGBoost LTR\nML Ranker\n(p_click predictor)"]
        GEMINI["✨ Gemini 1.5 Flash\nAgentic Copywriter\n(parallel calls)"]
    end

    subgraph PRIVACY["🔐 Privacy Clean Room"]
        DUCK["🦆 DuckDB\n(2.7M Retailrocket events\n+ Aggregates + Ads)"]
        DP["📐 Diffprivlib\nLaplace Mechanism\n(ε = 0.1/query)"]
        BUDGET["💰 Daily ε Budget\nTracker\n(ε_max = 0.9)"]
    end

    subgraph QUEUE["📬 Event Queue"]
        MEM["In-Memory Queue\n(fallback)"]
        REDIS["Redis Stream\n(optional)"]
    end

    UI -->|"1. Browse page"| EP_TRACK
    UI -->|"2. Get ranked ad"| EP_AD
    UI -->|"3. View metrics"| EP_METRICS

    EP_TRACK --> MEM
    EP_TRACK --> REDIS
    MEM --> WORKER
    REDIS --> WORKER
    WORKER -->|"flush every 5s"| DUCK

    EP_AD --> DUCK
    DUCK --> DP
    DP --> BUDGET
    BUDGET -->|"noisy stats"| EMBED
    EMBED -->|"context vector"| CHROMA
    CHROMA -->|"Top-10 candidates"| XGB
    XGB -->|"p(click) scores"| GEMINI
    GEMINI -->|"ranked + copy"| EP_AD
    EP_AD -->|"Ad + metadata"| UI
```

---

## 🔄 Diagram 2 — Ad Request Flow (Step-by-Step)

**What happens in `< 150ms` when a user clicks "Browse Page & Get Ad".**

```mermaid
sequenceDiagram
    actor User
    participant ST as 🖥️ Streamlit UI
    participant API as ⚡ FastAPI /get_ad
    participant CR as 🦆 DuckDB Clean Room
    participant DP as 📐 Laplace DP
    participant EMB as 🧠 SentenceTransformer
    participant VDB as 📦 ChromaDB
    participant XGB as 📈 XGBoost LTR
    participant LLM as ✨ Gemini Flash

    User->>ST: Selects page context & clicks Browse
    ST->>API: GET /get_ad?user_hash=...&page_text=...

    Note over API,DP: Step 1 · Privacy-Safe Stats (~10ms)
    API->>CR: get_all_ads() + get_dp_category_stats()
    CR->>DP: Query raw aggregates (views, carts, purchases)
    DP-->>CR: Add Laplace noise · burn 0.3ε budget
    CR-->>API: 42 ads + noisy {dp_views, dp_carts, dp_purchases}

    Note over API,VDB: Step 2 · Semantic Retrieval (~10ms)
    API->>EMB: encode(page_text + noisy_stats)
    EMB-->>VDB: 384-dim context vector
    VDB-->>API: Top-10 most similar ads (cosine distance)

    Note over API,XGB: Step 3 · ML Reranking (<1ms)
    API->>XGB: features[similarity, ctr, price, budget_intent]
    XGB-->>API: p(click) score per candidate

    Note over API,LLM: Step 4 · Parallel LLM Copy (~80ms)
    API->>LLM: generate_copy(Top-3 ads) — parallel threads
    LLM-->>API: 3x personalised 1-liner ad copy

    API-->>ST: AdResponse{top_ads, latency_ms, epsilon_used, ctr_lift}
    ST-->>User: Renders ranked ad cards + metrics
```

---

## 🔐 Diagram 3 — Differential Privacy Pipeline

**How user data is protected at every step in the Clean Room.**

```mermaid
flowchart LR
    subgraph RAW["📥 Raw Data (Private)"]
        EV["2.7M Retailrocket\nevent rows\n(views, carts, buys)"]
        ITEMS["Item → Category\nLookup Table"]
    end

    subgraph AGG["🧮 Aggregation Layer"]
        JOIN["SQL JOIN\nevents ↔ item_categories"]
        STATS["Real Category Stats\nn_views: 45,231\nn_carts: 3,412\nn_purchases: 891"]
    end

    subgraph DP_LAYER["🔐 Differential Privacy (Laplace Mechanism)"]
        CHECK{"ε_used + 0.3\n≤ ε_max (0.9)?"}
        BURN["Burn 0.3ε\nfrom daily budget"]
        NOISE["Add Laplace Noise\nnoise ~ Lap(Δf/ε)\nΔf=1, ε=0.1"]
        BLOCK["🚫 Block Query\nBudget Exhausted"]
    end

    subgraph OUT["📤 Noisy Output (Safe to Use)"]
        NSTATS["Noisy Stats\ndp_views: ~45,238\ndp_carts: ~3,409\ndp_purchases: ~895"]
        PERSIST["DuckDB:\ndate | epsilon_used\n2026-03-28 | 0.60"]
    end

    EV --> JOIN
    ITEMS --> JOIN
    JOIN --> STATS
    STATS --> CHECK
    CHECK -->|"YES · budget OK"| BURN
    BURN --> NOISE
    NOISE --> NSTATS
    BURN --> PERSIST
    CHECK -->|"NO · exhausted"| BLOCK

    NSTATS -->|"feeds into"| CTX["Embedding\nContext String"]
```

**Key guarantee:** Even if an attacker sees thousands of query outputs, they **cannot determine** whether any single user's data was included — mathematically proven by the ε-DP budget.

---

## 🤖 Diagram 4 — ML Ranking Pipeline

**Three-stage AI pipeline: Vector Search → XGBoost Reranking → LLM Copy.**

```mermaid
flowchart TD
    INPUT["📄 Page Context\ne.g. 'summer running shoes\nlightweight under ₹2000'"]

    subgraph STAGE1["Stage 1 · Semantic Retrieval (ChromaDB)"]
        ENC["SentenceTransformer\nall-MiniLM-L6-v2\n→ 384-dim vector"]
        HNSW["ChromaDB HNSW Index\n42 ad embeddings\n(cosine similarity)"]
        TOP10["Top-10 Candidates\n[A1, A13, A15, A5, ...]"]
    end

    subgraph STAGE2["Stage 2 · XGBoost LTR Reranking"]
        FEAT["Feature Engineering\n┌─────────────────────┐\n│ similarity score    │\n│ historical CTR      │\n│ normalized price    │\n│ budget_intent flag  │\n└─────────────────────┘"]
        XGB["XGBoost Regressor\n(trained on 10,000\nsynthetic interactions)"]
        SCORES["p(click) Scores\nA1: 0.809\nA13: 0.761\nA15: 0.698"]
    end

    subgraph STAGE3["Stage 3 · Agentic Copy (Gemini Flash)"]
        PARALLEL["ThreadPoolExecutor\n(parallel API calls)"]
        G1["Gemini → Ad #1 copy"]
        G2["Gemini → Ad #2 copy"]
        G3["Gemini → Ad #3 copy"]
    end

    OUT["🎯 Final Ranked Output\nTop-3 ads with\npersonalised copy\n+ scores + latency"]

    INPUT --> ENC
    ENC --> HNSW
    HNSW --> TOP10
    TOP10 --> FEAT
    FEAT --> XGB
    XGB --> SCORES
    SCORES --> PARALLEL
    PARALLEL --> G1
    PARALLEL --> G2
    PARALLEL --> G3
    G1 --> OUT
    G2 --> OUT
    G3 --> OUT
```

---

## 📬 Diagram 5 — Event Ingestion Pipeline

**How user browsing events are asynchronously captured and stored.**

```mermaid
flowchart LR
    subgraph FRONTEND["🖥️ Streamlit Frontend"]
        BTN["User clicks\nBrowse Page"]
    end

    subgraph INGEST["⚡ FastAPI /track_event"]
        PUSH["_push_event(payload)\nfire-and-forget"]
        REDIS_CHECK{"Redis\navailable?"}
        REDIS_Q["Redis Stream\nrmn:events\n(max 10k)"]
        MEM_Q["In-Memory List\n_event_queue\n(max 10k)"]
    end

    subgraph WORKER["🔄 Async Background Worker"]
        LOOP["asyncio loop\nsleeps 5s"]
        READ["Read up to\n1000 events"]
        FLUSH["flush_events()\n→ INSERT INTO\napi_tracking_events"]
        REBUILD["_rebuild_aggregates()\nforce=True\n(re-JOINs all events)"]
    end

    subgraph DB["🦆 DuckDB"]
        TRACK["api_tracking_events\ntable"]
        AGG["aggregates\ntable\n(category stats)"]
    end

    BTN -->|"POST /track_event\n{user_hash, page_text}"| PUSH
    PUSH --> REDIS_CHECK
    REDIS_CHECK -->|"YES"| REDIS_Q
    REDIS_CHECK -->|"NO fallback"| MEM_Q
    REDIS_Q --> LOOP
    MEM_Q --> LOOP
    LOOP -->|"every 5s"| READ
    READ --> FLUSH
    FLUSH --> TRACK
    FLUSH --> REBUILD
    REBUILD --> AGG
    AGG -->|"next ad request\nuses updated stats"| LOOP
```

---

## 🐳 Diagram 6 — Deployment Architecture

**How the project runs inside a single Docker container on Railway.**

```mermaid
flowchart TB
    subgraph RAILWAY["☁️ Railway (Docker Service)"]
        subgraph DOCKER["🐳 Docker Container (python:3.11-slim)"]
            subgraph STARTUP["start.sh"]
                S1["1. Start Uvicorn\n(background &)\nport 18000 (internal)"]
                S2["2. Health-poll\n/  endpoint\n(max 90s)"]
                S3["3. Start Streamlit\n(foreground)\nport=$PORT"]
                S1 --> S2 --> S3
            end

            subgraph PROCS["Running Processes"]
                UVICORN["uvicorn src.api:app\n127.0.0.1:18000\n(internal only)"]
                STREAMLIT["streamlit run\nsrc/streamlit_app.py\n0.0.0.0:$PORT\n(public)"]
            end

            subgraph VOLUMES["Data (ephemeral)"]
                DUCKDB_F["data/clean_room.duckdb"]
                CHROMA_F["data/chroma/\n(vector index)"]
                MODEL_F["~/.cache/huggingface/\n(model weights)"]
                XGB_F["data/ltr_model.json"]
            end
        end

        PORT_ENV["$PORT env var\n(Railway-assigned)\nHealth check + public traffic"]
    end

    subgraph LOCAL["💻 Local Dev"]
        SH["./start.sh\n(PORT defaults to 7860)"]
        LOCAL_UI["http://localhost:7860"]
        LOCAL_API["http://localhost:18000/docs"]
    end

    STREAMLIT -->|"HTTP calls\n127.0.0.1:18000"| UVICORN
    UVICORN --> DUCKDB_F
    UVICORN --> CHROMA_F
    UVICORN --> XGB_F
    STREAMLIT --> PORT

    SH -->|"same behavior"| LOCAL_UI
    SH -->|"same behavior"| LOCAL_API
```

---

## 🛠 Tech Stack

| Component | Technology | Why |
|---|---|---|
| API Backend | **FastAPI + Uvicorn** | Async execution, background task ingestion |
| Data Clean Room | **DuckDB** | Lightning-fast OLAP on 2.7M rows, persistent DP budget |
| ML Ranking | **XGBoost Regressor** | Learning-to-Rank `p(click)` probability |
| Vector Engine | **ChromaDB (HNSW)** | Scalable cosine similarity, <10ms at 10,000+ ads |
| Agentic LLM | **Gemini 1.5 Flash** | Near-instant hyper-personalized ad copy |
| Embeddings | **all-MiniLM-L6-v2** | CPU-efficient 384-dim sentence embeddings |
| Differential Privacy | **diffprivlib** | IBM Laplace mechanism, persistent daily ε budget |
| Dataset | **Retailrocket (Kaggle)** | 2.7M real e-commerce events |
| Frontend | **Streamlit** | Live demo UI with dark glassmorphism theme |
| Deployment | **Docker + Railway** | Single-container deployment with auto-scaling & $PORT routing |

---

## ⚡ Scale & Performance

| Metric | Target | How |
|---|---|---|
| **Ad Retrieval** | < 10ms | ChromaDB HNSW approximate nearest neighbor |
| **ML Reranking** | < 1ms | XGBoost compiled model (`ltr_model.json`) |
| **LLM Copy** | ~80ms | Parallel Gemini Flash calls via ThreadPoolExecutor |
| **End-to-End** | < 150ms p95 | Async FastAPI + cached embeddings |
| **Catalog Scale** | 10,000+ ads | HNSW index with zero latency degradation |
| **Privacy Budget** | ε ≤ 0.9/day | DuckDB-persisted budget, resets daily |

---

## 🔐 Differential Privacy Math

Every aggregate query on user data has **Laplace noise** mathematically injected:

$$\mathcal{M}(x) = x + \text{Lap}\!\left(\frac{\Delta f}{\varepsilon}\right)$$

- **ε = 0.1** per noise application (3 noisy values = 0.3ε per ad request)
- **ε_max = 0.9** — queries are hard-blocked beyond this  
- **Attack Protection:** The `ε_used` state is stored **persistently in DuckDB**. A malicious attacker cannot restart the server to restore budget and extract raw PII — the budget survives reboots.

---

## 🚀 Local Run Instructions

**Step 1 — Clone & install dependencies**
```bash
git clone https://github.com/shubhamkya/rmn-engine
cd rmn-engine
pip install -r requirements.txt
```

**Step 2 — Set API Key**
Create a `.env` file:
```env
GEMINI_API_KEY=your_api_key_here
```

**Step 3 — One-command startup**
```bash
./start.sh
```

This automatically starts **FastAPI on port 8000** (background) and **Streamlit on port 7860** (foreground).

| Service | URL |
|---|---|
| 🖥️ Streamlit UI | http://localhost:7860 |
| ⚙️ FastAPI Swagger | http://localhost:8000/docs |

---

## 📁 Repository Structure

```
rmn-engine/
├── data/                        # Auto-generated at runtime
│   ├── clean_room.duckdb        # DuckDB database (2.7M events + aggregates)
│   ├── ltr_model.json           # Trained XGBoost model
│   └── chroma/                  # ChromaDB vector index (42+ ad embeddings)
│
├── src/
│   ├── config.py                # 🔧 All constants, API keys, ad catalogue
│   ├── api.py                   # ⚡ FastAPI app + async background worker
│   ├── clean_room.py            # 🦆 DuckDB schema, DP queries, budget tracking
│   ├── embeddings.py            # 🧠 SentenceTransformer + ChromaDB integration
│   ├── ltr.py                   # 📈 XGBoost LTR trainer & inference
│   ├── ranking.py               # 🎯 Full ranking pipeline (Chroma → XGB → LLM)
│   ├── agent.py                 # ✨ Gemini Flash agentic copy generation
│   └── streamlit_app.py         # 🖥️ 4-tab Streamlit dashboard
│
├── Dockerfile                   # 🐳 Railway Docker container
├── start.sh                     # 🚀 Startup orchestration script
├── requirements.txt             # 📦 Python dependencies (CPU-optimized)
└── README.md
```

---

## 📊 Data Flow Summary

```
User browses page
      │
      ▼
POST /track_event ──► In-Memory Queue ──► [every 5s] ──► DuckDB
      │                                                       │
      ▼                                                       ▼
GET /get_ad                                          Rebuild Aggregates
      │
      ├─► DuckDB ──► Laplace DP ──► Noisy Category Stats
      │                                      │
      ├─────────────────────────────────────►│
      │                                      ▼
      │                            SentenceTransformer
      │                            (encode context vector)
      │                                      │
      │                                      ▼
      │                            ChromaDB HNSW Query
      │                            (Top-10 semantic matches)
      │                                      │
      │                                      ▼
      │                            XGBoost LTR Scoring
      │                            (p(click) per candidate)
      │                                      │
      │                               ┌──────┴──────┐
      │                               ▼             ▼
      │                         Gemini #1      Gemini #2,3
      │                         (parallel copy generation)
      │                               └──────┬──────┘
      │                                      │
      ◄──────────────────────────────────────┘
              Top-3 Ranked Ads + Personalized Copy
              + latency_ms + epsilon_used + ctr_lift_pct
```
