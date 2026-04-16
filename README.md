

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

Retailers (Myntra, Nykaa, etc.) want to show **personalised ads on their own website** without sending raw user data to advertisers. ShyftLabs solves this with a **Retail Media Network (RMN)** architecture — a Data Clean Room that keeps first-party data private while enabling smart ad targeting.

| ShyftLabs Real World | This Project Architecture |
|---|---|
| First-party Data Clean Room | **DuckDB** Async Background Worker (Scalable Flush) |
| Mathematical privacy guarantee | **diffprivlib Laplace** mechanism (Persistent DuckDB tracking) |
| Contextual / Semantic match | **ChromaDB** Vector Engine (`all-MiniLM-L6-v2`) |
| Real-time CTR Prediction | **XGBoost** Learning-to-Rank (LTR) |
| Ad Copy Personalization | **Gemini 1.5 Flash** Agentic generation |
| Advertiser Reporting | **Per-advertiser ε budget** isolated per brand |

---

## 🏗️ The Two Flows — Core Architecture Concept

This system handles **two completely separate flows** that must never be confused:

```
┌─────────────────────────────────────────────────────────────────┐
│  FLOW 1 — Ad Serving (User Side)                                │
│                                                                  │
│  User browses Myntra → RMN Engine picks best ad → User sees ad  │
│  Uses: get_raw_category_stats()  ← NO epsilon burned            │
│  Why: The user is not a threat. Stats stay internal.            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  FLOW 2 — Advertiser Reporting (Nike Side)                      │
│                                                                  │
│  Nike queries Myntra → Clean Room returns NOISY stats → Nike    │
│  Uses: get_advertiser_stats()    ← Burns Nike's own ε budget    │
│  Why: Nike IS a threat. DP prevents user re-identification.     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🗺️ Diagram 1 — Master System Architecture

**The complete end-to-end picture: both flows, every component.**

```mermaid
flowchart TB
    subgraph USERS["👤 User / Browser (Flow 1)"]
        UI["🖥️ Streamlit UI\n(port 7860)"]
    end

    subgraph ADVERTISERS["🏢 Advertiser / Nike (Flow 2)"]
        ADV["📡 GET /advertiser/stats\n(Nike's API call)"]
    end

    subgraph API["⚡ FastAPI Backend (port 18000)"]
        direction TB
        EP_AD["/get_ad\n(GET) — Flow 1"]
        EP_TRACK["/track_event\n(POST) — Flow 1"]
        EP_ADV["/advertiser/stats\n(GET) — Flow 2"]
        EP_METRICS["/metrics\n(GET)"]
        WORKER["🔄 Async Background\nEvent Worker\n(every 5s)"]
    end

    subgraph AI["🤖 AI / ML Layer (Flow 1 only)"]
        EMBED["🧠 SentenceTransformer\nall-MiniLM-L6-v2\n(CPU Encoder)"]
        CHROMA["📦 ChromaDB\nVector Store\n(HNSW Cosine)"]
        XGB["📈 XGBoost LTR\nML Ranker\n(p_click predictor)"]
        GEMINI["✨ Gemini 1.5 Flash\nAgentic Copywriter\n(parallel calls)"]
    end

    subgraph PRIVACY["🔐 Privacy Clean Room (DuckDB)"]
        DUCK["🦆 DuckDB\n(2.7M Retailrocket events\n+ Aggregates + Ads)"]
        RAW["get_raw_category_stats()\nFlow 1 — NO ε cost\n(user is not a threat)"]
        DP["get_advertiser_stats()\nFlow 2 — Burns ε per brand\nLaplace Noise Applied"]
        SYS_BUDGET["💰 System ε Budget\n(legacy /metrics)"]
        ADV_BUDGET["💰 Per-Advertiser ε Budget\nNike: 0.6/0.9\nSamsung: 0.3/0.9\nFabIndia: 0.3/0.9\nLakme: 0.3/0.9"]
    end

    subgraph QUEUE["📬 Event Queue"]
        MEM["In-Memory Queue\n(fallback)"]
        REDIS["Redis Stream\n(optional)"]
    end

    UI -->|"1. Browse page"| EP_TRACK
    UI -->|"2. Get ranked ad"| EP_AD
    ADV -->|"Nike queries stats"| EP_ADV

    EP_TRACK --> MEM
    EP_TRACK --> REDIS
    MEM --> WORKER
    REDIS --> WORKER
    WORKER -->|"flush every 5s"| DUCK

    EP_AD --> DUCK
    DUCK --> RAW
    RAW -->|"real stats (no noise)"| EMBED
    EMBED -->|"context vector"| CHROMA
    CHROMA -->|"Top-10 candidates"| XGB
    XGB -->|"p(click) scores"| GEMINI
    GEMINI -->|"ranked + copy"| EP_AD
    EP_AD -->|"Ad + metadata"| UI

    EP_ADV --> DUCK
    DUCK --> DP
    DP --> ADV_BUDGET
    ADV_BUDGET -->|"noisy stats per brand"| EP_ADV
    EP_ADV -->|"DP-noisy campaign data"| ADV
```

---

## 🔄 Diagram 2 — Flow 1: Ad Request (User Sees Ad)

**What happens in `< 150ms` when a user clicks "Browse Page & Get Ad".**

```mermaid
sequenceDiagram
    actor User
    participant ST as 🖥️ Streamlit UI
    participant API as ⚡ FastAPI /get_ad
    participant CR as 🦆 DuckDB (Raw Stats)
    participant EMB as 🧠 SentenceTransformer
    participant VDB as 📦 ChromaDB
    participant XGB as 📈 XGBoost LTR
    participant LLM as ✨ Gemini Flash

    User->>ST: Selects page context & clicks Browse
    ST->>API: GET /get_ad?user_hash=...&page_text=...

    Note over API,CR: Step 1 · Raw Internal Stats (~5ms) — NO ε burned
    API->>CR: get_raw_category_stats()
    CR-->>API: Real aggregates (views, carts, purchases) — internal only

    Note over API,VDB: Step 2 · Semantic Retrieval (~10ms)
    API->>EMB: encode(page_text + raw_stats)
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

## 📡 Diagram 3 — Flow 2: Advertiser Reporting (Nike Queries Stats)

**What happens when Nike calls the advertiser API.**

```mermaid
sequenceDiagram
    actor Nike as 🏢 Nike Marketing Team
    participant API as ⚡ FastAPI /advertiser/stats
    participant BUDGET as 💰 advertiser_budget (DuckDB)
    participant AGG as 🦆 aggregates (DuckDB)
    participant DP as 📐 Laplace DP

    Nike->>API: GET /advertiser/stats?advertiser_id=Nike

    Note over API,BUDGET: Step 1 · Check Nike's Own Budget
    API->>BUDGET: SELECT epsilon_used WHERE advertiser_id='Nike' AND date=TODAY
    BUDGET-->>API: current_eps = 0.30

    alt Budget OK (0.30 + 0.30 ≤ 0.90)
        API->>BUDGET: UPDATE epsilon_used = 0.60 (burn 0.30ε)
        Note over API,DP: Step 2 · Fetch + Noisify Stats
        API->>AGG: SELECT SUM(views, carts, purchases) for shoes category
        AGG-->>API: Real stats: 45231 views, 3412 carts, 891 purchases
        API->>DP: Apply Laplace noise (ε=0.1) to each value
        DP-->>API: Noisy: ~45238 views, ~3409 carts, ~895 purchases
        API-->>Nike: {noisy_views, noisy_carts, noisy_purchases, ε_used=0.60, ε_remaining=0.30}
    else Budget Exhausted (would exceed 0.90)
        API-->>Nike: HTTP 429 — Privacy budget exhausted. Resets tomorrow.
    end

    Note over Nike: Nike sees TRENDS not INDIVIDUALS
    Note over Nike: Cannot re-identify any specific Myntra user
```

---

## 🔐 Diagram 4 — Differential Privacy Pipeline (Updated)

**How user data is protected — and WHERE it applies.**

```mermaid
flowchart LR
    subgraph WHEN["❓ When is DP Applied?"]
        Q1{"Who is\nasking?"}
        Q1 -->|"User getting ad\n(Flow 1)"| NO_DP["✅ NO DP needed\nget_raw_category_stats()\nStats stay internal\nε = 0 burned"]
        Q1 -->|"Nike querying stats\n(Flow 2)"| YES_DP["🔐 DP REQUIRED\nget_advertiser_stats()\nLaplace noise added\nε = 0.3 burned"]
    end

    subgraph DP_LAYER["🔐 Differential Privacy (Laplace Mechanism)"]
        CHECK{"Nike ε_used + 0.3\n≤ ε_max (0.9)?"}
        BURN["Burn 0.3ε from\nNike's budget"]
        NOISE["Add Laplace Noise\nnoise ~ Lap(Δf/ε)\nΔf=1, ε=0.1"]
        BLOCK["🚫 Block Query\nHTTP 429\nBudget Exhausted"]
    end

    subgraph OUT["📤 Noisy Output (Safe for Nike)"]
        NSTATS["Noisy Stats\ndp_views: ~45,238\ndp_carts: ~3,409\ndp_purchases: ~895"]
        PERSIST["DuckDB advertiser_budget:\nNike  | 2026-04-16 | 0.60ε\nSamsung| 2026-04-16 | 0.30ε"]
    end

    YES_DP --> CHECK
    CHECK -->|"YES · budget OK"| BURN
    BURN --> NOISE
    NOISE --> NSTATS
    BURN --> PERSIST
    CHECK -->|"NO · exhausted"| BLOCK
```

---

## 🏢 Diagram 5 — Per-Advertiser Budget Isolation

**Each brand has their own daily ε quota — completely independent.**

```mermaid
flowchart TD
    POOL["🦆 DuckDB\nadvertiser_budget table"]

    subgraph NIKE["Nike's Budget"]
        N1["Query 1: ε = 0.30"]
        N2["Query 2: ε = 0.60"]
        N3["Query 3: ε = 0.90 ✅ max"]
        N4["Query 4: 🚫 BLOCKED"]
    end

    subgraph SAMSUNG["Samsung's Budget (independent)"]
        S1["Query 1: ε = 0.30"]
        S2["Query 2: ε = 0.60"]
        S3["Still has 0.30ε left"]
    end

    POOL --> NIKE
    POOL --> SAMSUNG

    NIKEF["Nike burns ALL budget\n→ Samsung is UNAFFECTED"]
    NIKE --> NIKEF
    SAMSUNG --> NIKEF
```

---

## 🤖 Diagram 6 — ML Ranking Pipeline (Flow 1)

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

## 📬 Diagram 7 — Event Ingestion Pipeline

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

## 🐳 Diagram 8 — Deployment Architecture

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
                DUCKDB_F["data/clean_room.duckdb\n(events + aggregates\n+ privacy_budget\n+ advertiser_budget)"]
                CHROMA_F["data/chroma/\n(vector index)"]
                XGB_F["data/ltr_model.json"]
            end
        end

        PORT_ENV["$PORT env var\n(Railway-assigned)\nHealth check + public traffic"]
    end

    STREAMLIT -->|"HTTP calls\n127.0.0.1:18000"| UVICORN
    UVICORN --> DUCKDB_F
    UVICORN --> CHROMA_F
    UVICORN --> XGB_F
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
| **User Ad Requests** | Unlimited/day | Flow 1 uses raw stats — no ε cost |
| **Advertiser Queries** | 3/day per brand | Flow 2 burns 0.3ε per query, max 0.9ε |

---

## 🔐 Differential Privacy Math

Every aggregate query on user data (Flow 2 only) has **Laplace noise** mathematically injected:

$$\mathcal{M}(x) = x + \text{Lap}\!\left(\frac{\Delta f}{\varepsilon}\right)$$

- **ε = 0.1** per noise application (3 noisy values = 0.3ε per advertiser query)
- **ε_max = 0.9** — queries are hard-blocked beyond this
- **Per-advertiser budget** — Nike's queries don't consume Samsung's budget
- **Attack Protection:** The `ε_used` state is stored **persistently in DuckDB** in `advertiser_budget` table. A malicious attacker cannot restart the server to restore budget.

### Why DP only in Flow 2 (not Flow 1)?

```
Flow 1 (User → Ad):
  Stats are used INTERNALLY by the ML pipeline only.
  They never leave the system. The user is not a threat.
  → Raw stats, zero ε cost, unlimited requests.

Flow 2 (Nike → Stats):
  Stats are RETURNED to an external party (Nike).
  Nike could run difference attacks to identify users.
  → Noisy stats, 0.3ε per query, hard cap at 0.9ε/day.
```

---

## 📡 API Reference

### Flow 1 — Ad Serving

| Endpoint | Method | Purpose | ε Cost |
|---|---|---|---|
| `/` | GET | Health check | 0 |
| `/healthz` | GET | Railway health probe | 0 |
| `/track_event` | POST | Log user browsing event (fire-and-forget) | 0 |
| `/get_ad` | GET | Return ranked personalised ad | **0** (uses raw stats) |
| `/metrics` | GET | System-level CTR lift + budget stats | 0 |

### Flow 2 — Advertiser Reporting

| Endpoint | Method | Purpose | ε Cost |
|---|---|---|---|
| `/advertiser/stats` | GET | Return DP-noisy campaign stats for one advertiser | **0.3ε per call** |

#### `/advertiser/stats` Request
```
GET /advertiser/stats?advertiser_id=Nike&retailer=Myntra
```

#### `/advertiser/stats` Response
```json
{
  "advertiser_id":     "Nike",
  "category":          "shoes",
  "noisy_views":       45238,
  "noisy_carts":       3409,
  "noisy_purchases":   895,
  "cart_rate_pct":     7.53,
  "purchase_rate_pct": 1.97,
  "epsilon_used":      0.60,
  "epsilon_remaining": 0.30,
  "epsilon_max":       0.90,
  "dp_noise_applied":  true
}
```

#### Error: Budget Exhausted
```
HTTP 429 Too Many Requests
{"detail": "[Nike] Privacy budget exhausted: ε=0.90 + 0.3 > 0.9"}
```

---

## 🎮 Demo Advertisers

| Advertiser | Category | Ad IDs | Brand Color |
|---|---|---|---|
| **Nike** | Shoes | A1, A5, A9, A13, A14, A15, A16, A17 | #f97316 (Orange) |
| **Samsung** | Electronics | A2, A6, A10, A19–A24 | #3b82f6 (Blue) |
| **FabIndia** | Ethnic Wear | A3, A7, A25–A30 | #f59e0b (Amber) |
| **Lakme** | Skincare | A4, A8, A31–A36 | #ec4899 (Pink) |

---

## 🚀 Local Run Instructions

**Step 1 — Clone & install dependencies**
```bash
git clone https://github.com/shubhamkya/rmn-engine
cd rmn-engine
pip install -r requirements.txt
```

**Step 2 — Set API Key**
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

**Step 3 — One-command startup**
```bash
./start.sh
```

| Service | URL |
|---|---|
| 🖥️ Streamlit UI | http://localhost:7860 |
| ⚙️ FastAPI Swagger | http://localhost:18000/docs |

---

## 📁 Repository Structure

```
rmn-engine/
├── data/
│   ├── clean_room.duckdb        # DuckDB (2.7M events + aggregates + ε budgets)
│   ├── ltr_model.json           # Trained XGBoost model
│   └── chroma/                  # ChromaDB vector index (42 ad embeddings)
│
├── src/
│   ├── config.py                # 🔧 Constants, API keys, ad catalogue, ADVERTISER_CATALOGUE
│   ├── api.py                   # ⚡ FastAPI — /get_ad, /track_event, /advertiser/stats
│   ├── clean_room.py            # 🦆 DuckDB schema, raw stats, DP stats, per-advertiser budget
│   ├── embeddings.py            # 🧠 SentenceTransformer + ChromaDB integration
│   ├── ltr.py                   # 📈 XGBoost LTR trainer & inference
│   ├── ranking.py               # 🎯 Full ranking pipeline (Chroma → XGB → LLM)
│   ├── agent.py                 # ✨ Gemini Flash agentic copy generation
│   └── streamlit_app.py         # 🖥️ 5-tab Streamlit dashboard
│
├── Dockerfile                   # 🐳 Railway Docker container
├── start.sh                     # 🚀 Startup orchestration script
├── requirements.txt             # 📦 Python dependencies
└── README.md
```

---

## 📊 Data Flow Summary

### Flow 1 — User Receives Ad
```
User browses "summer running shoes under ₹2000"
      │
      ├──► POST /track_event ──► Queue ──► [every 5s] ──► DuckDB
      │
      └──► GET /get_ad
                │
                ├─► DuckDB get_raw_category_stats() ─► Real stats (NO noise, NO ε)
                │                                             │
                ├─────────────────────────────────────────────┤
                │                                             ▼
                │                                   SentenceTransformer
                │                                   (encode context vector)
                │                                             │
                │                                             ▼
                │                                   ChromaDB HNSW Query
                │                                   (Top-10 semantic matches)
                │                                             │
                │                                             ▼
                │                                   XGBoost LTR Scoring
                │                                   (p(click) per candidate)
                │                                             │
                │                                    ┌────────┴────────┐
                │                                    ▼                 ▼
                │                              Gemini #1          Gemini #2,3
                │                              (parallel copy generation)
                │                                    └────────┬────────┘
                │                                             │
                ◄─────────────────────────────────────────────┘
                        Top-3 Ranked Ads + Personalised Copy
```

### Flow 2 — Nike Checks Campaign Performance
```
Nike calls: GET /advertiser/stats?advertiser_id=Nike
      │
      └──► DuckDB advertiser_budget: check Nike's ε used today
                │
                ├─ [0.60 + 0.30 = 0.90 ≤ 0.90] → OK
                │         Burn 0.30ε from Nike's budget → persist to DuckDB
                │
                ├─► DuckDB aggregates: fetch real views/carts/purchases
                │
                ├─► Laplace noise applied to each value
                │         views:     45231 + Lap(10) = 45238
                │         carts:     3412  + Lap(10) = 3409
                │         purchases: 891   + Lap(10) = 895
                │
                └──► Nike sees noisy stats (useful for bidding, safe for users)
                         ε_used=0.60, ε_remaining=0.30, queries_left=1
```
