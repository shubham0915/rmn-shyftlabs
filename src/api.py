"""
api.py — FastAPI Backend for the RMN Engine
Agent 2: Backend Specialist

Endpoints:
  POST /track_event   — Accept user event (user_hash, page_text)
  GET  /get_ad        — Return ranked ad + metadata in <100ms
  GET  /metrics       — Return live CTR lift + ε budget

Architecture:
  - Redis stream for event ingestion (falls back to in-memory queue if Redis absent)
  - Embedding + ranking are imported from Agent 3's modules
  - DP stats are fetched from Agent 4's clean_room module
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# NOTE: Do NOT set HF_HUB_OFFLINE=1 here.
# On Hugging Face Spaces the model is not pre-cached, so it must be downloaded
# at first startup. Keeping online mode allows SentenceTransformer to fetch
# all-MiniLM-L6-v2 on first boot.

import time
import uuid
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ------- Internal modules -------
from src.config import REDIS_HOST, REDIS_PORT, REDIS_DB, STREAM_KEY
from src.clean_room import (
    init_db, get_all_ads, get_dp_category_stats,
    get_raw_category_stats,
    get_advertiser_stats, get_advertiser_epsilon,
    simulate_ctr_lift, get_epsilon_used, flush_events
)
from src.embeddings import warmup
from src.ranking import rank_ads, get_similarity_scores
from src.ltr import train_and_save_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Redis helper (graceful fallback)
# ---------------------------------------------------------------------------
_redis_client = None
_redis_dead = False
_event_queue: list[dict] = []          # in-memory fallback

def _get_redis():
    global _redis_client, _redis_dead
    if _redis_client is not None:
        return _redis_client
    if _redis_dead:
        return None
        
    try:
        import redis
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, socket_timeout=0.1)
        r.ping()
        _redis_client = r
        logger.info("Redis connected ✓")
    except Exception:
        logger.warning("Redis not available — using in-memory queue fallback.")
        _redis_dead = True
        _redis_client = None
    return _redis_client


def _push_event(payload: dict) -> None:
    r = _get_redis()
    if r:
        r.xadd(STREAM_KEY, {k: str(v) for k, v in payload.items()}, maxlen=10000)
    else:
        _event_queue.append(payload)
        if len(_event_queue) > 10000:
            _event_queue.pop(0)


# ---------------------------------------------------------------------------
# Background Async Event Worker
# ---------------------------------------------------------------------------
async def _event_worker_loop():
    """Runs continuously in the background to flush tracked events to DuckDB."""
    logger.info("[Worker] Background event ingester started.")
    while True:
        await asyncio.sleep(5)
        
        events_to_flush = []
        r = _get_redis()
        
        if r:
            try:
                # Read from Redis stream explicitly mapping bytes to strings
                messages = r.xread({STREAM_KEY: '0'}, count=1000)
                if messages:
                    stream_msgs = messages[0][1]
                    msg_ids_to_del = []
                    for msg_id, msg_data in stream_msgs:
                        try:
                            event = {k.decode('utf-8'): v.decode('utf-8') for k, v in msg_data.items()}
                            events_to_flush.append(event)
                            msg_ids_to_del.append(msg_id)
                        except Exception:
                            # Safely ignore corrupted events
                            msg_ids_to_del.append(msg_id)
                    
                    if msg_ids_to_del:
                        r.xdel(STREAM_KEY, *msg_ids_to_del)
            except Exception as e:
                logger.error(f"[Worker] Redis stream read failed: {e}")
        else:
            # Fallback to local memory queue
            global _event_queue
            if _event_queue:
                events_to_flush = _event_queue[:1000]
                _event_queue = _event_queue[1000:]
                
        # Send to DB
        if events_to_flush:
            loop = asyncio.get_event_loop()
            flushed = await loop.run_in_executor(None, flush_events, events_to_flush)
            if flushed > 0:
                logger.info(f"[Worker] Successfully securely flushed {flushed} events into DB.")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Startup progress logger
# ---------------------------------------------------------------------------
_TOTAL_STEPS = 7

def _progress(step: int, label: str, note: str = "", elapsed: float = 0.0) -> None:
    """Print a formatted progress line to the log."""
    filled  = "█" * step
    empty   = "░" * (_TOTAL_STEPS - step)
    pct     = int(step / _TOTAL_STEPS * 100)
    timing  = f"  ({elapsed:.1f}s)" if elapsed else ""
    note_str = f"  » {note}" if note else ""
    logger.info(
        f"\n"
        f"  ┌─────────────────────────────────────────────────┐\n"
        f"  │  RMN ENGINE STARTUP  [{filled}{empty}] {pct:>3}%  Step {step}/{_TOTAL_STEPS}  │\n"
        f"  │  ▶ {label:<43}│\n"
        f"  └─────────────────────────────────────────────────┘{timing}{note_str}"
    )


# ---------------------------------------------------------------------------
# Lifespan: DB init + embedding warmup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    boot_start = time.perf_counter()
    logger.info(
        "\n\n"
        "  ╔═══════════════════════════════════════════════════╗\n"
        "  ║        🛒  RMN ENGINE — BOOTING UP  🛒            ║\n"
        "  ║  Privacy-Preserving Agentic Ad Serving Engine     ║\n"
        "  ╚═══════════════════════════════════════════════════╝"
    )

    loop = asyncio.get_event_loop()

    # ── Step 1: DuckDB Init ──────────────────────────────────
    _progress(0, "Initialising DuckDB Clean Room …", "Loading 2.7M Retailrocket events")
    t = time.perf_counter()
    await loop.run_in_executor(None, init_db)
    _progress(1, "DuckDB Clean Room ✓", "Schema + CSV + Aggregates ready", time.perf_counter() - t)

    # ── Step 2: Download embedding model if needed ───────────
    _progress(1, "Checking embedding model cache …", "all-MiniLM-L6-v2 (may download ~80 MB on first run)")
    t = time.perf_counter()
    # warmup() handles this — we just log before and after
    logger.info("  [Model] If downloading: this is a one-time ~80 MB download from HuggingFace …")

    # ── Step 3: Load SentenceTransformer ────────────────────
    _progress(2, "Loading SentenceTransformer model …", "all-MiniLM-L6-v2 on CPU")
    t3 = time.perf_counter()
    # warmup() calls get_model() then encode() then populates ChromaDB
    from src.embeddings import get_model
    get_model()   # load weights — this is the slow part on first boot
    _progress(3, "SentenceTransformer loaded ✓", f"Model ready in {time.perf_counter()-t3:.1f}s", time.perf_counter() - t)

    # ── Step 4: ChromaDB warmup + ad embedding ───────────────
    _progress(3, "Building ChromaDB vector index …", "Embedding 42 ads into HNSW index")
    t = time.perf_counter()
    from src.embeddings import encode, get_collection
    from src.config import SYNTHETIC_ADS
    collection = get_collection()
    if collection.count() == 0:
        logger.info(f"  [ChromaDB] First run — embedding {len(SYNTHETIC_ADS)} ads …")
        for i, ad in enumerate(SYNTHETIC_ADS, 1):
            price_text = f" price {int(ad['price'])}" if ad.get("price") else ""
            vec = encode(f"{ad['title']} {ad.get('desc', '')}{price_text}")
            collection.add(
                ids=[ad["ad_id"]],
                embeddings=[vec],
                metadatas=[{
                    "title": ad["title"], "category": ad["category"],
                    "ctr": float(ad["ctr"]),
                    "price": float(ad["price"]) if ad.get("price") else 0.0,
                    "desc": ad.get("desc", ""),
                }]
            )
            if i % 10 == 0 or i == len(SYNTHETIC_ADS):
                logger.info(f"  [ChromaDB] Embedded {i}/{len(SYNTHETIC_ADS)} ads …")
    else:
        logger.info(f"  [ChromaDB] Cache hit — {collection.count()} ads already indexed. Skipping.")
    _progress(4, "ChromaDB vector index ✓", f"{collection.count()} ads indexed", time.perf_counter() - t)

    # ── Step 5: XGBoost LTR model ───────────────────────────
    _progress(4, "Loading XGBoost LTR model …", "Training on 10,000 synthetic interactions if first run")
    t = time.perf_counter()
    train_and_save_model()
    _progress(5, "XGBoost LTR model ✓", "p(click) ranker ready", time.perf_counter() - t)

    # ── Step 6: Redis / event queue ──────────────────────────
    _progress(5, "Connecting to Redis event queue …", "Falls back to in-memory if Redis absent")
    t = time.perf_counter()
    _get_redis()
    redis_status = "Redis connected ✓" if _redis_client else "In-memory queue (Redis not available)"
    _progress(6, redis_status, "", time.perf_counter() - t)

    # ── Step 7: Background worker ────────────────────────────
    _progress(6, "Starting async event ingestion worker …", "Flushes events to DuckDB every 5s")
    worker_task = asyncio.create_task(_event_worker_loop())
    _progress(7, "Background worker started ✓", "Listening for events")

    # ── READY banner ─────────────────────────────────────────
    total_elapsed = time.perf_counter() - boot_start
    logger.info(
        f"\n\n"
        f"  ╔═══════════════════════════════════════════════════╗\n"
        f"  ║   ✅  RMN ENGINE IS LIVE AND READY!               ║\n"
        f"  ║                                                   ║\n"
        f"  ║   🌐  Streamlit UI  → http://localhost:7860       ║\n"
        f"  ║   ⚙️   API Docs      → http://localhost:8000/docs  ║\n"
        f"  ║   📊  Total boot time: {total_elapsed:>5.1f}s                  ║\n"
        f"  ╚═══════════════════════════════════════════════════╝\n"
    )

    yield   # ← server is running here

    worker_task.cancel()
    logger.info("  [RMN] Shutting down gracefully. Goodbye.")



# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="RMN Engine API",
    description="Privacy-Preserving Retail Media Network — ShyftLabs AdTech Demo",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class EventPayload(BaseModel):
    user_hash: str
    page_text: str
    retailer:  Optional[str] = "unknown"


class RankedAd(BaseModel):
    ad_id:       str
    title:       str
    ad_copy:     str          # renamed from 'copy' — avoids shadowing BaseModel.copy()
    category:    str
    ctr:         float
    price:       float | None = None
    similarity:  float
    final_score: float
    rank:        int | None = None


class AdResponse(BaseModel):
    ad_id:            str
    title:            str
    ad_copy:          str          # renamed from 'copy' — avoids shadowing BaseModel.copy()
    category:         str
    ctr:              float
    price:            float | None = None
    similarity:       float
    final_score:      float
    latency_ms:       float
    epsilon_used:     float
    ctr_lift_pct:     float
    top_ads:          list[RankedAd] = []


class AdvertiserStatsResponse(BaseModel):
    advertiser_id:      str
    category:           str
    noisy_views:        int
    noisy_carts:        int
    noisy_purchases:    int
    cart_rate_pct:      float
    purchase_rate_pct:  float
    epsilon_used:       float
    epsilon_remaining:  float
    epsilon_max:        float
    dp_noise_applied:   bool


class MetricsResponse(BaseModel):
    epsilon_used:     float
    epsilon_max:      float
    random_ctr:       float
    semantic_ctr:     float
    lift_pct:         float
    event_queue_size: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", tags=["health"])
def root():
    """Health check."""
    return {"status": "ok", "service": "RMN Engine API"}


@app.get("/healthz", tags=["health"])
def healthz():
    """Railway / K8s-compatible health check endpoint."""
    return {"status": "ok", "service": "RMN Engine API"}


@app.post("/track_event", tags=["events"])
async def track_event(payload: EventPayload):
    """
    Accept a user browsing event and push it to the Redis event stream.
    This is fire-and-forget; the Clean Room reads the stream asynchronously.
    """
    event = {
        "event_id":  str(uuid.uuid4()),
        "user_hash": payload.user_hash,
        "page_text": payload.page_text,
        "retailer":  payload.retailer,
        "ts":        str(time.time()),
    }
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _push_event, event)
    return {"status": "accepted", "event_id": event["event_id"]}


@app.get("/get_ad", response_model=AdResponse, tags=["serving"])
async def get_ad(
    user_hash:  str = "anon",
    page_text:  str = "products",
    retailer:   Optional[str] = None,
):
    """
    Core ad-serving endpoint.

    Steps:
      1. Fetch DP-noisy category stats from Clean Room.
      2. Rank all ads via cosine similarity + CTR blend.
      3. Return top-1 ad with personalised copy and latency measurement.

    Target p95 latency: <100ms (fully on CPU with cached embeddings).
    """
    t0 = time.perf_counter()

    loop = asyncio.get_event_loop()

    logger.info("  -> Starting get_all_ads")
    t_step = time.perf_counter()
    ads      = await loop.run_in_executor(None, get_all_ads)
    logger.info(f"  -> get_all_ads took {(time.perf_counter() - t_step)*1000:.1f}ms")
    
    t_step = time.perf_counter()
    # Flow 1 — internal ad serving: use raw stats (no epsilon cost, user is not a threat)
    raw_stats = await loop.run_in_executor(None, get_raw_category_stats)
    logger.info(f"  -> get_raw_category_stats took {(time.perf_counter() - t_step)*1000:.1f}ms [NO epsilon burned]")

    if not ads:
        raise HTTPException(status_code=503, detail="Ad catalogue unavailable")

    logger.info("  -> Starting rank_ads (PyTorch inference)")
    t_step = time.perf_counter()
    # Rank (Running synchronously on Main Thread prevents PyTorch thread deadlocks on macOS)
    ranked = rank_ads(page_text, ads, raw_stats)
    logger.info(f"  -> rank_ads took {(time.perf_counter() - t_step)*1000:.1f}ms")

    if not ranked:
        raise HTTPException(status_code=503, detail="Ranking returned no results")

    best = ranked[0]
    top_ads = ranked[:3]

    logger.info("  -> Starting simulate_ctr_lift")
    t_step = time.perf_counter()
    # CTR lift
    scores   = [r["similarity"] for r in ranked]
    lift_res = simulate_ctr_lift(scores)
    logger.info(f"  -> simulate_ctr_lift took {(time.perf_counter() - t_step)*1000:.1f}ms")

    latency_ms = (time.perf_counter() - t0) * 1000

    logger.info(
        f"[get_ad] user={user_hash[:6]} ad={best['ad_id']} "
        f"score={best['final_score']:.3f} latency={latency_ms:.1f}ms ε={get_epsilon_used():.2f}"
    )

    return AdResponse(
        ad_id        = best["ad_id"],
        title        = best["title"],
        ad_copy      = best["copy"],
        category     = best["category"],
        ctr          = best["ctr"],
        price        = best.get("price"),
        similarity   = best["similarity"],
        final_score  = best["final_score"],
        latency_ms   = round(latency_ms, 1),
        epsilon_used = get_epsilon_used(),
        ctr_lift_pct = lift_res["lift_pct"],
        top_ads      = [
            RankedAd(
                ad_id       = a["ad_id"],
                title       = a["title"],
                ad_copy     = a["copy"],
                category    = a["category"],
                ctr         = a["ctr"],
                price       = a.get("price"),
                similarity  = a["similarity"],
                final_score = a["final_score"],
                rank        = a.get("rank"),
            )
            for a in top_ads
        ],
    )


@app.get("/metrics", response_model=MetricsResponse, tags=["monitoring"])
async def metrics():
    """Live metrics: ε budget + simulated CTR lift."""
    from src.config import EPSILON_MAX
    
    # 1. Fetch ads synchronously
    loop = asyncio.get_event_loop()
    ads = await loop.run_in_executor(None, get_all_ads)
    
    # 2. PyTorch & Compute strictly on Main Thread
    scores = get_similarity_scores("general retail products", ads)
    lift   = simulate_ctr_lift(scores)
    return MetricsResponse(
        epsilon_used     = get_epsilon_used(),
        epsilon_max      = EPSILON_MAX,
        random_ctr       = lift["random_ctr"],
        semantic_ctr     = lift["semantic_ctr"],
        lift_pct         = lift["lift_pct"],
        event_queue_size = len(_event_queue),
    )


# ---------------------------------------------------------------------------
# Advertiser Stats Endpoint (Flow 2 — Nike queries Myntra's Clean Room)
# ---------------------------------------------------------------------------
@app.get("/advertiser/stats", response_model=AdvertiserStatsResponse, tags=["advertiser"])
async def advertiser_stats(
    advertiser_id: str = "Nike",
    retailer:      str = "Myntra",
):
    """
    Flow 2 — Advertiser-Facing API.

    Nike (or any advertiser) calls this to get campaign performance stats.
    - Returns DP-NOISY aggregate data (views, carts, purchases)
    - Burns from Nike's OWN daily epsilon budget (isolated from other advertisers)
    - Hard-blocks when Nike's budget is exhausted (resets next day)

    This is the correct place for Differential Privacy to be applied.
    It protects Myntra's users from being re-identified by Nike.
    """
    from src.config import ADVERTISER_CATALOGUE
    loop = asyncio.get_event_loop()

    if advertiser_id not in ADVERTISER_CATALOGUE:
        raise HTTPException(
            status_code=404,
            detail=f"Advertiser '{advertiser_id}' not found. "
                   f"Valid advertisers: {list(ADVERTISER_CATALOGUE.keys())}"
        )

    category = ADVERTISER_CATALOGUE[advertiser_id]["category"]

    try:
        result = await loop.run_in_executor(
            None, get_advertiser_stats, advertiser_id, category
        )
        logger.info(
            f"[AdvertiserAPI] {advertiser_id} queried stats. "
            f"ε={result['epsilon_used']:.2f}/{result['epsilon_max']}"
        )
        return AdvertiserStatsResponse(**result)
    except RuntimeError as e:
        raise HTTPException(status_code=429, detail=str(e))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_dp_stats() -> Optional[dict]:
    """Return DP stats without crashing if budget is exhausted."""
    try:
        return get_dp_category_stats()
    except RuntimeError as e:
        logger.warning(f"DP budget exhausted: {e}")
        return None


# ---------------------------------------------------------------------------
# Run directly (dev mode)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    from src.config import API_HOST, API_PORT
    uvicorn.run("src.api:app", host=API_HOST, port=API_PORT, reload=True)
