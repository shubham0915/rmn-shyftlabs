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
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

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
# Lifespan: DB init + embedding warmup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[Agent2] Starting up — initialising DB and embedding cache …")
    
    # 1. Init DuckDB (blocking I/O) in executor
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, init_db)
    
    # 2. Init PyTorch on the MAIN THREAD (critical on macOS: loading in executor deadlocks)
    warmup()
    
    # 3. Train/Load XGBoost Model
    train_and_save_model()
    
    # 4. Redis check
    _get_redis()
    
    # 4. Start Event Ingestion Worker
    # Store task reference to avoid garbage collection
    worker_task = asyncio.create_task(_event_worker_loop())
    
    logger.info("[Agent2] Startup complete.")
    yield
    
    worker_task.cancel()
    logger.info("[Agent2] Shutdown.")


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
    copy:        str
    category:    str
    ctr:         float
    price:       float | None = None
    similarity:  float
    final_score: float
    rank:        int | None = None


class AdResponse(BaseModel):
    ad_id:            str
    title:            str
    copy:             str
    category:         str
    ctr:              float
    price:            float | None = None
    similarity:       float
    final_score:      float
    latency_ms:       float
    epsilon_used:     float
    ctr_lift_pct:     float
    top_ads:          list[RankedAd] = []


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
    dp_stats = await loop.run_in_executor(None, lambda: _safe_dp_stats())
    logger.info(f"  -> _safe_dp_stats took {(time.perf_counter() - t_step)*1000:.1f}ms")

    if not ads:
        raise HTTPException(status_code=503, detail="Ad catalogue unavailable")

    logger.info("  -> Starting rank_ads (PyTorch inference)")
    t_step = time.perf_counter()
    # Rank (Running synchronously on Main Thread prevents PyTorch thread deadlocks on macOS)
    ranked = rank_ads(page_text, ads, dp_stats)
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
        copy         = best["copy"],
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
                copy        = a["copy"],
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
