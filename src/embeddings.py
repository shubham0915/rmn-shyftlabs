"""
embeddings.py — Sentence-Transformer Embedding Layer
Agent 3: AI/ML Specialist

Responsibilities:
 - Load all-MiniLM-L6-v2 (CPU-friendly, ~80 MB)
 - Build context strings from page_text + DP-noisy stats
 - Cache ad embeddings in Redis (avoids re-computing on every request)
 - Expose encode() and get_ad_embeddings() for ranking.py
"""

import json
import logging
import numpy as np
from functools import lru_cache
from typing import Optional

from sentence_transformers import SentenceTransformer

from src.config import (
    EMBEDDING_MODEL_NAME,
    REDIS_HOST, REDIS_PORT, REDIS_DB, EMBED_CACHE,
    SYNTHETIC_ADS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model (lazy singleton — loads once, stays in memory)
# ---------------------------------------------------------------------------
_model: Optional[SentenceTransformer] = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME} (CPU) …")
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
        logger.info("Model loaded.")
    return _model


# ---------------------------------------------------------------------------
# In-memory helper (fast fallback since this is a 0-budget laptop demo)
# ---------------------------------------------------------------------------

_mem_cache: dict[str, list] = {}   # fallback in-memory store

def _cache_get(key: str) -> Optional[np.ndarray]:
    return _mem_cache.get(key)

def _cache_set(key: str, vec: np.ndarray) -> None:
    serialised = json.dumps(vec.tolist())
    _mem_cache[key] = np.array(json.loads(serialised), dtype=np.float32)


# ---------------------------------------------------------------------------
# Context string builder
# ---------------------------------------------------------------------------
def build_context_string(page_text: str, dp_stats: Optional[dict] = None) -> str:
    """
    Combine page text with DP-noisy stats into a single context string.
    The noisy numbers add a weak personalisation signal without exposing raw data.

    Args:
        page_text : Free-text description of the current retailer page.
        dp_stats  : Dict from clean_room.get_dp_category_stats() — may be None.

    Returns:
        A rich context string fed to the embedding model.
    """
    parts = [page_text.strip()]
    if dp_stats:
        parts.append(
            f"category views ~{int(dp_stats.get('dp_views', 0))} "
            f"carts ~{int(dp_stats.get('dp_carts', 0))} "
            f"purchases ~{int(dp_stats.get('dp_purchases', 0))}"
        )
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------
def encode(text: str) -> np.ndarray:
    """Encode a single string into a unit-normalised vector."""
    model = get_model()
    vec = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
    return vec.astype(np.float32)


def get_ad_embeddings(ads: list[dict]) -> tuple[list[str], np.ndarray]:
    """
    Return cached embeddings for each ad in `ads`.
    Cache key: EMBED_CACHE + ad_id

    Returns:
        ad_ids  : list of ad_id strings in the same order
        matrix  : ndarray of shape (N, embedding_dim)
    """
    ad_ids  = []
    vectors = []

    for ad in ads:
        key = EMBED_CACHE + ad["ad_id"]
        cached = _cache_get(key)
        if cached is not None:
            vectors.append(cached)
        else:
            # Embed: title + description combined
            price = ad.get("price")
            price_text = f" price {int(price)}" if price is not None else ""
            text = f"{ad['title']} {ad.get('desc', '')}{price_text}"
            vec  = encode(text)
            _cache_set(key, vec)
            vectors.append(vec)
        ad_ids.append(ad["ad_id"])

    matrix = np.vstack(vectors).astype(np.float32)   # (N, dim)
    return ad_ids, matrix


# ---------------------------------------------------------------------------
# Warm-up (call once at startup to pre-cache all synthetic ad embeddings)
# ---------------------------------------------------------------------------
def warmup() -> None:
    """Pre-cache all synthetic ad embeddings into Redis / mem_cache."""
    logger.info("[Agent3] Warming up ad embedding cache …")
    get_model()   # Force model to load into memory NOW, not on first user request
    encode("warmup") # Force the first physically heavy PyTorch inference to occur on the main thread
    get_ad_embeddings(SYNTHETIC_ADS)
    logger.info(f"[Agent3] Cached {len(SYNTHETIC_ADS)} ad embeddings.")
