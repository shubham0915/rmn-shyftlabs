"""
embeddings.py — Sentence-Transformer Embedding Layer & ChromaDB Storage
Agent 3: AI/ML Specialist

Responsibilities:
 - Load all-MiniLM-L6-v2 (CPU-friendly, ~80 MB)
 - Build context strings from page_text + DP-noisy stats
 - Initialize ChromaDB and populate it with ad embeddings
 - Expose get_collection() and encode() for ranking.py
"""

import logging
import numpy as np
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer

from src.config import (
    EMBEDDING_MODEL_NAME,
    CHROMA_DB_DIR,
    SYNTHETIC_ADS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
_model: Optional[SentenceTransformer] = None
_chroma_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[chromadb.Collection] = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME} (CPU) …")
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cpu")
        logger.info("Model loaded.")
    return _model

def get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        global _chroma_client
        logger.info(f"Initializing ChromaDB at {CHROMA_DB_DIR} …")
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        _collection = _chroma_client.get_or_create_collection(
            name="ad_catalog",
            metadata={"hnsw:space": "cosine"} # Enforce Cosine Similarity
        )
    return _collection


# ---------------------------------------------------------------------------
# Context string builder
# ---------------------------------------------------------------------------
def build_context_string(page_text: str, dp_stats: Optional[dict] = None) -> str:
    """
    Combine page text with DP-noisy stats into a single context string.
    The noisy numbers add a weak personalisation signal without exposing raw data.
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
def encode(text: str) -> list[float]:
    """Encode a single string into a unit-normalised vector (Python list for Chroma)."""
    model = get_model()
    # Ensure it's returned as a list of floats (Chroma expects native python types)
    vec = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
    return vec.astype(np.float32).tolist()


# ---------------------------------------------------------------------------
# Warm-up (call once at startup to pre-cache all synthetic ad embeddings into Chroma)
# ---------------------------------------------------------------------------
def warmup() -> None:
    """Initialize model, Chroma DB, and pre-cache all synthetic ads."""
    logger.info("[Agent3] Warming up model and ChromaDB …")
    
    # Force model load and first inference
    get_model()
    encode("warmup")
    
    # Initialize Chroma Collection
    collection = get_collection()
    
    # Check if we already populated the DB
    if collection.count() > 0:
        logger.info(f"[Agent3] ChromaDB already contains {collection.count()} ads. Skipping injection.")
        return
        
    logger.info("[Agent3] Populating ChromaDB with Synthetic Ads...")
    
    ids = []
    embeddings = []
    metadatas = []
    
    for ad in SYNTHETIC_ADS:
        price = ad.get("price")
        price_text = f" price {int(price)}" if price is not None else ""
        text = f"{ad['title']} {ad.get('desc', '')}{price_text}"
        
        vec = encode(text)
        
        ids.append(ad["ad_id"])
        embeddings.append(vec)
        metadatas.append({
            "title": ad["title"],
            "category": ad["category"],
            "ctr": float(ad["ctr"]),
            "price": float(price) if price is not None else 0.0,
            "desc": ad.get("desc", ""),
        })
        
    # Batch add to Chroma
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas
    )
    
    logger.info(f"[Agent3] Successfully cached {len(SYNTHETIC_ADS)} ad embeddings into ChromaDB.")
