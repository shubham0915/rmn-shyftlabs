"""
ranking.py — Semantic Ranking + Ad Selection
Agent 3: AI/ML Specialist

Pipeline:
 1. Query ChromaDB for the Top N most semantically similar ads to the context. 
 2. Rerank using XGBoost Learning-to-Rank probability models.
 3. Return top-K ranked ads + personalised copy.
"""

import logging
import numpy as np
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from src.config import TOP_K
from src.embeddings import encode, get_collection, build_context_string
from src.ltr import score_candidates

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core ranking
# ---------------------------------------------------------------------------
def rank_ads(
    page_text: str,
    ads: list[dict],
    dp_stats: Optional[dict] = None,
) -> list[dict]:
    """
    Rank ads using ChromaDB for semantic retrieval, followed by XGBoost LTR reranking.
    """
    collection = get_collection()
    if collection.count() == 0:
        return []

    # 1. Build context vector
    ctx_text = build_context_string(page_text, dp_stats)
    ctx_vec  = encode(ctx_text)

    # 2. Retrieve Top-N from ChromaDB
    results = collection.query(
        query_embeddings=[ctx_vec],
        n_results=10,
        include=["metadatas", "distances"]
    )

    if not results["ids"] or len(results["ids"][0]) == 0:
        return []

    retrieved_ids = results["ids"][0]
    distances     = results["distances"][0]
    metadatas     = results["metadatas"][0]

    # 3. Extract features for XGBoost
    page_lower = page_text.lower()
    budget_intent = 1.0 if ("under" in page_lower or "budget" in page_lower) else 0.0

    prices = np.array([meta.get("price", 9999) for meta in metadatas], dtype=np.float32)
    price_min, price_max = prices.min(), prices.max()
    price_denom = (price_max - price_min) if (price_max - price_min) > 0 else 1e-9

    features_matrix = []
    candidates = []

    for idx, (ad_id, dist, meta) in enumerate(zip(retrieved_ids, distances, metadatas)):
        similarity = 1.0 - float(dist)
        norm_price = (meta.get("price", 9999) - price_min) / price_denom
        norm_ctr   = meta.get("ctr", 0.0)
        
        # [similarity, historical_ctr, price_normalized, budget_intent]
        features_matrix.append([
            similarity,
            float(norm_ctr),
            float(norm_price),
            budget_intent
        ])
        
        candidates.append({
            "ad_id": ad_id,
            "title": meta["title"],
            "desc": meta.get("desc", ""),
            "category": meta["category"],
            "ctr": meta["ctr"],
            "price": meta.get("price"),
            "similarity": similarity,
        })

    # 4. ML Inference
    predictions = score_candidates(features_matrix)

    # 5. Append scores
    for i, ad in enumerate(candidates):
        ad["final_score"] = float(predictions[i])

    # 6. Sort descending by our ML predicted probability
    candidates.sort(key=lambda x: x["final_score"], reverse=True)

    # 7. Finalize output and add copy formatting
    ranked = []
    top_candidates = candidates[:TOP_K]

    # Parallelize LLM copy generation using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=TOP_K) as executor:
        # Submit all copy generation tasks
        future_to_ad = {
            executor.submit(generate_copy, ad, page_text): ad 
            for ad in top_candidates
        }
        
        for i, ad in enumerate(top_candidates):
            ad["rank"] = i + 1
            # We'll wait for the futures in the same order as the ads
            # But the actual API calls happen concurrently
            future = list(future_to_ad.keys())[i] 
            ad["copy"] = future.result()
            ranked.append(ad)

    return ranked


# ---------------------------------------------------------------------------
# Personalised copy generation
# ---------------------------------------------------------------------------
from src.agent import generate_personalized_copy

def generate_copy(ad: dict, page_text: str) -> str:
    """
    Generate a 1-line personalised copy string using the Agentic Copy layer.
    """
    return generate_personalized_copy(ad["title"], ad.get("desc", ""), page_text)


# ---------------------------------------------------------------------------
# Scores list helper (used by CTR lift simulation in clean_room.py)
# ---------------------------------------------------------------------------
def get_similarity_scores(page_text: str, ads: list[dict]) -> list[float]:
    """Return just the similarity scores (no ranking) — used for CTR simulation."""
    collection = get_collection()
    if collection.count() == 0:
        return []
    
    ctx_vec = encode(page_text)
    
    total_ads = collection.count()
    results = collection.query(
        query_embeddings=[ctx_vec],
        n_results=total_ads,
        include=["distances"]
    )
    
    distances = results["distances"][0]
    return [1.0 - float(d) for d in distances]
