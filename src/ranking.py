"""
ranking.py — Semantic Ranking + Ad Selection
Agent 3: AI/ML Specialist

Pipeline:
 1. Compute cosine similarity between context vector and every cached ad vector.
 2. Blend with popularity (CTR) signal.
 3. Return top-K ranked ads + personalised copy (agentic stub ready for Ollama).

Formula:
    final_score[i] = 0.7 * cosine_sim[i]  +  0.3 * normalised_ctr[i]
"""

import logging
import numpy as np
from typing import Optional

from src.config import SIMILARITY_WEIGHT, POPULARITY_WEIGHT, TOP_K
from src.embeddings import encode, get_ad_embeddings, build_context_string

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
    Rank a list of ad dicts by relevance to the current page context.

    Args:
        page_text : Raw text of the current retailer page.
        ads       : List of ad dicts with keys [ad_id, title, category, ctr, desc].
        dp_stats  : DP-noisy aggregate stats from clean_room (optional).

    Returns:
        Sorted list of ads (highest score first) each augmented with:
          - 'similarity'   : raw cosine similarity
          - 'final_score'  : blended score
          - 'copy'         : personalised ad copy string
    """
    if not ads:
        return []

    # 1. Build context vector
    ctx_text = build_context_string(page_text, dp_stats)
    ctx_vec  = encode(ctx_text)                              # (dim,)

    # 2. Get / build ad embedding matrix
    ad_ids, ad_matrix = get_ad_embeddings(ads)               # (N, dim)

    # 3. Cosine similarities — both vectors are already L2-normalised
    similarities = ad_matrix @ ctx_vec                       # (N,)

    # 4. Normalise CTR into [0, 1]
    ctrs = np.array([a["ctr"] for a in ads], dtype=np.float32)
    ctr_norm = (ctrs - ctrs.min()) / (ctrs.max() - ctrs.min() + 1e-9)

    # 5. Blend
    final_scores = SIMILARITY_WEIGHT * similarities + POPULARITY_WEIGHT * ctr_norm  # (N,)

    # 5b. Bonus for budget intent
    budget_bonus = np.zeros(len(ads), dtype=np.float32)
    page_lower = page_text.lower()
    if "under" in page_lower or "budget" in page_lower:
        prices = np.array([a.get("price", 9999) for a in ads], dtype=np.float32)
        budget_bonus = np.where(prices < 3000, 0.15, 0.0).astype(np.float32)

    final_scores = final_scores + budget_bonus

    # 6. Sort descending
    order = np.argsort(final_scores)[::-1]

    # 7. Build annotated output
    ranked = []
    for rank, idx in enumerate(order[:TOP_K]):
        ad = dict(ads[idx])
        ad["similarity"]  = float(similarities[idx])
        ad["final_score"] = float(final_scores[idx])
        ad["rank"]        = rank + 1
        ad["copy"]        = generate_copy(ad, page_text)
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
    if not ads:
        return []
    ctx_vec = encode(page_text)
    _, ad_matrix = get_ad_embeddings(ads)
    return (ad_matrix @ ctx_vec).tolist()
