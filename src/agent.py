"""
agent.py — Agentic Personalised Copy Generation
Agent 5: Creative Copywriter (Gemini LLM Layer)

This module uses Google Gemini (Gemini 1.5 Flash) via the new `google-genai` SDK
to generate dynamic, context-aware ad copy in real-time.
"""

import logging
from functools import lru_cache
from typing import Optional

from src.config import GEMINI_API_KEY

logger = logging.getLogger(__name__)

# Initialize Gemini client only if key is available
# Uses the new `google-genai` SDK (google.generativeai is deprecated)
_client = None
try:
    from google import genai
    from google.genai import types as genai_types
    if GEMINI_API_KEY:
        _client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info("[Agent5] Gemini client initialized via google-genai SDK.")
    else:
        logger.warning("[Agent5] GEMINI_API_KEY not set. Will use fallback copy generation.")
except Exception as e:
    logger.warning(f"Failed to initialize Gemini client: {e}. Will use fallback copy generation.")
    _client = None

_SYSTEM_INSTRUCTION = (
    "You are a concise ad copywriter. "
    "Output exactly one short sentence connecting the product to the user's context. "
    "No quotes, no extra words."
)

# We cache up to 1024 unique (ad, context) combinations in memory
# to ensure zero latency and zero cost for repeated ad servings.
@lru_cache(maxsize=1024)
def generate_personalized_copy(ad_title: str, ad_desc: str, context: str) -> str:
    """
    Calls Gemini 1.5 Flash to generate a highly personalized 1-liner ad copy.
    Falls back to a fast rule-based method if the API fails or is unavailable.
    """
    if not _client:
        return _fallback_copy(ad_title, ad_desc, context)

    prompt = (
        f"Product Title: {ad_title}\n"
        f"Product Description: {ad_desc}\n"
        f"User is currently browsing: {context}\n\n"
        f"Task: Write EXACTLY ONE short, engaging sentence (under 10 words) that pitches "
        f"the product to a user with this context. "
        f"Do not include quotes, pleasantries, or anything else. Just the pure ad copy."
    )

    try:
        response = _client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=_SYSTEM_INSTRUCTION,
                temperature=0.7,
                max_output_tokens=30,
            ),
        )

        ad_copy = response.text.strip().strip('"').strip("'")

        # Safety check: if the model returned something too long, use fallback
        if len(ad_copy.split()) > 15:
            return _fallback_copy(ad_title, ad_desc, context)

        return f"✨ {ad_copy}"

    except Exception as e:
        logger.warning(f"[Gemini Agent Failed] {e}. Using fallback.")
        return _fallback_copy(ad_title, ad_desc, context)


def _fallback_copy(ad_title: str, ad_desc: str, context: str) -> str:
    """Fast, local fallback if Gemini is unavailable."""
    context_lower = context.lower()
    prefixes = []

    if "under" in context_lower or "budget" in context_lower or "₹" in context_lower:
        prefixes.append("Perfect budget-friendly choice — ")
    if any(w in context_lower for w in ["winter", "cold", "jacket", "sweater"]):
        prefixes.append("Stay warm this winter with ")
    elif any(w in context_lower for w in ["summer", "hot", "lightweight"]):
        prefixes.append("Beat the heat in ")
    if "running" in context_lower or "training" in context_lower:
        prefixes.append("Level up your runs with ")
    elif "noise" in context_lower or "focus" in context_lower:
        prefixes.append("Block distractions with ")
    elif "wedding" in context_lower or "festive" in context_lower:
        prefixes.append("Shine at every celebration in ")
    if "spf" in context_lower or "sunscreen" in context_lower or "skin" in context_lower:
        prefixes.append("Protect & glow with ")

    if not prefixes:
        prefixes = ["Discover ", "We recommend ", "You'll love "]

    prefix = prefixes[0]
    return f"{prefix}{ad_title} — {ad_desc}"
