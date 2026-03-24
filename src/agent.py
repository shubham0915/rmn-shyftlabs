"""
agent.py — Agentic Personalised Copy Generation
Agent 5: Creative Copywriter (Groq LLM Layer)

This module uses Groq (Llama 3 8B) to generate dynamic, context-aware 
ad copy in real-time (<100ms latency).
"""

import logging
from functools import lru_cache
from typing import Optional

from src.config import GROQ_API_KEY

logger = logging.getLogger(__name__)

# Initialize Groq client only if key is available
try:
    from groq import Groq
    _client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    logger.warning(f"Failed to initialize Groq client: {e}. Will use fallback copy generation.")
    _client = None

# We cache up to 1024 unique (ad, context) combinations in memory 
# to ensure zero latency and zero cost for repeated ad servings.
@lru_cache(maxsize=1024)
def generate_personalized_copy(ad_title: str, ad_desc: str, context: str) -> str:
    """
    Calls Groq's Llama 3 8B model to generate a highly personalized 1-liner ad copy.
    Falls back to a fast rule-based method if the API fails or is unavailable.
    """
    if not _client:
        return _fallback_copy(ad_title, ad_desc, context)

    prompt = f"""
You are an expert e-commerce ad copywriter producing highly personalized ad copy.
Your goal is to connect a specific product to the user's current page context.

Product Title: {ad_title}
Product Description: {ad_desc}
User is currently browsing: {context}

Task: Write EXACTLY ONE short, engaging sentence (under 10 words) that pitches the product to a user with this context. 
Do not include quotes, pleasantries, or anything else. Just the pure ad copy.
    """.strip()

    try:
        completion = _client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a concise ad copywriter. Output exactly one short sentence connecting the product to the user's context. No quotes."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.1-8b-instant",
            temperature=0.7,
            max_tokens=30,
            timeout=1.5 # Strict 1.5s timeout so we never block ad serving for too long
        )
        
        copy = completion.choices[0].message.content.strip().strip('"').strip("'")
        
        # Safety check: if the model hallucinated a long response, fallback to something shorter
        if len(copy.split()) > 15:
            return _fallback_copy(ad_title, ad_desc, context)
            
        return f"✨ {copy}"

    except Exception as e:
        logger.warning(f"[Groq Agent Failed] {e}. Using fallback.")
        return _fallback_copy(ad_title, ad_desc, context)


def _fallback_copy(ad_title: str, ad_desc: str, context: str) -> str:
    """Fast, local fallback if Groq is unavailable."""
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
