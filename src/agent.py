"""
agent.py — Agentic Personalised Copy Generation
Agent 5: Creative Copywriter (Mock Agentic Layer)

This module simulates an LLM call to Phi-3-mini/Ollama to generate 
dynamic, context-aware ad copy.
"""

def generate_personalized_copy(ad_title: str, ad_desc: str, context: str) -> str:
    """
    Simulates a call to an LLM (e.g. Ollama + Phi-3) to generate 
    a highly personalized 1-liner.
    """
    context_lower = context.lower()

    prefixes = []

    # Price signals
    if "under" in context_lower or "budget" in context_lower or "₹" in context_lower:
        prefixes.append("Perfect budget-friendly choice — ")

    # Season / weather
    if any(w in context_lower for w in ["winter", "cold", "jacket", "sweater"]):
        prefixes.append("Stay warm this winter with ")
    elif any(w in context_lower for w in ["summer", "hot", "lightweight"]):
        prefixes.append("Beat the heat in ")

    # Activity / use-case
    if "running" in context_lower or "training" in context_lower:
        prefixes.append("Level up your runs with ")
    elif "noise" in context_lower or "focus" in context_lower:
        prefixes.append("Block distractions with ")
    elif "wedding" in context_lower or "festive" in context_lower:
        prefixes.append("Shine at every celebration in ")

    # Skin / beauty
    if "spf" in context_lower or "sunscreen" in context_lower or "skin" in context_lower:
        prefixes.append("Protect & glow with ")

    # Default
    if not prefixes:
        prefixes = ["Discover ", "We recommend ", "You'll love "]

    prefix = prefixes[0]  # take the first matching one

    return f"{prefix}{ad_title} — {ad_desc}"
