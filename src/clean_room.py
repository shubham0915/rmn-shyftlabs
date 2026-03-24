"""
clean_room.py — DuckDB Data Clean Room + Differential Privacy Layer
Agent 4: Privacy & Integration Architect

This module:
 1. Initialises a DuckDB database (file-backed for speed, ~2.7M events).
 2. Provides aggregate queries that are always made DP-safe via Laplace noise.
 3. Tracks the running ε (epsilon) privacy budget and hard-stops at ε_max.
 4. Offers a helper to derive noisy category stats used by the ML ranking layer.
"""

import os
import time
import logging
from pathlib import Path

import duckdb
import numpy as np
from diffprivlib.mechanisms import Laplace

from src.config import (
    DATA_DIR, DB_PATH, PRIVACY_LOG,
    EPSILON_PER_QUERY, EPSILON_MAX, SENSITIVITY, SYNTHETIC_ADS
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema SQL
# ---------------------------------------------------------------------------
_SCHEMA_SQL = """
-- Raw e-commerce event log (Retailrocket format)
-- Columns from events.csv: timestamp, visitorid, event, itemid, transactionid
CREATE TABLE IF NOT EXISTS events (
    timestamp      BIGINT,
    visitor_id     VARCHAR,
    event_type     VARCHAR,   -- 'view' | 'addtocart' | 'transaction'
    item_id        BIGINT,
    transaction_id VARCHAR
);

-- Product properties (from item_properties.csv)
-- Columns: timestamp, itemid, property, value
-- When property='categoryid', value holds the category
CREATE TABLE IF NOT EXISTS items (
    timestamp   BIGINT,
    item_id     BIGINT,
    property    VARCHAR,
    value       VARCHAR
);

-- Item-to-category lookup (derived from items where property='categoryid')
CREATE TABLE IF NOT EXISTS item_categories (
    item_id     BIGINT,
    category_id BIGINT
);

-- Pre-aggregated stats table (updated from raw events)
CREATE TABLE IF NOT EXISTS aggregates (
    category_id  BIGINT,
    n_views      BIGINT,
    n_carts      BIGINT,
    n_purchases  BIGINT
);

-- Synthetic ad catalogue (always present, acts as fallback)
CREATE TABLE IF NOT EXISTS ads (
    ad_id     VARCHAR PRIMARY KEY,
    title     VARCHAR,
    category  VARCHAR,
    ctr       DOUBLE,
    "desc"    VARCHAR,
    price     DOUBLE
);
"""

# ---------------------------------------------------------------------------
# Budget tracker (module-level singleton)
# ---------------------------------------------------------------------------
_epsilon_used: float = 0.0

def get_epsilon_used() -> float:
    return _epsilon_used

def reset_epsilon() -> None:
    global _epsilon_used
    _epsilon_used = 0.0
    _log_budget()

def _log_budget() -> None:
    """Append current ε to the privacy log file."""
    with open(PRIVACY_LOG, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}  ε_used={_epsilon_used:.3f}  ε_max={EPSILON_MAX}\n")

# ---------------------------------------------------------------------------
# DB connection (per-thread/process – DuckDB handles concurrent reads fine)
# ---------------------------------------------------------------------------
def get_connection() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(DB_PATH)

# ---------------------------------------------------------------------------
# Initialise schema + seed data
# ---------------------------------------------------------------------------
def init_db(force_reload: bool = False) -> None:
    """
    Create tables and load CSVs if they exist.
    Safe to call multiple times (idempotent).
    """
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    conn = get_connection()
    conn.execute(_SCHEMA_SQL)
    # Backward-compatible migration for existing DB files created before `price` / `desc` existed
    conn.execute("ALTER TABLE ads ADD COLUMN IF NOT EXISTS price DOUBLE")
    conn.execute('ALTER TABLE ads ADD COLUMN IF NOT EXISTS "desc" VARCHAR')
    # Migrate legacy `descr` -> `desc` if old column exists
    try:
        conn.execute('UPDATE ads SET "desc" = descr WHERE "desc" IS NULL')
    except Exception:
        pass

    # ---- Load Retailrocket events.csv if present ----
    events_csv = os.path.join(DATA_DIR, "events.csv")
    if os.path.exists(events_csv):
        n = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        if n == 0 or force_reload:
            logger.info("Loading events.csv into DuckDB (~2.7M rows, may take 10s)…")
            conn.execute(f"""
                INSERT INTO events
                SELECT
                    CAST("timestamp" AS BIGINT),
                    CAST(visitorid   AS VARCHAR),
                    event,
                    CAST(itemid      AS BIGINT),
                    transactionid
                FROM read_csv_auto('{events_csv}', header=True,
                                   nullstr='', ignore_errors=True)
            """)
            logger.info(f"Loaded {conn.execute('SELECT COUNT(*) FROM events').fetchone()[0]:,} events.")

    # ---- Load item_properties.csv if present ----
    items_csv = os.path.join(DATA_DIR, "item_properties.csv")
    if os.path.exists(items_csv):
        n = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
        if n == 0 or force_reload:
            logger.info("Loading item_properties.csv …")
            conn.execute(f"""
                INSERT INTO items
                SELECT
                    CAST("timestamp" AS BIGINT),
                    CAST(itemid      AS BIGINT),
                    property,
                    value
                FROM read_csv_auto('{items_csv}', header=True, ignore_errors=True)
            """)

    # ---- Build item → category lookup ----
    n_cat = conn.execute("SELECT COUNT(*) FROM item_categories").fetchone()[0]
    if n_cat == 0 or force_reload:
        logger.info("Building item_categories lookup …")
        conn.execute("DELETE FROM item_categories")
        conn.execute("""
            INSERT INTO item_categories
            SELECT CAST(item_id AS BIGINT), CAST(value AS BIGINT)
            FROM items
            WHERE property = 'categoryid'
        """)

    # ---- Always refresh aggregates ----
    _rebuild_aggregates(conn)

    # ---- Seed synthetic ad catalogue ----
    _seed_ads(conn)

    conn.close()
    logger.info("DuckDB Clean Room initialised.")

def _rebuild_aggregates(conn: duckdb.DuckDBPyConnection) -> None:
    # Skip if already built (avoids slow 2.75M-row JOIN on every restart)
    n_existing = conn.execute("SELECT COUNT(*) FROM aggregates").fetchone()[0]
    if n_existing > 0:
        logger.info(f"Aggregates already built ({n_existing:,} rows) — skipping rebuild.")
        return

    conn.execute("DELETE FROM aggregates")
    n_events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    n_cats   = conn.execute("SELECT COUNT(*) FROM item_categories").fetchone()[0]
    if n_events > 0 and n_cats > 0:
        logger.info("Building category aggregates …")
        conn.execute("""
            INSERT INTO aggregates
            SELECT
                ic.category_id,
                COUNT(*) FILTER (WHERE e.event_type = 'view')        AS n_views,
                COUNT(*) FILTER (WHERE e.event_type = 'addtocart')   AS n_carts,
                COUNT(*) FILTER (WHERE e.event_type = 'transaction') AS n_purchases
            FROM events e
            JOIN item_categories ic ON e.item_id = ic.item_id
            GROUP BY ic.category_id
        """)
        n_agg = conn.execute("SELECT COUNT(*) FROM aggregates").fetchone()[0]
        logger.info(f"Built {n_agg:,} category aggregates from {n_events:,} events.")

def _seed_ads(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("DELETE FROM ads")
    for ad in SYNTHETIC_ADS:
        conn.execute(
            "INSERT INTO ads (ad_id, title, category, ctr, \"desc\", price) VALUES (?, ?, ?, ?, ?, ?)",
            [ad["ad_id"], ad["title"], ad["category"], ad["ctr"], ad["desc"], ad.get("price")]
        )

# ---------------------------------------------------------------------------
# Differentially-Private aggregate queries
# ---------------------------------------------------------------------------
def _apply_laplace(true_value: float) -> float:
    """Add Laplace noise and deduct from ε budget."""
    global _epsilon_used
    if _epsilon_used + EPSILON_PER_QUERY > EPSILON_MAX:
        raise RuntimeError(
            f"Privacy budget exhausted: ε={_epsilon_used:.3f} + {EPSILON_PER_QUERY} > {EPSILON_MAX}"
        )
    mechanism = Laplace(epsilon=EPSILON_PER_QUERY, sensitivity=SENSITIVITY)
    noisy = mechanism.randomise(float(true_value))
    _epsilon_used += EPSILON_PER_QUERY
    _log_budget()
    return max(0.0, noisy)   # counts cannot be negative

def get_dp_category_stats(category_id: int | None = None) -> dict:
    """
    Returns DP-noisy aggregates.
    If category_id is None, returns overall stats.
    These noisy stats are fed into the embedding context string.
    """
    conn = get_connection()
    try:
        if category_id is not None:
            row = conn.execute(
                "SELECT n_views, n_carts, n_purchases FROM aggregates WHERE category_id = ?",
                [category_id]
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT SUM(n_views), SUM(n_carts), SUM(n_purchases) FROM aggregates"
            ).fetchone()

        if row is None:
            row = (0, 0, 0)

        return {
            "dp_views":     _apply_laplace(row[0] or 0),
            "dp_carts":     _apply_laplace(row[1] or 0),
            "dp_purchases": _apply_laplace(row[2] or 0),
            "epsilon_used": _epsilon_used,
        }
    finally:
        conn.close()

# ---------------------------------------------------------------------------
# Ad catalogue helpers (no DP needed – public catalogue data)
# ---------------------------------------------------------------------------
def get_all_ads() -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute('SELECT ad_id, title, category, ctr, "desc", price FROM ads').fetchall()
    except Exception:
        rows = conn.execute("SELECT ad_id, title, category, ctr, descr, price FROM ads").fetchall()
    conn.close()
    return [
        {"ad_id": r[0], "title": r[1], "category": r[2], "ctr": r[3], "desc": r[4], "price": r[5]}
        for r in rows
    ]

# ---------------------------------------------------------------------------
# CTR lift simulation vs random baseline
# ---------------------------------------------------------------------------
def simulate_ctr_lift(semantic_scores: list[float], random_n: int = 1000) -> dict:
    """
    Simulate CTR improvement of semantic ranking vs a random baseline.
    Both baselines are sampled over the synthetic ad catalogue CTRs.
    """
    ads = get_all_ads()
    ctrs = np.array([a["ctr"] for a in ads])

    # Random policy: expected CTR is mean of all ads
    random_ctr = float(np.mean(ctrs))

    # Semantic policy: weighted average CTR (weights = softmax of scores)
    if semantic_scores:
        weights = np.exp(semantic_scores) / np.sum(np.exp(semantic_scores))
        semantic_ctr = float(np.dot(weights, ctrs[:len(weights)]))
    else:
        semantic_ctr = random_ctr

    lift_pct = max(0.0, (semantic_ctr - random_ctr) / random_ctr * 100)
    return {
        "random_ctr":   round(random_ctr,   4),
        "semantic_ctr": round(semantic_ctr, 4),
        "lift_pct":     round(lift_pct,     1),
    }

# ---------------------------------------------------------------------------
# Integration test entry-point (called by Agent 4's test suite)
# ---------------------------------------------------------------------------
def integration_test() -> bool:
    """Quick smoke-test of the Clean Room. Returns True if all checks pass."""
    try:
        init_db()
        ads = get_all_ads()
        assert len(ads) == len(SYNTHETIC_ADS), f"Expected {len(SYNTHETIC_ADS)} ads, got {len(ads)}"
        stats = get_dp_category_stats()
        assert "dp_views" in stats
        logger.info(f"[Agent4 Integration] Clean Room OK. ε={stats['epsilon_used']:.2f}")
        return True
    except Exception as e:
        logger.error(f"[Agent4 Integration] Clean Room FAILED: {e}")
        return False
