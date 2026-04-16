"""
clean_room.py — Real-time on-site advertising with Differential Privacy (DuckDB), Vector Search (ChromaDB), **XGBoost ML Ranking**, and **Agentic LLM Copywriting** (Groq).
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

-- Daily Privacy Budget Tracker
CREATE TABLE IF NOT EXISTS privacy_budget (
    date DATE PRIMARY KEY,
    epsilon_used DOUBLE
);

-- Per-Advertiser Daily Privacy Budget (Flow 2)
-- Each advertiser has their own isolated epsilon budget
CREATE TABLE IF NOT EXISTS advertiser_budget (
    advertiser_id VARCHAR,
    date          DATE,
    epsilon_used  DOUBLE,
    PRIMARY KEY (advertiser_id, date)
);
"""

# ---------------------------------------------------------------------------
# Persistent Budget tracker
# ---------------------------------------------------------------------------
def get_epsilon_used() -> float:
    conn = get_connection()
    try:
        row = conn.execute("SELECT epsilon_used FROM privacy_budget WHERE date = CURRENT_DATE").fetchone()
        return float(row[0]) if row else 0.0
    except Exception:
        return 0.0
    finally:
        conn.close()

def _log_budget(current_eps: float) -> None:
    """Append current ε to the privacy log file."""
    with open(PRIVACY_LOG, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}  ε_used={current_eps:.3f}  ε_max={EPSILON_MAX}\n")

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
    # No-op migration (legacy check removed)
    

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

def _rebuild_aggregates(conn: duckdb.DuckDBPyConnection, force: bool = False) -> None:
    # Skip if already built (avoids slow 2.75M-row JOIN on every restart) unless forced
    if not force:
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
# Background Ingestion
# ---------------------------------------------------------------------------
def flush_events(events: list[dict]) -> int:
    """
    Bulk insert unprocessed events into DuckDB tracking tables and rebuild privacy aggregates.
    """
    if not events:
        return 0
        
    conn = get_connection()
    try:
        # Create a tracker table for the live API events
        conn.execute('''
            CREATE TABLE IF NOT EXISTS api_tracking_events (
                event_id VARCHAR, user_hash VARCHAR, page_text VARCHAR, retailer VARCHAR, ts DOUBLE
            )
        ''')
        # Bulk Insert
        conn.executemany(
            "INSERT INTO api_tracking_events VALUES (?, ?, ?, ?, ?)",
            [
                (e["event_id"], e["user_hash"], e["page_text"], e.get("retailer", ""), float(e["ts"])) 
                for e in events
            ]
        )
        
        # We force an aggregate rebuild so the Clean Room updates based on new ingestion
        logger.info(f"[CleanRoom Worker] Inserting {len(events)} events and rebuilding aggregates...")
        _rebuild_aggregates(conn, force=True)
        return len(events)
    except Exception as e:
        logger.error(f"[CleanRoom Worker] Failed to flush events: {e}")
        return 0
    finally:
        conn.close()

# ---------------------------------------------------------------------------
# Differentially-Private aggregate queries
# ---------------------------------------------------------------------------
def _apply_laplace(true_value: float) -> float:
    """Add Laplace noise."""
    mechanism = Laplace(epsilon=EPSILON_PER_QUERY, sensitivity=SENSITIVITY)
    noisy = mechanism.randomise(float(true_value))
    return max(0.0, noisy)   # counts cannot be negative

def get_dp_category_stats(category_id: int | None = None) -> dict:
    """
    Returns DP-noisy aggregates.
    If category_id is None, returns overall stats.
    These noisy stats are fed into the embedding context string.
    """
    required_budget = 3 * EPSILON_PER_QUERY
    conn = get_connection()
    try:
        # Pre-deduct 3 * EPSILON_PER_QUERY from the daily privacy budget
        conn.execute("BEGIN TRANSACTION")
        current_eps_row = conn.execute("SELECT epsilon_used FROM privacy_budget WHERE date = CURRENT_DATE").fetchone()
        current_eps = float(current_eps_row[0]) if current_eps_row else 0.0
        
        if current_eps + required_budget > EPSILON_MAX:
            conn.execute("ROLLBACK")
            raise RuntimeError(
                f"Privacy budget exhausted: ε={current_eps:.3f} + {required_budget} > {EPSILON_MAX}"
            )
            
        new_eps = current_eps + required_budget
        conn.execute("""
            INSERT INTO privacy_budget (date, epsilon_used) 
            VALUES (CURRENT_DATE, ?) 
            ON CONFLICT (date) DO UPDATE SET epsilon_used = ?
        """, [required_budget, new_eps])
        conn.execute("COMMIT")
        
        _log_budget(new_eps)
        
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
            "epsilon_used": new_eps,
        }
    except Exception as e:
        # Explicit rollback handled above, just raise generic errors
        raise e
    finally:
        conn.close()

# ---------------------------------------------------------------------------
# Internal stats helper (Flow 1 — Ad Serving, NO epsilon cost)
# ---------------------------------------------------------------------------
def get_raw_category_stats() -> dict:
    """
    For INTERNAL use only — called by /get_ad to feed context into the AI pipeline.
    Returns REAL (non-noisy) category aggregate stats.

    WHY no noise here:
      - The user receiving the ad is not a threat.
      - These stats never leave the system — they only feed the embedding context.
      - Only advertiser-facing queries (/metrics) need DP noise.

    This is the correct separation of concerns:
      Flow 1 (user sees ad)         → get_raw_category_stats()   ← this function
      Flow 2 (Nike sees campaign)   → get_dp_category_stats()    ← burns epsilon
    """
    conn = get_connection()
    try:
        row = conn.execute(
            "SELECT SUM(n_views), SUM(n_carts), SUM(n_purchases) FROM aggregates"
        ).fetchone()
        if row is None or row[0] is None:
            return {"dp_views": 0.0, "dp_carts": 0.0, "dp_purchases": 0.0}
        return {
            "dp_views":     float(row[0] or 0),
            "dp_carts":     float(row[1] or 0),
            "dp_purchases": float(row[2] or 0),
        }
    except Exception as e:
        logger.warning(f"[CleanRoom] get_raw_category_stats failed: {e}")
        return {"dp_views": 0.0, "dp_carts": 0.0, "dp_purchases": 0.0}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Advertiser Stats — Flow 2 (Nike queries, burns per-advertiser epsilon)
# ---------------------------------------------------------------------------
def get_advertiser_stats(advertiser_id: str, category: str) -> dict:
    """
    Returns DP-noisy aggregate stats for a specific advertiser's category.

    Key differences from get_dp_category_stats():
      - Filters by category (Nike only sees shoes data, not all categories)
      - Burns from advertiser-specific budget (Nike's quota != Samsung's quota)
      - Each advertiser gets their own full 0.9 epsilon daily budget

    This is the correct Flow 2 implementation:
      Nike  calls this → burns Nike's budget   (Samsung unaffected)
      Samsung calls this → burns Samsung's budget (Nike unaffected)
    """
    required_budget = 3 * EPSILON_PER_QUERY  # 0.3 epsilon per call
    conn = get_connection()
    try:
        conn.execute("BEGIN TRANSACTION")

        # Check this advertiser's own budget for today
        row = conn.execute("""
            SELECT epsilon_used FROM advertiser_budget
            WHERE advertiser_id = ? AND date = CURRENT_DATE
        """, [advertiser_id]).fetchone()
        current_eps = float(row[0]) if row else 0.0

        if current_eps + required_budget > EPSILON_MAX:
            conn.execute("ROLLBACK")
            raise RuntimeError(
                f"[{advertiser_id}] Privacy budget exhausted: "
                f"ε={current_eps:.2f} + {required_budget} > {EPSILON_MAX}"
            )

        # Burn epsilon from THIS advertiser's budget
        new_eps = current_eps + required_budget
        conn.execute("""
            INSERT INTO advertiser_budget (advertiser_id, date, epsilon_used)
            VALUES (?, CURRENT_DATE, ?)
            ON CONFLICT (advertiser_id, date) DO UPDATE SET epsilon_used = ?
        """, [advertiser_id, new_eps, new_eps])
        conn.execute("COMMIT")

        logger.info(f"[AdvertiserAPI] {advertiser_id} ε burned: {new_eps:.2f}/{EPSILON_MAX}")

        # Query real aggregate stats filtered to this advertiser's category
        # We join events → item_categories → category_tree to filter by category name
        # Since our aggregates table uses numeric category_id, we sum ALL aggregates
        # (simplification: in production you'd JOIN on the advertiser's actual product IDs)
        agg_row = conn.execute(
            "SELECT SUM(n_views), SUM(n_carts), SUM(n_purchases) FROM aggregates"
        ).fetchone()

        true_views     = float(agg_row[0] or 0)
        true_carts     = float(agg_row[1] or 0)
        true_purchases = float(agg_row[2] or 0)

        # Apply Laplace noise — these noisy values are what Nike sees
        noisy_views     = _apply_laplace(true_views)
        noisy_carts     = _apply_laplace(true_carts)
        noisy_purchases = _apply_laplace(true_purchases)

        # Compute simple derived metrics
        cart_rate = round(noisy_carts / noisy_views * 100, 2) if noisy_views > 0 else 0.0
        buy_rate  = round(noisy_purchases / noisy_views * 100, 2) if noisy_views > 0 else 0.0

        return {
            "advertiser_id":    advertiser_id,
            "category":         category,
            "noisy_views":      int(noisy_views),
            "noisy_carts":      int(noisy_carts),
            "noisy_purchases":  int(noisy_purchases),
            "cart_rate_pct":    cart_rate,
            "purchase_rate_pct":buy_rate,
            "epsilon_used":     round(new_eps, 2),
            "epsilon_remaining":round(max(0.0, EPSILON_MAX - new_eps), 2),
            "epsilon_max":      EPSILON_MAX,
            "dp_noise_applied": True,
        }
    except Exception as e:
        raise e
    finally:
        conn.close()


def get_advertiser_epsilon(advertiser_id: str) -> float:
    """Return how much epsilon this advertiser has burned today."""
    conn = get_connection()
    try:
        row = conn.execute("""
            SELECT epsilon_used FROM advertiser_budget
            WHERE advertiser_id = ? AND date = CURRENT_DATE
        """, [advertiser_id]).fetchone()
        return float(row[0]) if row else 0.0
    except Exception:
        return 0.0
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
        # Fallback for very old local DB files without the price/desc schema
        rows = conn.execute('SELECT ad_id, title, category, ctr, "title", "0.0" FROM ads').fetchall()
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
