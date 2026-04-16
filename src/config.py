"""
config.py — Central configuration for the RMN Engine.
Agent 4 owns this file. All constants go here.
"""
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH  = os.path.join(DATA_DIR, "clean_room.duckdb")
PRIVACY_LOG = os.path.join(BASE_DIR, "privacy_budget.log")

# ---------------------------------------------------------------------------
# Differential Privacy
# ---------------------------------------------------------------------------
EPSILON_PER_QUERY = 0.1   # Each DP query costs 0.1 ε
EPSILON_MAX       = 0.9   # Hard budget cap
SENSITIVITY       = 1.0   # L1 sensitivity for aggregates

# ---------------------------------------------------------------------------
# Embedding / Ranking
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
SIMILARITY_WEIGHT    = 0.7
POPULARITY_WEIGHT    = 0.3
TOP_K                = 5

# ---------------------------------------------------------------------------
# Redis
# ---------------------------------------------------------------------------
REDIS_HOST    = "127.0.0.1"
REDIS_PORT    = 6379
REDIS_DB      = 0
STREAM_KEY    = "rmn:events"
EMBED_CACHE   = "rmn:embeddings:"   # prefix, append ad_id

# ---------------------------------------------------------------------------
# App Config
# ---------------------------------------------------------------------------
API_HOST = "0.0.0.0"
API_PORT = 8000
CHROMA_DB_DIR = "./data/chroma"

# ---------------------------------------------------------------------------
# Gemini API
# ---------------------------------------------------------------------------
import os
from dotenv import load_dotenv

load_dotenv() # Load from .env file if it exists locally

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# ---------------------------------------------------------------------------
# Demo Retailer pages (used by Streamlit)
# ---------------------------------------------------------------------------
RETAILER_PAGES = {
    "Myntra – Summer Shoes": "summer running shoes lightweight under 2000 breathable",
    "Myntra – Electronics":  "wireless earbuds noise cancelling battery 30hr",
    "Myntra – Men's Ethnic": "kurta men wedding festive cotton embroidered",
    "Myntra – Skincare":     "moisturizer SPF sunscreen dermatologist tested",
}

# ---------------------------------------------------------------------------
# Synthetic Ad Catalogue (used when DuckDB is empty / items not loaded)
# ---------------------------------------------------------------------------
SYNTHETIC_ADS = [
    {
        "ad_id": "A1",
        "title": "CloudStep Pro Running Shoes",
        "category": "shoes",
        "ctr": 0.08,
        "price": 1999,
        "desc": "Ultra-light for your summer runs"
    },
    {
        "ad_id": "A2",
        "title": "AirBeat X3 Wireless Earbuds",
        "category": "electronics",
        "ctr": 0.12,
        "price": 2499,
        "desc": "40hr battery, active noise cancellation"
    },
    {
        "ad_id": "A3",
        "title": "FestiveWeave Men's Kurta",
        "category": "ethnic",
        "ctr": 0.06,
        "price": 1799,
        "desc": "Handcrafted cotton for every celebration"
    },
    {
        "ad_id": "A4",
        "title": "GlowShield SPF50 Moisturizer",
        "category": "skincare",
        "ctr": 0.09,
        "price": 699,
        "desc": "Dermatologist-tested daily protection"
    },
    {
        "ad_id": "A5",
        "title": "PaceMax Training Shoes",
        "category": "shoes",
        "ctr": 0.07,
        "price": 2299,
        "desc": "Grip + comfort for serious athletes"
    },
    {
        "ad_id": "A6",
        "title": "SoundWave 200 Over-Ear Headphones",
        "category": "electronics",
        "ctr": 0.10,
        "price": 2999,
        "desc": "Studio-quality sound, foldable design"
    },
    {
        "ad_id": "A7",
        "title": "EthnicLoom Embroidered Kurta",
        "category": "ethnic",
        "ctr": 0.05,
        "price": 1499,
        "desc": "Traditional motifs, modern fit"
    },
    {
        "ad_id": "A8",
        "title": "DermaCool Niacinamide Serum",
        "category": "skincare",
        "ctr": 0.11,
        "price": 799,
        "desc": "Brightening + oil control in one"
    },
    {
        "ad_id": "A9",
        "title": "StrideEase Casual Sneakers",
        "category": "shoes",
        "ctr": 0.09,
        "price": 1899,
        "desc": "All-day comfort, street-ready look"
    },
    {
        "ad_id": "A10",
        "title": "TechPulse Smart Fitness Band",
        "category": "electronics",
        "ctr": 0.08,
        "price": 2699,
        "desc": "Sleep + steps + SpO2 tracking"
    },
    {
        "ad_id": "A11",
        "title": "ArcticShield Winter Jacket",
        "category": "jackets",
        "ctr": 0.11,
        "price": 3499,
        "desc": "Heavy-duty insulated for cold weather"
    },
    {
        "ad_id": "A12",
        "title": "UrbanFrost Men's Puffer Jacket",
        "category": "jackets",
        "ctr": 0.09,
        "price": 2799,
        "desc": "Stylish & warm under ₹3000"
    },
    {
        "ad_id": "A13",
        "title": "AeroRun Summer Trainers",
        "category": "shoes",
        "ctr": 0.08,
        "price": 1499,
        "desc": "Lightweight running shoes designed for summer comfort under budget price."
    },
    {
        "ad_id": "A14",
        "title": "ProStride Elite Sports",
        "category": "shoes",
        "ctr": 0.11,
        "price": 3499,
        "desc": "Premium performance sports shoes with advanced cushioning technology for athletes."
    },
    {
        "ad_id": "A15",
        "title": "UrbanWalk Casual Sneakers",
        "category": "shoes",
        "ctr": 0.09,
        "price": 1799,
        "desc": "Casual sneakers offering great style and comfort under 2000 rupees only."
    },
    {
        "ad_id": "A16",
        "title": "ComfortStep Daily Walkers",
        "category": "shoes",
        "ctr": 0.06,
        "price": 2499,
        "desc": "Comfortable walking shoes ideal for daily wear and long hours."
    },
    {
        "ad_id": "A17",
        "title": "GymFlex Pro Trainers",
        "category": "shoes",
        "ctr": 0.10,
        "price": 3299,
        "desc": "Pro training shoes featuring durable sole and enhanced grip support."
    },
    {
        "ad_id": "A18",
        "title": "OfficeStyle Leather Loafers",
        "category": "shoes",
        "ctr": 0.07,
        "price": 1399,
        "desc": "Stylish loafers perfect for office wear at affordable market price."
    },
    {
        "ad_id": "A19",
        "title": "SoundWave ANC Earbuds",
        "category": "electronics",
        "ctr": 0.12,
        "price": 4999,
        "desc": "Wireless earbuds with active noise cancelling and long battery life."
    },
    {
        "ad_id": "A20",
        "title": "BassPod Bluetooth Headset",
        "category": "electronics",
        "ctr": 0.08,
        "price": 1599,
        "desc": "Bluetooth headphones delivering clear sound quality under 2000 rupees budget."
    },
    {
        "ad_id": "A21",
        "title": "FitTrack Smart Watch Pro",
        "category": "electronics",
        "ctr": 0.13,
        "price": 5999,
        "desc": "Premium fitness smartwatch tracking health metrics with water resistance feature."
    },
    {
        "ad_id": "A22",
        "title": "BoomBox Portable Speaker",
        "category": "electronics",
        "ctr": 0.07,
        "price": 2499,
        "desc": "Portable speaker with deep bass and wireless connectivity for music."
    },
    {
        "ad_id": "A23",
        "title": "PowerUp Fast Charger",
        "category": "electronics",
        "ctr": 0.05,
        "price": 1499,
        "desc": "Fast charging power bank ensuring reliable backup for all devices."
    },
    {
        "ad_id": "A24",
        "title": "ClearTone Wireless Buds",
        "category": "electronics",
        "ctr": 0.09,
        "price": 1699,
        "desc": "True wireless earbuds providing excellent value deal for daily users."
    },
    {
        "ad_id": "A25",
        "title": "RoyalWedd Kurta Set",
        "category": "ethnic",
        "ctr": 0.11,
        "price": 2999,
        "desc": "Men festive kurta wedding edition with premium embroidery and fabric."
    },
    {
        "ad_id": "A26",
        "title": "CottonEase Women Kurta",
        "category": "ethnic",
        "ctr": 0.08,
        "price": 1499,
        "desc": "Women cotton kurta set available under 2000 for casual wear."
    },
    {
        "ad_id": "A27",
        "title": "Heritage Blend Sherwani",
        "category": "ethnic",
        "ctr": 0.10,
        "price": 3499,
        "desc": "Traditional sherwani blend suitable for special occasions and gatherings."
    },
    {
        "ad_id": "A28",
        "title": "DailyWear Men Kurta",
        "category": "ethnic",
        "ctr": 0.06,
        "price": 999,
        "desc": "Simple cotton kurta offering comfort at budget price for men."
    },
    {
        "ad_id": "A29",
        "title": "FestiveLayer Ethnic Jacket",
        "category": "ethnic",
        "ctr": 0.07,
        "price": 2499,
        "desc": "Ethnic jacket layered design perfect for festive season celebrations today."
    },
    {
        "ad_id": "A30",
        "title": "ClassicPathani Suit Set",
        "category": "ethnic",
        "ctr": 0.05,
        "price": 1899,
        "desc": "Pathani suit set providing great value and traditional look always."
    },
    {
        "ad_id": "A31",
        "title": "DermaGuard SPF Moisturizer",
        "category": "skincare",
        "ctr": 0.12,
        "price": 1299,
        "desc": "Moisturizer SPF dermatologist tested for daily sun protection and hydration."
    },
    {
        "ad_id": "A32",
        "title": "PureClean Face Wash",
        "category": "skincare",
        "ctr": 0.06,
        "price": 399,
        "desc": "Gentle face wash suitable for all skin types affordable care."
    },
    {
        "ad_id": "A33",
        "title": "SunShield Premium Lotion",
        "category": "skincare",
        "ctr": 0.09,
        "price": 899,
        "desc": "Sunscreen lotion premium formula protecting skin from harmful UV rays."
    },
    {
        "ad_id": "A34",
        "title": "NightRepair Cream Pack",
        "category": "skincare",
        "ctr": 0.07,
        "price": 599,
        "desc": "Night cream repair formula available in value pack for savings."
    },
    {
        "ad_id": "A35",
        "title": "GlowPro Vitamin Serum",
        "category": "skincare",
        "ctr": 0.10,
        "price": 1499,
        "desc": "Vitamin C serum pro grade brightening skin tone effectively quickly."
    },
    {
        "ad_id": "A36",
        "title": "SoftLip Balm Combo",
        "category": "skincare",
        "ctr": 0.04,
        "price": 299,
        "desc": "Lip balm combo pack ensuring hydration at low cost price."
    },
    {
        "ad_id": "A37",
        "title": "WinterGuard Men Jacket",
        "category": "jackets",
        "ctr": 0.09,
        "price": 2499,
        "desc": "Winter jackets for men under 3000 with warm lining inside."
    },
    {
        "ad_id": "A38",
        "title": "ThermoPuffer Insulated Coat",
        "category": "jackets",
        "ctr": 0.13,
        "price": 4999,
        "desc": "Insulated puffer jacket warm enough for extreme cold weather conditions."
    },
    {
        "ad_id": "A39",
        "title": "WindBlocker Proof Jacket",
        "category": "jackets",
        "ctr": 0.08,
        "price": 2999,
        "desc": "Windcheater winter proof design blocking wind and keeping you dry."
    },
    {
        "ad_id": "A40",
        "title": "FleeceHood Cold Resistant",
        "category": "jackets",
        "ctr": 0.07,
        "price": 3499,
        "desc": "Hooded jacket cold weather resistant with soft inner fleece material."
    },
    {
        "ad_id": "A41",
        "title": "EcoWarm Fleece Jacket",
        "category": "jackets",
        "ctr": 0.06,
        "price": 1999,
        "desc": "Fleece jacket budget friendly option for layering during winter season."
    },
    {
        "ad_id": "A42",
        "title": "AlpineParka Premium Coat",
        "category": "jackets",
        "ctr": 0.14,
        "price": 5499,
        "desc": "Parka jacket premium insulated hooded coat for maximum warmth protection."
    }
]

# ---------------------------------------------------------------------------
# Advertiser Catalogue — maps advertisers to their categories & ad IDs
# Used by Flow 2: /advertiser/stats endpoint (Nike's view)
# ---------------------------------------------------------------------------
ADVERTISER_CATALOGUE = {
    "Nike": {
        "category": "shoes",
        "ads": ["A1", "A5", "A9", "A13", "A14", "A15", "A16", "A17"],
        "color": "#f97316",
    },
    "Samsung": {
        "category": "electronics",
        "ads": ["A2", "A6", "A10", "A19", "A20", "A21", "A22", "A23", "A24"],
        "color": "#3b82f6",
    },
    "FabIndia": {
        "category": "ethnic",
        "ads": ["A3", "A7", "A25", "A26", "A27", "A28", "A29", "A30"],
        "color": "#f59e0b",
    },
    "Lakme": {
        "category": "skincare",
        "ads": ["A4", "A8", "A31", "A32", "A33", "A34", "A35", "A36"],
        "color": "#ec4899",
    },
}

