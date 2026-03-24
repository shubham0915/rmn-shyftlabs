"""
ltr.py — Learning-to-Rank (XGBoost) Machine Learning Layer
Agent 3: AI/ML Specialist

Responsibilities:
 - Synthesise 10,000 contextual ad interactions.
 - Train an XGBoost Regressor to predict P(click) based on features.
 - Persist the model to disk so it doesn't retrain on every request.
 - Expose `score_candidates()` for the ranking pipeline.
"""

import os
import logging
import numpy as np
import xgboost as xgb

from src.config import DATA_DIR

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(DATA_DIR, "ltr_model.json")
_model: xgb.Booster | None = None

def _generate_synthetic_interactions(n_samples: int = 10000) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic training data.
    Features: [similarity, historical_ctr, price_normalized, budget_intent]
    Label: 0 (no click) or 1 (click)
    """
    np.random.seed(42)  # Deterministic compilation bounds

    # 1. Generate random features
    sims = np.random.beta(a=2, b=5, size=n_samples) 
    ctrs = np.random.uniform(0.01, 0.15, size=n_samples)
    prices = np.random.uniform(0.0, 1.0, size=n_samples)
    budget_intent = np.random.randint(0, 2, size=n_samples)

    # 2. Mathematical "true" probability function (simulating reality)
    base_prob = (0.6 * sims) + (0.3 * (ctrs / 0.15))
    
    # Ground truth: Budget intent + low price = massive boost.
    budget_bonus = np.where((budget_intent == 1) & (prices < 0.3), 0.3, 0.0)
    budget_penalty = np.where((budget_intent == 1) & (prices > 0.7), -0.3, 0.0)
    
    final_prob = base_prob + budget_bonus + budget_penalty
    final_prob = np.clip(final_prob + np.random.normal(0, 0.1, size=n_samples), 0, 1)
    
    # 3. Generate binary labels based on the probability
    labels = np.random.binomial(n=1, p=final_prob)

    features = np.column_stack((sims, ctrs, prices, budget_intent))
    return features, labels


def train_and_save_model(force: bool = False) -> None:
    """Train the model and save to disk if it doesn't already exist."""
    if os.path.exists(MODEL_PATH) and not force:
        logger.info("[Agent3 LTR] XGBoost model already exists on disk. Skipping training.")
        return

    logger.info(f"[Agent3 LTR] Synthesizing 10,000 interactions to train XGBoost LTR model...")
    X, y = _generate_synthetic_interactions()

    # xgb native training is extremely fast and lightweight
    dtrain = xgb.DMatrix(X, label=y)
    
    params = {
        "objective": "reg:logistic",  # Probability outputs
        "eval_metric": "logloss",
        "max_depth": 4,
        "eta": 0.1,
        "nthread": 1  # CPU friendly
    }

    logger.info("[Agent3 LTR] Training model...")
    bst = xgb.train(params, dtrain, num_boost_round=50)
    
    os.makedirs(DATA_DIR, exist_ok=True)
    logger.info(f"[Agent3 LTR] Saving compiled model to {MODEL_PATH}")
    bst.save_model(MODEL_PATH)


def get_model() -> xgb.Booster:
    """Load the model into memory natively (lazy loaded)."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            logger.warning("[Agent3 LTR] Model file missing! Triggering emergency retraining...")
            train_and_save_model()
            
        _model = xgb.Booster()
        _model.load_model(MODEL_PATH)
    return _model


def score_candidates(features_matrix: list[list[float]]) -> list[float]:
    """
    Given an MxN feature list representing [sim, ctr, price_norm, budget_intent], 
    return a list of predicted p(click) probabilities.
    """
    if not features_matrix:
        return []
        
    bst = get_model()
    dtest = xgb.DMatrix(np.array(features_matrix, dtype=np.float32))
    
    # predict returns a numpy array of floats (probabilities)
    preds = bst.predict(dtest)
    return preds.tolist()
