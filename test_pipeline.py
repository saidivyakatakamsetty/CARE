import pandas as pd
from features import compute_features, load_patient
from model_class import HybridRiskModel
import joblib


def test_features_have_expected_columns():
    df = load_patient("p000009")
    feats = compute_features(df)
    assert "shock_index" in feats.columns
    assert "qsofa_approx" in feats.columns
    assert len(feats) == len(df)


def test_model_scores_are_valid_probabilities():
    model = joblib.load("data/hybrid_model.joblib")
    df = load_patient("p000009")
    feats = compute_features(df).fillna(0)
    scored = model.score(feats)
    assert scored["risk_score"].between(0, 1).all()


def test_known_septic_patient_flagged_high_risk():
    """Sanity check: our known septic patient should score high risk
    by the end of their stay (this is a regression test, not a proof
    the model is 'correct' in general)."""
    model = joblib.load("data/hybrid_model.joblib")
    df = load_patient("p000009")
    feats = compute_features(df).fillna(0)
    scored = model.score(feats)
    final_risk = scored.iloc[-1]["risk_score"]
    assert final_risk > 0.5