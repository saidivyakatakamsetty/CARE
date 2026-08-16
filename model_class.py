import pandas as pd


class HybridRiskModel:
    def __init__(self):
        self.scaler = None
        self.iso_forest = None
        self.xgb_model = None
        self.feature_cols = None
        self.risk_threshold = 0.5

    def score(self, X):
        """Given a dataframe of features, return risk_score and anomaly flag."""
        X = X[self.feature_cols]
        risk_scores = self.xgb_model.predict_proba(X)[:, 1]
        X_scaled = self.scaler.transform(X)
        iso_pred = self.iso_forest.predict(X_scaled)
        return pd.DataFrame({
            "risk_score": risk_scores,
            "high_risk": risk_scores >= self.risk_threshold,
            "baseline_anomaly": iso_pred == -1,
        }, index=X.index)