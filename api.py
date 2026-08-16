import joblib
from fastapi import FastAPI, HTTPException

from features import load_patient, compute_features
from model_class import HybridRiskModel
from agents import build_graph

app = FastAPI(title="CARE API")

# Load the trained model once, when the server starts (not on every request)
model = joblib.load("data/hybrid_model.joblib")

# Build the LangGraph agent workflow once, when the server starts
agent_app = build_graph()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/patients/{patient_id}/risk")
def get_patient_risk(patient_id: str):
    df = load_patient(patient_id)

    if len(df) == 0:
        raise HTTPException(status_code=404, detail=f"No data found for patient {patient_id}")

    feats = compute_features(df)
    feats = feats.fillna(0)  # same cleanup as in training

    scored = model.score(feats)

    # Return the LATEST hour's risk assessment (most current status)
    latest = scored.iloc[-1]
    latest_hour = feats.iloc[-1]

    return {
        "patient_id": patient_id,
        "icu_los_hour": int(latest_hour["icu_los_hour"]),
        "risk_score": float(latest["risk_score"]),
        "high_risk": bool(latest["high_risk"]),
        "baseline_anomaly": bool(latest["baseline_anomaly"]),
        "hours_of_data": len(df),
    }


@app.get("/patients/{patient_id}/assess")
def assess_patient(patient_id: str):
    df = load_patient(patient_id)
    if len(df) == 0:
        raise HTTPException(status_code=404, detail=f"No data found for patient {patient_id}")

    result = agent_app.invoke({"patient_id": patient_id})

    return {
        "patient_id": patient_id,
        "route": result["route"],
        "risk_score": result["risk_score"],
        "recommendation": result["recommendation"],
    }