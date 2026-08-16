from typing import TypedDict
import joblib
from langgraph.graph import StateGraph, END

from features import load_patient, compute_features
from model_class import HybridRiskModel
from rag import retrieve, query_from_driving_feature


class AgentState(TypedDict, total=False):
    patient_id: str
    risk_score: float
    route: str
    evidence: list
    recommendation: str


model = joblib.load("data/hybrid_model.joblib")


def monitor_node(state: AgentState) -> AgentState:
    df = load_patient(state["patient_id"])
    feats = compute_features(df).fillna(0)
    scored = model.score(feats)
    latest_risk = float(scored.iloc[-1]["risk_score"])

    route = "elevated" if latest_risk >= 0.5 else "stable"

    return {
        **state,
        "risk_score": latest_risk,
        "route": route,
    }


def route_decision(state: AgentState) -> str:
    return state["route"]


def diagnose_node(state: AgentState) -> AgentState:
    query = query_from_driving_feature("lactate_slope")  # simplified for now
    evidence = retrieve(query, k=2)
    return {**state, "evidence": evidence}


def recommend_node(state: AgentState) -> AgentState:
    evidence_text = "\n".join(f"- {e['title']}: {e['text']}" for e in state["evidence"])
    rec = (
        f"Patient {state['patient_id']}: ELEVATED risk (score={state['risk_score']:.2f}).\n"
        f"Relevant clinical context:\n{evidence_text}"
    )
    return {**state, "recommendation": rec}


def stable_report_node(state: AgentState) -> AgentState:
    rec = f"Patient {state['patient_id']}: stable (score={state['risk_score']:.2f})."
    return {**state, "recommendation": rec}


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("monitor", monitor_node)
    graph.add_node("diagnose", diagnose_node)
    graph.add_node("recommend", recommend_node)
    graph.add_node("stable_report", stable_report_node)

    graph.set_entry_point("monitor")

    graph.add_conditional_edges(
        "monitor",
        route_decision,
        {"elevated": "diagnose", "stable": "stable_report"},
    )
    graph.add_edge("diagnose", "recommend")
    graph.add_edge("recommend", END)
    graph.add_edge("stable_report", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()

    print("=== Septic patient ===")
    result = app.invoke({"patient_id": "p000009"})
    print(result["recommendation"])

    print("\n=== Different patient (try one that stayed stable) ===")
    result2 = app.invoke({"patient_id": "p000001"})
    print(result2["recommendation"])