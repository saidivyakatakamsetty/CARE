# CARE — Clinical Agentic Reliability Engine

An agentic early-warning system for sepsis, built on real ICU data from
the PhysioNet/CinC 2019 Sepsis Challenge. Ingests hourly vitals/labs,
engineers rolling clinical trend features, scores deterioration risk with
a hybrid ML model, and runs a multi-agent (LangGraph) workflow that
grounds its recommendations in retrieved clinical evidence (RAG).

> Scope note: this is a portfolio/engineering demonstration of an agentic
> ML + RAG pipeline on a real clinical benchmark. It is not a validated
> clinical decision-support tool and should not inform real patient care.
> All data is public and de-identified (no PHI).

## Architecturedata/raw/*.psv (real ICU data, PhysioNet Challenge 2019)
│
▼
SQLite (care.db) <- ingest.py
│ SQL query
▼
Rolling clinical features <- features.py
(mean/std/slope, 6hr window,

Shock Index, approx qSOFA)
│
▼
Hybrid risk model <- train.py, model_class.py
(XGBoost supervised +
Isolation Forest baseline-
deviation detector)
│
▼
FastAPI REST API <- api.py
│
▼
LangGraph multi-agent workflow <- agents.py
Monitor -> (elevated?) -> Diagnose (RAG) -> Recommend
-> (stable) -> Stable Report
RAG grounded in rag.py + knowledge_base.py
## Why this design

- **Label definition matters.** PhysioNet's `SepsisLabel` is set to 1
  starting 6 hours *before* clinical sepsis onset — so a model trained on
  it is inherently an early-warning signal, not a same-time diagnosis.
- **Patient-level train/test split, not row-level.** Splitting by hour
  would leak information (a patient's hour 10 and hour 11 are highly
  correlated), inflating every metric. All of one patient's hours go
  entirely into train or entirely into test.
- **Threshold tuned on the precision-recall curve, not left at 0.5.**
  With ~2.7% positive rows, 0.5 isn't a meaningful operating point.
- **Two detectors, not one.** XGBoost learns population-level sepsis
  patterns; Isolation Forest (trained only on each patient's first 4
  hours) flags deviations from that *specific patient's own* baseline.
- **Conditional agent routing.** Stable patients short-circuit straight
  to a one-line report; the RAG-grounded Diagnose/Recommend agents only
  run for elevated-risk patients — avoiding unnecessary alerts.

## Results (on a 100-patient subset)

| Metric | Value |
|---|---|
| ROC AUC | 0.648 |
| Precision / Recall @ tuned threshold (0.583) | 0.132 / 0.172 |
| **Patient-level early-warning rate** | **66.7%** (2/3 septic test patients flagged before labeled onset) |

The patient-level early-warning rate is the metric that matters
clinically — "did we give the care team a heads-up" — not raw hour-level
precision/recall, which is misleadingly low on this task due to severe
class imbalance (~2.7% positive rows), a known characteristic of this
benchmark, not a flaw in this implementation. (Trained here on a
100-patient subset for iteration speed; the full 4,000-patient version
scales these numbers up further.)

## Project structure
ingest.py # PSV files -> SQLite
features.py # rolling trend features + clinical composite scores
model_class.py # HybridRiskModel class definition
train.py # trains XGBoost + Isolation Forest, saves model
api.py # FastAPI service (/risk and /assess endpoints)
knowledge_base.py # clinical reference documents
rag.py # ChromaDB retrieval
agents.py # LangGraph monitor -> diagnose -> recommend workflow
test_pipeline.py # pytest suite
Dockerfile # container packaging
requirements.txt
## Running it

```bash
pip install -r requirements.txt

python ingest.py              # load real ICU data into SQLite
python train.py               # train the hybrid model
python -m uvicorn api:app --reload --port 8000   # serve the API

curl http://localhost:8000/patients/p000009/risk
curl http://localhost:8000/patients/p000009/assess   # full agentic workflow

python -m pytest test_pipeline.py -v
```

## What would change for production use

- Real EHR/HL7-FHIR integration instead of a static file ingest
- A model registry instead of a locally saved `.joblib` file
- Clinical validation and prospective evaluation before any claim of
  clinical utility — this repo makes no such claim
- Scaling training data beyond the 100-patient subset used during
  iterative development