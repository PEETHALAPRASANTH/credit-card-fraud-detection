import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schemas import TransactionRequest
from .prediction import FraudPredictor
from .config import METRICS_PATH, METADATA_PATH

app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Machine-learning API comparing Random Forest and AdaBoost.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_predictor = None

@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": _predictor is not None}

@app.on_event("startup")
def load_model():
    global _predictor
    try:
        _predictor = FraudPredictor()
    except FileNotFoundError:
        _predictor = None

@app.get("/api/metrics")
def metrics():
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="Metrics unavailable. Train the models first.")
    with open(METRICS_PATH, encoding="utf-8") as f:
        return json.load(f)

@app.post("/api/predict")
def predict(request: TransactionRequest):
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model unavailable. Train models first.")
    try:
        return _predictor.predict(request.model_dump())
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
