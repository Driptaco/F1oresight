"""
F1oreSight API Server
Run with: uvicorn api_server:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import your existing predictor
from main import F1Predictor

app = FastAPI(title="F1oreSight API")

# Allow your HTML file to call this API (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance (loaded once at startup)
predictor = F1Predictor()
model_trained = False


# ── Request/Response schemas ──────────────────────────────────────────────────

class TrainRequest(BaseModel):
    year_start: int = 2022
    year_end: int = 2024

class PredictRequest(BaseModel):
    year: int
    round_number: int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_trained": model_trained}


@app.post("/train")
def train(req: TrainRequest):
    """Load historical data and train the model."""
    global model_trained
    try:
        print(f"Loading data {req.year_start}–{req.year_end}...")
        df = predictor.load_race_data(req.year_start, req.year_end)
        print(f"Loaded {len(df)} records. Training...")
        predictor.train(df)
        model_trained = True
        return {
            "success": True,
            "records": len(df),
            "message": f"Model trained on {len(df)} records ({req.year_start}–{req.year_end})"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
def predict(req: PredictRequest):
    """Predict race results for a given year + round."""
    if not model_trained:
        raise HTTPException(status_code=400, detail="Model not trained yet. Call /train first.")
    try:
        result_df = predictor.predict_race(req.year, req.round_number)
        if result_df is None:
            raise HTTPException(status_code=500, detail="Prediction returned no results.")

        # Convert DataFrame to list of dicts for JSON response
        records = result_df.to_dict(orient="records")
        return {"success": True, "predictions": records}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
