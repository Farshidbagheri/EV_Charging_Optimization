from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import uvicorn
from datetime import datetime
import json
import os
from pathlib import Path

from .model_manager import ModelManager
from .decision_maker import ChargingDecisionMaker

app = FastAPI(
    title="EV Charging Optimization API",
    description="API for real-time EV charging optimization",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI at /docs
    redoc_url="/redoc"  # ReDoc at /redoc
)

# Initialize model manager and decision maker
model_manager = ModelManager()
model, normalizer = model_manager.load_model()
inference_pipeline = model_manager.create_inference_pipeline(model, normalizer)
decision_maker = ChargingDecisionMaker(inference_pipeline)

class ChargingState(BaseModel):
    """Input state for charging decision."""
    queue_length: int
    available_stations: int
    current_load: float
    electricity_price: float
    time_of_day: float

class ChargingConstraints(BaseModel):
    """Constraints for charging decisions."""
    max_queue_length: Optional[int] = None
    min_charging_rate: Optional[float] = None
    max_charging_rate: Optional[float] = None
    price_threshold: Optional[float] = None
    load_threshold: Optional[float] = None

class ChargingDecision(BaseModel):
    """Output charging decision."""
    timestamp: str
    action: int
    state: Dict[str, Any]
    info: Dict[str, Any]
    constraints_applied: bool

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "EV Charging Optimization API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "redoc": "/redoc",
            "make_decision": "/make_decision",
            "decision_history": "/decision_history",
            "model_info": "/model_info",
            "save_history": "/save_history"
        }
    }

@app.post("/make_decision", response_model=ChargingDecision)
async def make_charging_decision(
    state: ChargingState,
    constraints: Optional[ChargingConstraints] = None
) -> ChargingDecision:
    """Make a charging decision based on current state and constraints."""
    try:
        decision = decision_maker.make_decision(
            current_state=state.dict(),
            constraints=constraints.dict() if constraints else None
        )
        return ChargingDecision(**decision)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/decision_history", response_model=List[ChargingDecision])
async def get_decision_history(limit: Optional[int] = None) -> List[ChargingDecision]:
    """Get the history of charging decisions."""
    try:
        history = decision_maker.get_decision_history(limit)
        return [ChargingDecision(**decision) for decision in history]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model_info")
async def get_model_info() -> Dict[str, Any]:
    """Get information about the deployed model."""
    try:
        return {
            "model_type": "PPO",
            "input_shape": model.observation_space.shape,
            "output_shape": (model.action_space.n,),
            "version": "1.0.0",
            "last_updated": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/save_history")
async def save_decision_history() -> Dict[str, str]:
    """Save the decision history to file."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"decision_history_{timestamp}.json"
        path = os.path.join("results", filename)
        
        decision_maker.save_decision_history(path)
        return {"message": f"History saved to {path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def start_api(host: str = "0.0.0.0", port: int = 8000):
    """Start the FastAPI server."""
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_api() 