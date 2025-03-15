import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
import json
import os
from pathlib import Path

class ChargingDecisionMaker:
    """Real-time decision maker for EV charging optimization."""
    
    def __init__(self, inference_pipeline: callable, config_path: str = "configs/deployment_config.json"):
        """Initialize the decision maker."""
        self.inference_pipeline = inference_pipeline
        self.config = self._load_config(config_path)
        self.decision_history = []
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load deployment configuration."""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            return {
                'max_queue_length': 10,
                'min_charging_rate': 0.1,
                'max_charging_rate': 1.0,
                'price_threshold': 0.8,
                'load_threshold': 0.9,
                'decision_interval': 5  # minutes
            }
    
    def make_decision(self, 
                     current_state: Dict[str, Any],
                     constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a charging decision based on current state and constraints."""
        # Prepare observation
        observation = self._prepare_observation(current_state)
        
        # Get model prediction
        action, info = self.inference_pipeline(observation)
        
        # Apply constraints if provided
        if constraints:
            action = self._apply_constraints(action, constraints)
        
        # Create decision
        decision = {
            'timestamp': datetime.now().isoformat(),
            'action': action.tolist(),
            'state': current_state,
            'info': info,
            'constraints_applied': bool(constraints)
        }
        
        # Store decision in history
        self.decision_history.append(decision)
        
        return decision
    
    def _prepare_observation(self, state: Dict[str, Any]) -> np.ndarray:
        """Prepare observation for the model."""
        # Extract relevant features from state
        features = [
            state.get('battery_level', 0),
            state.get('grid_load', 0),
            state.get('price', 0),
            state.get('queue_length', 0),
            state.get('time_of_day', 0),
            state.get('day_of_week', 0)
        ]
        
        return np.array(features, dtype=np.float32)
    
    def _apply_constraints(self, action: np.ndarray, constraints: Dict[str, Any]) -> np.ndarray:
        """Apply constraints to the action."""
        constrained_action = action.copy()
        
        # Apply queue length constraint
        if constraints.get('max_queue_length'):
            if action[0] > constraints['max_queue_length']:
                constrained_action[0] = constraints['max_queue_length']
        
        # Apply charging rate constraints
        if constraints.get('min_charging_rate'):
            constrained_action[1] = max(action[1], constraints['min_charging_rate'])
        if constraints.get('max_charging_rate'):
            constrained_action[1] = min(action[1], constraints['max_charging_rate'])
        
        return constrained_action
    
    def get_decision_history(self, limit: Optional[int] = None) -> list:
        """Get the decision history."""
        if limit:
            return self.decision_history[-limit:]
        return self.decision_history
    
    def save_decision_history(self, path: str) -> None:
        """Save decision history to file."""
        with open(path, 'w') as f:
            json.dump(self.decision_history, f, indent=2)
    
    def load_decision_history(self, path: str) -> None:
        """Load decision history from file."""
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.decision_history = json.load(f) 