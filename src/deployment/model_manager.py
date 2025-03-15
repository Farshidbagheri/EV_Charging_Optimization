import os
import torch
import numpy as np
from typing import Dict, Any, Optional
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from pathlib import Path
from gymnasium import spaces
import pickle

class ModelManager:
    """Manages model saving, loading, and deployment."""
    
    def __init__(self, model_dir: str = "models"):
        """Initialize the model manager."""
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "final_model")
        self.normalizer_path = os.path.join(model_dir, "vec_normalize.pkl")
        os.makedirs(model_dir, exist_ok=True)
    
    def _create_dummy_env(self, observation_space=None):
        """Create a dummy environment with the correct observation space."""
        from gymnasium import Env
        
        class DummyEnv(Env):
            def __init__(self, obs_space=None):
                super().__init__()
                if obs_space is None:
                    self.observation_space = spaces.Box(
                        low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
                    )
                else:
                    self.observation_space = obs_space
                self.action_space = spaces.Box(
                    low=-1, high=1, shape=(2,), dtype=np.float32
                )
            
            def reset(self, seed=None):
                return np.zeros(self.observation_space.shape, dtype=np.float32), {}
            
            def step(self, action):
                return np.zeros(self.observation_space.shape, dtype=np.float32), 0, True, False, {}
        
        return DummyEnv(observation_space)
    
    def save_model(self, model: PPO, normalizer: Optional[VecNormalize] = None) -> None:
        """Save the trained model and normalizer."""
        # Save the model
        model.save(self.model_path)
        print(f"Model saved to {self.model_path}")
        
        # Save the normalizer if provided
        if normalizer is not None:
            normalizer.save(self.normalizer_path)
            print(f"Normalizer saved to {self.normalizer_path}")
    
    def load_model(self) -> tuple[PPO, Optional[VecNormalize]]:
        """Load the saved model and normalizer."""
        # Load the model
        model = PPO.load(self.model_path)
        print(f"Model loaded from {self.model_path}")
        
        # Load the normalizer if it exists
        normalizer = None
        if os.path.exists(self.normalizer_path):
            # Create a dummy environment with the model's observation space
            dummy_env = DummyVecEnv([lambda: self._create_dummy_env(model.observation_space)])
            
            # Load the normalizer with the correct venv
            normalizer = VecNormalize.load(self.normalizer_path, venv=dummy_env)
            print(f"Normalizer loaded from {self.normalizer_path}")
        
        return model, normalizer
    
    def export_to_onnx(self, model: PPO, input_shape: tuple) -> str:
        """Export the model to ONNX format."""
        onnx_path = os.path.join(self.model_dir, "model.onnx")
        
        # Create dummy input
        dummy_input = torch.randn(1, *input_shape)
        
        # Export to ONNX
        torch.onnx.export(
            model.policy.mlp_extractor,
            dummy_input,
            onnx_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"Model exported to ONNX format at {onnx_path}")
        return onnx_path
    
    def create_inference_pipeline(self, model: PPO, normalizer: Optional[VecNormalize] = None) -> callable:
        """Create an inference pipeline for real-time predictions."""
        def inference_pipeline(observation: np.ndarray) -> tuple[np.ndarray, Dict[str, Any]]:
            """Process input and return predictions."""
            # Normalize observation if normalizer is available
            if normalizer is not None:
                observation = normalizer.normalize_obs(observation)
            
            # Get prediction
            action, _ = model.predict(observation, deterministic=True)
            
            # Denormalize action if normalizer is available
            if normalizer is not None:
                action = normalizer.denormalize_act(action)
            
            # Create info dictionary
            info = {
                'action': action,
                'timestamp': np.datetime64('now'),
                'model_version': '1.0'
            }
            
            return action, info
        
        return inference_pipeline 