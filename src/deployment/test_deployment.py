import unittest
import requests
import json
import os
from pathlib import Path
import numpy as np
from datetime import datetime
import multiprocessing
import time

from .model_manager import ModelManager
from .decision_maker import ChargingDecisionMaker
from .evaluate_real_world import RealWorldEvaluator
from .api import app
import uvicorn

def run_server():
    """Run the FastAPI server in a separate process."""
    uvicorn.run(app, host="127.0.0.1", port=8000)

class TestDeployment(unittest.TestCase):
    """Test suite for deployment components."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Initialize components
        cls.model_manager = ModelManager()
        cls.model, cls.normalizer = cls.model_manager.load_model()
        cls.inference_pipeline = cls.model_manager.create_inference_pipeline(cls.model, cls.normalizer)
        cls.decision_maker = ChargingDecisionMaker(cls.inference_pipeline)
        
        # API endpoint
        cls.api_url = "http://127.0.0.1:8000"
        
        # Start API server in a separate process
        cls.server_process = multiprocessing.Process(target=run_server)
        cls.server_process.start()
        time.sleep(2)  # Wait for server to start
        
        # Create test data matching environment's observation space
        cls.test_state = {
            "queue_length": 3,
            "available_stations": 5,
            "current_load": 600.0,
            "electricity_price": 0.3,
            "time_of_day": 12.0
        }
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        # Stop the API server
        cls.server_process.terminate()
        cls.server_process.join()
    
    def test_model_loading(self):
        """Test if model and normalizer load correctly."""
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.normalizer)
        self.assertEqual(self.model.observation_space.shape, (5,))  # [queue_length, available_stations, current_load, electricity_price, time_of_day]
        self.assertEqual(self.model.action_space.n, 11)  # num_charging_stations + 1
    
    def test_decision_maker(self):
        """Test decision maker functionality."""
        decision = self.decision_maker.make_decision(
            current_state=self.test_state,
            constraints=None
        )
        
        self.assertIsInstance(decision, dict)
        self.assertIn("action", decision)
        self.assertIn("state", decision)
        self.assertIn("info", decision)
        self.assertIn("constraints_applied", decision)
        self.assertIsInstance(decision["action"], int)  # Discrete action space
    
    def test_api_endpoints(self):
        """Test API endpoints."""
        # Test root endpoint
        response = requests.get(f"{self.api_url}/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("endpoints", response.json())
        
        # Test make_decision endpoint
        response = requests.post(
            f"{self.api_url}/make_decision",
            json=self.test_state
        )
        self.assertEqual(response.status_code, 200)
        decision = response.json()
        self.assertIn("action", decision)
        self.assertIn("state", decision)
        
        # Test decision_history endpoint
        response = requests.get(f"{self.api_url}/decision_history")
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json(), list)
        
        # Test model_info endpoint
        response = requests.get(f"{self.api_url}/model_info")
        self.assertEqual(response.status_code, 200)
        info = response.json()
        self.assertIn("model_type", info)
        self.assertIn("input_shape", info)
        self.assertIn("output_shape", info)
        
        # Test save_history endpoint
        response = requests.post(f"{self.api_url}/save_history")
        self.assertEqual(response.status_code, 200)
        self.assertIn("message", response.json())
    
    def test_real_world_evaluation(self):
        """Test real-world evaluation functionality."""
        evaluator = RealWorldEvaluator()
        
        # Generate scenarios
        scenarios = evaluator.generate_real_world_scenarios(n_scenarios=5)
        self.assertEqual(len(scenarios), 5)
        
        # Evaluate scenarios
        results = evaluator.evaluate_scenarios(scenarios)
        self.assertEqual(len(results), 5)
        
        # Analyze results
        analysis = evaluator.analyze_results(results)
        self.assertIn("total_scenarios", analysis)
        self.assertIn("average_charging_rate", analysis)
        self.assertIn("average_queue_management", analysis)
        
        # Create and save plots
        plot_path = evaluator.plot_results(results, analysis)
        self.assertTrue(os.path.exists(plot_path))
        
        # Save results
        results_path = evaluator.save_results(results, analysis, plot_path)
        self.assertTrue(os.path.exists(results_path))
        
        # Clean up
        os.remove(plot_path)
        os.remove(results_path)
    
    def test_constraints(self):
        """Test constraint handling."""
        constraints = {
            "max_queue_length": 5,
            "min_charging_rate": 0.1,
            "max_charging_rate": 0.8,
            "price_threshold": 0.4,
            "load_threshold": 0.7
        }
        
        # Test with constraints
        decision = self.decision_maker.make_decision(
            current_state=self.test_state,
            constraints=constraints
        )
        
        self.assertTrue(decision["constraints_applied"])
        self.assertGreaterEqual(decision["action"], 0)
        self.assertLessEqual(decision["action"], 10)  # num_charging_stations
    
    def test_error_handling(self):
        """Test error handling in API endpoints."""
        # Test invalid state
        invalid_state = {
            "queue_length": "invalid",
            "available_stations": 5,
            "current_load": 600.0,
            "electricity_price": 0.3,
            "time_of_day": 12.0
        }
        
        response = requests.post(
            f"{self.api_url}/make_decision",
            json=invalid_state
        )
        self.assertEqual(response.status_code, 422)  # Validation error
        
        # Test invalid constraints
        invalid_constraints = {
            "max_queue_length": "invalid",
            "min_charging_rate": "invalid"
        }
        
        response = requests.post(
            f"{self.api_url}/make_decision",
            json={"state": self.test_state, "constraints": invalid_constraints}
        )
        self.assertEqual(response.status_code, 422)  # Validation error

if __name__ == "__main__":
    unittest.main() 