import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns

from .model_manager import ModelManager
from .decision_maker import ChargingDecisionMaker

class RealWorldEvaluator:
    """Evaluates the model on real-world data streams."""
    
    def __init__(self, model_dir: str = "models"):
        """Initialize the evaluator."""
        self.model_manager = ModelManager(model_dir)
        self.model, self.normalizer = self.model_manager.load_model()
        self.inference_pipeline = self.model_manager.create_inference_pipeline(self.model, self.normalizer)
        self.decision_maker = ChargingDecisionMaker(self.inference_pipeline)
        
        # Create results directory
        self.results_dir = Path("results/real_world_evaluation")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_real_world_scenarios(self, n_scenarios: int = 5) -> List[Dict[str, Any]]:
        """Generate realistic scenarios for evaluation."""
        scenarios = []
        
        # Generate scenarios for different times of day and conditions
        for i in range(n_scenarios):
            # Random time between 0 and 24
            time_of_day = np.random.uniform(0, 24)
            
            # Generate realistic queue length (0-20)
            queue_length = np.random.randint(0, 21)
            
            # Generate realistic available stations (0-10)
            available_stations = np.random.randint(0, 11)
            
            # Generate realistic current load (500-1500 kW)
            current_load = np.random.uniform(500, 1500)
            
            # Generate realistic electricity price (0.1-0.5)
            electricity_price = np.random.uniform(0.1, 0.5)
            
            scenario = {
                "queue_length": int(queue_length),
                "available_stations": int(available_stations),
                "current_load": float(current_load),
                "electricity_price": float(electricity_price),
                "time_of_day": float(time_of_day),
                "timestamp": datetime.now().isoformat()
            }
            
            scenarios.append(scenario)
        
        return scenarios
    
    def evaluate_scenarios(self, scenarios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate the model on given scenarios."""
        results = []
        
        for scenario in scenarios:
            # Make decision
            decision = self.decision_maker.make_decision(
                current_state=scenario,
                constraints=None
            )
            
            # Combine scenario and decision
            result = {
                "scenario": scenario,
                "decision": decision
            }
            
            results.append(result)
        
        return results
    
    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the evaluation results."""
        analysis = {
            "total_scenarios": len(results),
            "average_charging_rate": np.mean([r["decision"]["action"] for r in results]),
            "queue_management": np.mean([r["decision"]["action"] for r in results]),
            "constraints_applied_count": sum(1 for r in results if r["decision"]["constraints_applied"]),
            "timestamp": datetime.now().isoformat()
        }
        
        return analysis
    
    def plot_results(self, results: List[Dict[str, Any]], analysis: Dict[str, Any]):
        """Create visualization plots for the results."""
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Charging Rate vs Queue Length
        axes[0, 0].scatter(
            [r["scenario"]["queue_length"] for r in results],
            [r["decision"]["action"] for r in results],
            alpha=0.5
        )
        axes[0, 0].set_xlabel("Queue Length")
        axes[0, 0].set_ylabel("Charging Rate")
        axes[0, 0].set_title("Charging Rate vs Queue Length")
        
        # Plot 2: Charging Rate vs Available Stations
        axes[0, 1].scatter(
            [r["scenario"]["available_stations"] for r in results],
            [r["decision"]["action"] for r in results],
            alpha=0.5
        )
        axes[0, 1].set_xlabel("Available Stations")
        axes[0, 1].set_ylabel("Charging Rate")
        axes[0, 1].set_title("Charging Rate vs Available Stations")
        
        # Plot 3: Charging Rate vs Current Load
        axes[1, 0].scatter(
            [r["scenario"]["current_load"] for r in results],
            [r["decision"]["action"] for r in results],
            alpha=0.5
        )
        axes[1, 0].set_xlabel("Current Load")
        axes[1, 0].set_ylabel("Charging Rate")
        axes[1, 0].set_title("Charging Rate vs Current Load")
        
        # Plot 4: Charging Rate vs Electricity Price
        axes[1, 1].scatter(
            [r["scenario"]["electricity_price"] for r in results],
            [r["decision"]["action"] for r in results],
            alpha=0.5
        )
        axes[1, 1].set_xlabel("Electricity Price")
        axes[1, 1].set_ylabel("Charging Rate")
        axes[1, 1].set_title("Charging Rate vs Electricity Price")
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = self.results_dir / f"evaluation_plots_{timestamp}.png"
        plt.savefig(plot_path)
        plt.close()
        
        return str(plot_path)
    
    def save_results(self, results: List[Dict[str, Any]], analysis: Dict[str, Any], plot_path: str):
        """Save evaluation results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_path = self.results_dir / f"evaluation_results_{timestamp}.json"
        with open(results_path, "w") as f:
            json.dump({
                "results": results,
                "analysis": analysis,
                "plot_path": plot_path
            }, f, indent=2)
        
        return str(results_path)

def main():
    """Run the real-world evaluation."""
    # Initialize evaluator
    evaluator = RealWorldEvaluator()
    
    # Generate scenarios
    scenarios = evaluator.generate_real_world_scenarios(n_scenarios=100)
    
    # Evaluate scenarios
    results = evaluator.evaluate_scenarios(scenarios)
    
    # Analyze results
    analysis = evaluator.analyze_results(results)
    
    # Create plots
    plot_path = evaluator.plot_results(results, analysis)
    
    # Save results
    results_path = evaluator.save_results(results, analysis, plot_path)
    
    print(f"Evaluation completed successfully!")
    print(f"Results saved to: {results_path}")
    print(f"Plots saved to: {plot_path}")
    print("\nAnalysis Summary:")
    print(f"Total scenarios evaluated: {analysis['total_scenarios']}")
    print(f"Average charging rate: {analysis['average_charging_rate']:.3f}")
    print(f"Average queue management: {analysis['queue_management']:.3f}")
    print(f"Constraints applied in {analysis['constraints_applied_count']} scenarios")

if __name__ == "__main__":
    main() 