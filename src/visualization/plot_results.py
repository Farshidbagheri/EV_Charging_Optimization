import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import pandas as pd

def load_results(results_path: str) -> Dict:
    """Load evaluation results from JSON file."""
    with open(results_path, 'r') as f:
        return json.load(f)

def create_scenario_comparison(results: List[Dict], save_path: str):
    """Create comparison plots for different scenarios."""
    scenarios = list(results[0]['evaluation'].keys())
    metrics = ['mean_reward', 'mean_queue_length', 'mean_grid_load', 'mean_completion_rate']
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Scenario Comparison Analysis', fontsize=16, y=0.95)
    
    # Plot 1: Reward Performance
    ax = axes[0, 0]
    rewards = [results[0]['evaluation'][s]['mean_reward'] for s in scenarios]
    std_rewards = [results[0]['evaluation'][s]['std_reward'] for s in scenarios]
    ax.bar(scenarios, rewards, yerr=std_rewards, capsize=5)
    ax.set_title('Mean Reward by Scenario')
    ax.set_ylabel('Mean Reward')
    ax.grid(True)
    
    # Plot 2: Queue Length Analysis
    ax = axes[0, 1]
    queue_lengths = [results[0]['evaluation'][s]['mean_queue_length'] for s in scenarios]
    ax.bar(scenarios, queue_lengths)
    ax.set_title('Queue Length by Scenario')
    ax.set_ylabel('Mean Queue Length')
    ax.grid(True)
    
    # Plot 3: Grid Load Impact
    ax = axes[1, 0]
    grid_loads = [results[0]['evaluation'][s]['mean_grid_load'] for s in scenarios]
    ax.bar(scenarios, grid_loads)
    ax.set_title('Grid Load by Scenario')
    ax.set_ylabel('Mean Grid Load')
    ax.grid(True)
    
    # Plot 4: Completion Rate
    ax = axes[1, 1]
    completion_rates = [results[0]['evaluation'][s]['mean_completion_rate'] for s in scenarios]
    ax.bar(scenarios, completion_rates)
    ax.set_title('Completion Rate by Scenario')
    ax.set_ylabel('Mean Completion Rate')
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'scenario_comparison.png'))
    plt.close()

def create_trade_off_analysis(results: List[Dict], save_path: str):
    """Create trade-off analysis plots."""
    scenarios = list(results[0]['evaluation'].keys())
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Performance Trade-offs Analysis', fontsize=16, y=0.95)
    
    # Plot 1: Queue Length vs Completion Rate
    plt.subplot(121)
    for scenario in scenarios:
        queue_length = results[0]['evaluation'][scenario]['mean_queue_length']
        completion_rate = results[0]['evaluation'][scenario]['mean_completion_rate']
        plt.scatter(queue_length, completion_rate, s=100, label=scenario)
    plt.xlabel('Mean Queue Length')
    plt.ylabel('Mean Completion Rate')
    plt.title('Queue Length vs Completion Rate')
    plt.grid(True)
    plt.legend()
    
    # Plot 2: Grid Load vs Reward
    plt.subplot(122)
    for scenario in scenarios:
        grid_load = results[0]['evaluation'][scenario]['mean_grid_load']
        reward = results[0]['evaluation'][scenario]['mean_reward']
        plt.scatter(grid_load, reward, s=100, label=scenario)
    plt.xlabel('Mean Grid Load')
    plt.ylabel('Mean Reward')
    plt.title('Grid Load vs Reward')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'trade_off_analysis.png'))
    plt.close()

def create_performance_trends(results: List[Dict], save_path: str):
    """Create performance trend plots."""
    scenarios = list(results[0]['evaluation'].keys())
    metrics = ['mean_reward', 'mean_queue_length', 'mean_grid_load', 'mean_completion_rate']
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Performance Trends Across Scenarios', fontsize=16, y=0.95)
    
    # Plot metrics across scenarios
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        values = [results[0]['evaluation'][s][metric] for s in scenarios]
        plt.plot(scenarios, values, marker='o')
        plt.title(f'{metric.replace("_", " ").title()}')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'performance_trends.png'))
    plt.close()

def main():
    # Create results directory if it doesn't exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Load results
    results_path = os.path.join(results_dir, "evaluation_results.json")
    results = load_results(results_path)
    
    # Generate visualizations
    create_scenario_comparison(results, results_dir)
    create_trade_off_analysis(results, results_dir)
    create_performance_trends(results, results_dir)
    
    print("Enhanced visualizations have been generated in the results directory.")

if __name__ == "__main__":
    main() 