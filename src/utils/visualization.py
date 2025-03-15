import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

def setup_plot_style():
    """Set up the plotting style."""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = [12, 6]
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12

def plot_training_performance(data, save_path):
    """Plot training performance metrics."""
    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Training reward over episodes
    episodes = np.arange(len(data['rewards']))
    ax1.plot(episodes, data['rewards'], label='Episode Reward')
    ax1.fill_between(episodes,
                    data['rewards'] - data['reward_std'],
                    data['rewards'] + data['reward_std'],
                    alpha=0.3)
    ax1.set_title('Training Progress')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Average Reward')
    ax1.grid(True)
    
    # Learning curves comparison
    ax2.plot(episodes, data['rl_performance'], label='RL Strategy')
    ax2.plot(episodes, data['baseline_performance'], label='Static Strategy')
    ax2.set_title('RL vs. Static Strategy')
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Performance Score')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_charging_strategies(data, save_path):
    """Plot comparison of different charging strategies."""
    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Strategy comparison
    strategies = ['RL Strategy', 'Static Strategy', 'Random Strategy']
    metrics = [data['rl_efficiency'], data['static_efficiency'], data['random_efficiency']]
    ax1.bar(strategies, metrics)
    ax1.set_title('Charging Strategy Comparison')
    ax1.set_ylabel('Efficiency (%)')
    ax1.grid(True)
    
    # Hourly performance
    hours = np.arange(24)
    ax2.plot(hours, data['rl_hourly'], label='RL')
    ax2.plot(hours, data['static_hourly'], label='Static')
    ax2.set_title('Hourly Performance')
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Average Throughput')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_queue_analysis(data, save_path):
    """Plot queue length and wait time analysis."""
    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Queue length distribution
    sns.histplot(data=data['queue_lengths'], ax=ax1, bins=30)
    ax1.axvline(data['queue_lengths'].mean(), color='r', linestyle='--',
                label=f'Mean: {data["queue_lengths"].mean():.2f}')
    ax1.set_title('Queue Length Distribution')
    ax1.set_xlabel('Number of Vehicles')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    
    # Wait time analysis
    sns.boxplot(data=data['wait_times_by_hour'], ax=ax2)
    ax2.set_title('Wait Times by Hour')
    ax2.set_xlabel('Hour of Day')
    ax2.set_ylabel('Wait Time (minutes)')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_economic_impact(data, save_path):
    """Plot economic impact and cost savings."""
    setup_plot_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Cost savings over time
    ax1.plot(data['timeline'], data['cumulative_savings'])
    ax1.set_title('Cumulative Cost Savings')
    ax1.set_xlabel('Days')
    ax1.set_ylabel('Total Savings ($)')
    ax1.grid(True)
    
    # ROI metrics
    metrics = ['Energy Cost', 'Peak Demand', 'Operation Cost']
    savings = [data['energy_savings'], data['peak_savings'], data['op_savings']]
    ax2.bar(metrics, savings)
    ax2.set_title('Cost Reduction by Category')
    ax2.set_ylabel('Reduction (%)')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_sample_data():
    """Generate sample data for visualization."""
    np.random.seed(42)
    n_episodes = 100
    n_hours = 24
    
    data = {
        # Training performance data
        'rewards': np.cumsum(np.random.normal(0.5, 0.1, n_episodes)),
        'reward_std': np.full(n_episodes, 0.2),
        'rl_performance': np.cumsum(np.random.normal(0.6, 0.1, n_episodes)),
        'baseline_performance': np.cumsum(np.random.normal(0.4, 0.1, n_episodes)),
        
        # Strategy comparison data
        'rl_efficiency': 85.5,
        'static_efficiency': 70.2,
        'random_efficiency': 55.8,
        'rl_hourly': 70 + np.random.normal(0, 5, n_hours),
        'static_hourly': 60 + np.random.normal(0, 5, n_hours),
        
        # Queue analysis data
        'queue_lengths': np.random.normal(5, 2, 1000),
        'wait_times_by_hour': pd.DataFrame(np.random.normal(10, 3, (100, n_hours))),
        
        # Economic impact data
        'timeline': np.arange(30),
        'cumulative_savings': np.cumsum(np.random.normal(100, 10, 30)),
        'energy_savings': 23.5,
        'peak_savings': 15.8,
        'op_savings': 18.2
    }
    
    return data

def generate_all_plots(results_dir='results'):
    """Generate all visualization plots."""
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)
    
    # Generate sample data
    data = generate_sample_data()
    
    # Generate plots
    plot_training_performance(data, results_dir / 'training_performance.png')
    plot_charging_strategies(data, results_dir / 'charging_strategies.png')
    plot_queue_analysis(data, results_dir / 'queue_analysis.png')
    plot_economic_impact(data, results_dir / 'economic_impact.png')

if __name__ == "__main__":
    generate_all_plots() 