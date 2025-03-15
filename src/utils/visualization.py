import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib.gridspec import GridSpec

def setup_plot_style():
    """Set up the plotting style for professional visualizations."""
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = [15, 8]
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = '#f8f9fa'

def plot_comprehensive_training(data, save_path):
    """Plot comprehensive training metrics with advanced visualizations."""
    setup_plot_style()
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig)
    
    # 1. Training Progress with Multiple Metrics
    ax1 = fig.add_subplot(gs[0, :2])
    episodes = np.arange(len(data['rewards']))
    ax1.plot(episodes, data['rewards'], label='Our Method', color='#2ecc71', linewidth=2)
    ax1.fill_between(episodes, 
                    data['rewards'] - data['reward_std'],
                    data['rewards'] + data['reward_std'],
                    color='#2ecc71', alpha=0.2)
    
    # Add competitor methods
    ax1.plot(episodes, data['competitor1_rewards'], label='DDPG (2023)', 
             color='#e74c3c', linestyle='--', linewidth=2)
    ax1.plot(episodes, data['competitor2_rewards'], label='SAC (2024)', 
             color='#3498db', linestyle='-.', linewidth=2)
    ax1.set_title('Training Progress Comparison', pad=20)
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Average Reward')
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax1.grid(True, alpha=0.3)
    
    # 2. Performance Metrics Radar Chart
    ax2 = fig.add_subplot(gs[0, 2], projection='polar')
    metrics = ['Charging Rate', 'Grid Stability', 'Wait Time', 
              'Cost Efficiency', 'Energy Usage', 'Peak Reduction']
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
    
    # Add the last value to close the polygon
    values = [data['metrics'][m] for m in metrics]
    values += values[:1]
    angles = np.concatenate((angles, [angles[0]]))
    
    ax2.plot(angles, values, 'o-', linewidth=2, label='Our Method')
    ax2.fill(angles, values, alpha=0.25)
    ax2.set_thetagrids(angles[:-1] * 180/np.pi, metrics)
    ax2.set_title('Performance Metrics', pad=20)
    
    # 3. Comparative Analysis
    ax3 = fig.add_subplot(gs[1, 0])
    methods = ['Our Method', 'DDPG (2023)', 'SAC (2024)', 'Rule-based']
    metrics = [data['completion_rate'], data['competitor1_rate'],
              data['competitor2_rate'], data['baseline_rate']]
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#95a5a6']
    
    bars = ax3.bar(methods, metrics, color=colors)
    ax3.set_title('Charging Completion Rate')
    ax3.set_ylabel('Completion Rate (%)')
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # 4. Economic Impact
    ax4 = fig.add_subplot(gs[1, 1])
    timeline = np.arange(len(data['cumulative_savings']))
    ax4.plot(timeline, data['cumulative_savings'], 
             label='Our Method', color='#2ecc71', linewidth=2)
    ax4.plot(timeline, data['competitor_savings'], 
             label='Best Competitor', color='#e74c3c', 
             linestyle='--', linewidth=2)
    ax4.set_title('Cumulative Cost Savings')
    ax4.set_xlabel('Days')
    ax4.set_ylabel('Savings ($)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Queue Management Efficiency
    ax5 = fig.add_subplot(gs[1, 2])
    sns.violinplot(data=[data['our_wait_times'], 
                        data['competitor1_wait_times'],
                        data['competitor2_wait_times']], 
                  ax=ax5)
    ax5.set_title('Wait Time Distribution')
    ax5.set_ylabel('Wait Time (minutes)')
    ax5.set_xticks([0, 1, 2])
    ax5.set_xticklabels(['Our Method', 'DDPG', 'SAC'])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_detailed_analysis(data, save_path):
    """Plot detailed analysis of system performance."""
    setup_plot_style()
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 2, figure=fig)
    
    # 1. Hourly Load Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    hours = np.arange(24)
    ax1.plot(hours, data['our_load'], label='Our Method', 
             color='#2ecc71', linewidth=2)
    ax1.plot(hours, data['baseline_load'], label='Baseline', 
             color='#95a5a6', linestyle='--', linewidth=2)
    ax1.fill_between(hours, data['grid_capacity'], 
                    color='#e74c3c', alpha=0.2, 
                    label='Grid Capacity')
    ax1.set_title('24-Hour Load Profile')
    ax1.set_xlabel('Hour of Day')
    ax1.set_ylabel('Grid Load (kW)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Price vs. Demand Correlation
    ax2 = fig.add_subplot(gs[0, 1])
    sns.scatterplot(data=pd.DataFrame({
        'Price': data['price_points'],
        'Demand': data['demand_points'],
        'Strategy': data['strategy_labels']
    }), x='Price', y='Demand', hue='Strategy', ax=ax2)
    ax2.set_title('Price-Demand Relationship')
    ax2.set_xlabel('Price ($/kWh)')
    ax2.set_ylabel('Demand (kW)')
    
    # 3. System Stability Metrics
    ax3 = fig.add_subplot(gs[1, 0])
    stability_metrics = ['Voltage\nStability', 'Frequency\nResponse', 
                        'Power\nQuality', 'Load\nBalance']
    our_stability = [data['stability'][m] for m in stability_metrics]
    competitor_stability = [data['competitor_stability'][m] 
                          for m in stability_metrics]
    
    x = np.arange(len(stability_metrics))
    width = 0.35
    ax3.bar(x - width/2, our_stability, width, label='Our Method', 
            color='#2ecc71')
    ax3.bar(x + width/2, competitor_stability, width, 
            label='Best Competitor', color='#e74c3c')
    ax3.set_title('System Stability Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(stability_metrics)
    ax3.legend()
    
    # 4. Multi-objective Optimization Results
    ax4 = fig.add_subplot(gs[1, 1])
    sns.scatterplot(data=pd.DataFrame({
        'Cost': data['pareto_cost'],
        'Efficiency': data['pareto_efficiency'],
        'Method': data['pareto_labels']
    }), x='Cost', y='Efficiency', hue='Method', 
    style='Method', s=100, ax=ax4)
    ax4.set_title('Pareto Frontier: Cost vs. Efficiency')
    ax4.set_xlabel('Operational Cost ($)')
    ax4.set_ylabel('System Efficiency (%)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_sample_data():
    """Generate comprehensive sample data for visualization."""
    np.random.seed(42)
    n_episodes = 100
    n_hours = 24
    n_points = 200
    
    data = {
        # Training performance data
        'rewards': np.cumsum(np.random.normal(0.5, 0.1, n_episodes)),
        'reward_std': np.full(n_episodes, 0.2),
        'competitor1_rewards': np.cumsum(np.random.normal(0.3, 0.1, n_episodes)),
        'competitor2_rewards': np.cumsum(np.random.normal(0.4, 0.1, n_episodes)),
        
        # Performance metrics
        'metrics': {
            'Charging Rate': 0.85,
            'Grid Stability': 0.92,
            'Wait Time': 0.78,
            'Cost Efficiency': 0.88,
            'Energy Usage': 0.83,
            'Peak Reduction': 0.90
        },
        
        # Completion rates
        'completion_rate': 85.5,
        'competitor1_rate': 70.2,
        'competitor2_rate': 75.8,
        'baseline_rate': 60.5,
        
        # Economic data
        'cumulative_savings': np.cumsum(np.random.normal(100, 10, 30)),
        'competitor_savings': np.cumsum(np.random.normal(80, 10, 30)),
        
        # Wait times
        'our_wait_times': np.random.normal(5, 1, 100),
        'competitor1_wait_times': np.random.normal(8, 2, 100),
        'competitor2_wait_times': np.random.normal(7, 1.5, 100),
        
        # Load profiles
        'our_load': 500 + 300 * np.sin(np.linspace(0, 2*np.pi, 24)),
        'baseline_load': 600 + 400 * np.sin(np.linspace(0, 2*np.pi, 24)),
        'grid_capacity': np.full(24, 1000),
        
        # Price-demand data
        'price_points': np.random.uniform(0.1, 0.5, n_points),
        'demand_points': np.random.uniform(300, 800, n_points),
        'strategy_labels': np.random.choice(['Our Method', 'DDPG', 'SAC'], n_points),
        
        # Stability metrics
        'stability': {
            'Voltage\nStability': 0.92,
            'Frequency\nResponse': 0.88,
            'Power\nQuality': 0.90,
            'Load\nBalance': 0.85
        },
        'competitor_stability': {
            'Voltage\nStability': 0.85,
            'Frequency\nResponse': 0.82,
            'Power\nQuality': 0.84,
            'Load\nBalance': 0.80
        },
        
        # Pareto frontier data
        'pareto_cost': np.concatenate([
            np.random.uniform(1000, 1500, 30),
            np.random.uniform(1200, 1800, 30),
            np.random.uniform(1400, 2000, 30)
        ]),
        'pareto_efficiency': np.concatenate([
            np.random.uniform(80, 95, 30),
            np.random.uniform(70, 85, 30),
            np.random.uniform(60, 75, 30)
        ]),
        'pareto_labels': np.repeat(['Our Method', 'DDPG', 'SAC'], 30)
    }
    
    return data

def generate_all_plots(results_dir='results'):
    """Generate all visualization plots."""
    results_dir = Path(results_dir)
    results_dir.mkdir(exist_ok=True)
    
    # Generate sample data
    data = generate_sample_data()
    
    # Generate plots
    plot_comprehensive_training(data, results_dir / 'comprehensive_training.png')
    plot_detailed_analysis(data, results_dir / 'detailed_analysis.png')

if __name__ == "__main__":
    generate_all_plots() 