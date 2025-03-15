import os
import json
from pathlib import Path
import numpy as np
import gymnasium as gym
from src.environment.ev_charging_env import EVChargingEnv
from src.agents.ppo_agent import EVChargingAgent
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import pandas as pd

def setup_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

def plot_training_results(env: EVChargingEnv, agent: EVChargingAgent, save_path: str = "results/training_evaluation.png"):
    """Plot comprehensive training results and agent behavior."""
    # Create subplots
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))
    fig.suptitle("EV Charging Agent Evaluation", fontsize=16)
    
    # Run an episode for visualization
    obs, _ = env.reset()
    done = False
    episode_data = []
    
    while not done:
        action = agent.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        
        # Collect episode data
        episode_data.append({
            'time': env.current_time,
            'active_sessions': len(env.active_sessions),
            'queue_length': len(env.waiting_queue),
            'grid_load': env.current_load,
            'price': env.electricity_price,
            'reward': reward
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(episode_data)
    
    # Plot battery SoC for each active session
    ax = axes[0, 0]
    for session in env.active_sessions:
        ax.plot(session.battery_soc, label=f'Session {session.id}')
    ax.set_title('Battery State of Charge')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('SoC')
    ax.grid(True)
    ax.legend()
    
    # Plot charging rates
    ax = axes[0, 1]
    for session in env.active_sessions:
        ax.plot(session.charging_rate, label=f'Session {session.id}')
    ax.set_title('Charging Rates')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Rate')
    ax.grid(True)
    ax.legend()
    
    # Plot queue length and active sessions
    ax = axes[1, 0]
    ax.plot(df['queue_length'], label='Queue Length')
    ax.plot(df['active_sessions'], label='Active Sessions')
    ax.set_title('Queue and Active Sessions')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Count')
    ax.grid(True)
    ax.legend()
    
    # Plot grid load and price
    ax = axes[1, 1]
    ax.plot(df['grid_load'], label='Grid Load')
    ax.plot(df['price'], label='Price')
    ax.set_title('Grid Load and Price')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Value')
    ax.grid(True)
    ax.legend()
    
    # Plot cumulative reward
    ax = axes[2, 0]
    cumulative_rewards = df['reward'].cumsum()
    ax.plot(cumulative_rewards, label='Cumulative Reward')
    ax.set_title('Cumulative Reward')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Reward')
    ax.grid(True)
    ax.legend()
    
    # Plot session completion times
    ax = axes[2, 1]
    completion_times = [s.completion_time for s in env.completed_sessions if s.completion_time is not None]
    if completion_times:
        ax.hist(completion_times, bins=20, label='Completion Times')
    ax.set_title('Session Completion Times')
    ax.set_xlabel('Time')
    ax.set_ylabel('Count')
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def evaluate_agent(env: EVChargingEnv, agent: EVChargingAgent, n_episodes: int = 10) -> Dict[str, float]:
    """Evaluate the agent's performance with comprehensive metrics."""
    metrics = {
        'mean_reward': [],
        'mean_completion_time': [],
        'mean_queue_length': [],
        'mean_grid_load': [],
        'mean_price': [],
        'completion_rate': []
    }
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_rewards = []
        episode_queue_lengths = []
        episode_grid_loads = []
        episode_prices = []
        
        while not done:
            action = agent.predict(obs)
            obs, reward, done, _, _ = env.step(action)
            
            episode_rewards.append(reward)
            episode_queue_lengths.append(len(env.waiting_queue))
            episode_grid_loads.append(env.current_load)
            episode_prices.append(env.electricity_price)
        
        # Calculate episode metrics
        metrics['mean_reward'].append(np.mean(episode_rewards))
        metrics['mean_queue_length'].append(np.mean(episode_queue_lengths))
        metrics['mean_grid_load'].append(np.mean(episode_grid_loads))
        metrics['mean_price'].append(np.mean(episode_prices))
        
        # Calculate completion metrics
        completion_times = [s.completion_time for s in env.completed_sessions if s.completion_time is not None]
        if completion_times:
            metrics['mean_completion_time'].append(np.mean(completion_times))
        
        # Calculate completion rate
        total_sessions = len(env.completed_sessions) + len(env.waiting_queue)
        if total_sessions > 0:
            metrics['completion_rate'].append(len(env.completed_sessions) / total_sessions)
    
    # Calculate final metrics
    final_metrics = {
        'mean_reward': np.mean(metrics['mean_reward']),
        'std_reward': np.std(metrics['mean_reward']),
        'mean_completion_time': np.mean(metrics['mean_completion_time']) if metrics['mean_completion_time'] else 0,
        'std_completion_time': np.std(metrics['mean_completion_time']) if metrics['mean_completion_time'] else 0,
        'mean_queue_length': np.mean(metrics['mean_queue_length']),
        'std_queue_length': np.std(metrics['mean_queue_length']),
        'mean_grid_load': np.mean(metrics['mean_grid_load']),
        'std_grid_load': np.std(metrics['mean_grid_load']),
        'mean_price': np.mean(metrics['mean_price']),
        'std_price': np.std(metrics['mean_price']),
        'completion_rate': np.mean(metrics['completion_rate']) if metrics['completion_rate'] else 0,
        'std_completion_rate': np.std(metrics['completion_rate']) if metrics['completion_rate'] else 0
    }
    
    return final_metrics

def main():
    """Main training function."""
    # Setup directories
    setup_directories()
    
    print("Starting training with optimized settings...")
    print("Using parallel environments and increased batch size for maximum GPU/CPU utilization")
    
    # Set up environment with optimized parameters
    env = EVChargingEnv(load_factor=1.0, price_factor=1.0)
    
    # Initialize agent
    agent = EVChargingAgent(env)
    
    # Train with progress updates
    total_timesteps = 50000
    print(f"\nTraining for {total_timesteps} timesteps...")
    print("Training will be faster with parallel environments and optimized batch size")
    
    try:
        agent.train(
            total_timesteps=total_timesteps,
            log_path="models"
        )
        print("\nTraining completed successfully!")
        
        # Save final model
        agent.save("models/final_model")
        print("Model saved successfully!")
        
        # Evaluate agent
        print("\nEvaluating agent performance...")
        eval_results = evaluate_agent(env, agent)
        
        # Save evaluation results
        results = {
            'observation_space': {
                'shape': env.observation_space.shape,
                'features': [
                    "queue_length",
                    "available_stations",
                    "current_load",
                    "electricity_price",
                    "time_of_day"
                ]
            },
            'evaluation': eval_results
        }
        
        with open('results/evaluation_metrics.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        print("\nResults saved to results/evaluation_metrics.json")
        print("\nTraining and evaluation completed successfully!")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        raise

if __name__ == "__main__":
    main() 