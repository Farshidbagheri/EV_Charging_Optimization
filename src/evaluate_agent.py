import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from datetime import datetime
from environment.ev_charging_env import EVChargingEnv
from agents.ppo_agent import EVChargingAgent
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
import multiprocessing
from stable_baselines3.common.monitor import Monitor
import sys

def make_env(rank, env_params):
    """Helper function to create environments for parallel processing."""
    def _init():
        env = EVChargingEnv(**env_params)
        env.reset(seed=rank)
        return env
    set_random_seed(rank)
    return _init

class EVChargingEvaluator:
    def __init__(self, 
                 num_sessions: int = 3,
                 timesteps_per_session: int = 5000,
                 eval_freq: int = 8000,
                 eval_episodes: int = 5,
                 num_envs: int = 8):
        """Initialize the evaluator."""
        self.num_sessions = num_sessions
        self.timesteps_per_session = timesteps_per_session
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.num_envs = num_envs
        
        # Environment parameters
        self.env_params = {
            'load_factor': 1.0,
            'price_factor': 1.0
        }
        
        # Create results directory
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)
    
    def make_env(self, rank: int, env_params: Dict = None):
        """Create a vectorized environment."""
        def _init():
            env = EVChargingEnv(**env_params if env_params else self.env_params)
            env = Monitor(env)
            return env
        return _init
    
    def run_training_sessions(self) -> List[Dict]:
        """Run multiple training sessions and evaluate performance."""
        training_results = []
        
        for session in range(self.num_sessions):
            print(f"\nTraining Session {session + 1}/{self.num_sessions}")
            
            # Create vectorized environment for training
            env = SubprocVecEnv([self.make_env(i) for i in range(self.num_envs)])
            
            # Initialize agent
            agent = EVChargingAgent(env)
            
            # Train agent
            log_path = os.path.join(self.results_dir, f"session_{session}")
            agent.train(
                total_timesteps=self.timesteps_per_session,
                log_path=log_path
            )
            
            # Evaluate agent across different scenarios
            evaluation_results = self.evaluate_agent(agent)
            
            # Store results
            training_results.append({
                'session_id': session,
                'evaluation': evaluation_results
            })
            
            env.close()
        
        return training_results
    
    def evaluate_agent(self, agent) -> Dict:
        """Evaluate agent performance across different scenarios."""
        scenarios = {
            'normal': {'load_factor': 1.0, 'price_factor': 1.0},
            'high_load': {'load_factor': 1.5, 'price_factor': 1.2},
            'off_peak': {'load_factor': 0.7, 'price_factor': 0.8},
            'unpredictable': {'load_factor': 1.3, 'price_factor': 1.1}
        }
        
        results = {}
        
        for scenario_name, scenario_params in scenarios.items():
            print(f"\nEvaluating {scenario_name} scenario...")
            
            # Create evaluation environment with scenario parameters
            eval_env = EVChargingEnv(**scenario_params)
            
            # Run evaluation episodes
            episode_rewards = []
            episode_lengths = []
            episode_queue_lengths = []
            episode_grid_loads = []
            episode_prices = []
            episode_completions = []
            
            for _ in range(self.eval_episodes):
                state, _ = eval_env.reset()
                done = False
                episode_reward = 0
                episode_length = 0
                episode_queue = []
                episode_load = []
                episode_price = []
                episode_complete = 0
                
                while not done:
                    action = agent.predict(state, deterministic=True)[0]
                    state, reward, done, truncated, info = eval_env.step(action)
                    episode_reward += reward
                    episode_length += 1
                    episode_queue.append(info.get('queue_length', 0))
                    episode_load.append(info.get('grid_load', 0))
                    episode_price.append(info.get('price', 0))
                    episode_complete += info.get('completed_charges', 0)
                
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                episode_queue_lengths.append(np.mean(episode_queue))
                episode_grid_loads.append(np.mean(episode_load))
                episode_prices.append(np.mean(episode_price))
                episode_completions.append(episode_complete)
            
            # Store results for this scenario
            results[scenario_name] = {
                'mean_reward': np.mean(episode_rewards),
                'std_reward': np.std(episode_rewards),
                'mean_length': np.mean(episode_lengths),
                'mean_queue_length': np.mean(episode_queue_lengths),
                'mean_grid_load': np.mean(episode_grid_loads),
                'mean_price': np.mean(episode_prices),
                'mean_completion_rate': np.mean(episode_completions) / np.mean(episode_lengths)
            }
            
            eval_env.close()
        
        return results
    
    def plot_results(self, training_results: List[Dict]):
        """Generate comprehensive visualization of results."""
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 2)
        
        # Plot 1: Training Progress
        ax1 = fig.add_subplot(gs[0, 0])
        for session in training_results:
            rewards = [session['evaluation']['normal']['mean_reward']]
            ax1.plot(rewards, label=f'Session {session["session_id"]}')
        ax1.set_title('Training Progress (Normal Scenario)')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Mean Reward')
        ax1.grid(True)
        ax1.legend()
        
        # Plot 2: Scenario Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        scenarios = list(training_results[0]['evaluation'].keys())
        metrics = ['mean_reward', 'mean_queue_length', 'mean_grid_load', 'mean_completion_rate']
        
        x = np.arange(len(scenarios))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = [training_results[-1]['evaluation'][s][metric] for s in scenarios]
            ax2.bar(x + i*width, values, width, label=metric)
        
        ax2.set_title('Performance Across Scenarios')
        ax2.set_xticks(x + width*1.5)
        ax2.set_xticklabels(scenarios)
        ax2.legend()
        
        # Plot 3: Learning Stability
        ax3 = fig.add_subplot(gs[1, 0])
        for session in training_results:
            std_rewards = [session['evaluation'][s]['std_reward'] for s in scenarios]
            ax3.plot(std_rewards, label=f'Session {session["session_id"]}')
        ax3.set_title('Learning Stability')
        ax3.set_xlabel('Scenario')
        ax3.set_ylabel('Reward Standard Deviation')
        ax3.grid(True)
        ax3.legend()
        
        # Plot 4: Grid Impact
        ax4 = fig.add_subplot(gs[1, 1])
        for scenario in scenarios:
            grid_loads = [session['evaluation'][scenario]['mean_grid_load'] for session in training_results]
            prices = [session['evaluation'][scenario]['mean_price'] for session in training_results]
            ax4.scatter(grid_loads, prices, label=scenario, s=100)
        ax4.set_title('Grid Load vs Price')
        ax4.set_xlabel('Grid Load')
        ax4.set_ylabel('Price')
        ax4.grid(True)
        ax4.legend()
        
        # Plot 5: Queue Management
        ax5 = fig.add_subplot(gs[2, 0])
        for scenario in scenarios:
            queue_lengths = [session['evaluation'][scenario]['mean_queue_length'] for session in training_results]
            completion_rates = [session['evaluation'][scenario]['mean_completion_rate'] for session in training_results]
            ax5.scatter(queue_lengths, completion_rates, label=scenario, s=100)
        ax5.set_title('Queue Length vs Completion Rate')
        ax5.set_xlabel('Queue Length')
        ax5.set_ylabel('Completion Rate')
        ax5.grid(True)
        ax5.legend()
        
        # Plot 6: Cost Efficiency
        ax6 = fig.add_subplot(gs[2, 1])
        for scenario in scenarios:
            rewards = [session['evaluation'][scenario]['mean_reward'] for session in training_results]
            prices = [session['evaluation'][scenario]['mean_price'] for session in training_results]
            ax6.scatter(prices, rewards, label=scenario, s=100)
        ax6.set_title('Price vs Reward')
        ax6.set_xlabel('Price')
        ax6.set_ylabel('Reward')
        ax6.grid(True)
        ax6.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, "comprehensive_evaluation.png"))
        plt.close()
    
    def save_results(self, training_results: List[Dict]):
        """Save evaluation results to JSON file."""
        results_file = os.path.join(self.results_dir, "evaluation_results.json")
        with open(results_file, 'w') as f:
            json.dump(training_results, f, indent=2)
    
    def print_summary(self, training_results: List[Dict]):
        """Print summary of evaluation results."""
        print("\nEvaluation Summary:")
        print("=" * 50)
        
        # Calculate average metrics across all sessions
        avg_metrics = {
            'mean_reward': np.mean([r['evaluation']['normal']['mean_reward'] for r in training_results]),
            'mean_queue_length': np.mean([r['evaluation']['normal']['mean_queue_length'] for r in training_results]),
            'mean_completion_rate': np.mean([r['evaluation']['normal']['mean_completion_rate'] for r in training_results])
        }
        
        print(f"\nAverage Performance (Normal Scenario):")
        print(f"Mean Reward: {avg_metrics['mean_reward']:.2f}")
        print(f"Mean Queue Length: {avg_metrics['mean_queue_length']:.2f}")
        print(f"Mean Completion Rate: {avg_metrics['mean_completion_rate']:.2f}")
        
        print("\nScenario Comparison:")
        for scenario in training_results[0]['evaluation'].keys():
            print(f"\n{scenario.upper()}:")
            metrics = training_results[0]['evaluation'][scenario]
            print(f"  Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
            print(f"  Mean Queue Length: {metrics['mean_queue_length']:.2f} ± {metrics['mean_length']:.2f}")
            print(f"  Mean Grid Load: {metrics['mean_grid_load']:.2f} ± {metrics['mean_grid_load']:.2f}")
            print(f"  Mean Completion Rate: {metrics['mean_completion_rate']:.2f} ± {metrics['mean_completion_rate']:.2f}")

def main():
    # Get number of CPU cores and set process priority
    num_cores = multiprocessing.cpu_count()
    print(f"Using {num_cores} CPU cores")
    
    # Set process priority to maximum
    try:
        import psutil
        process = psutil.Process()
        if sys.platform == 'darwin':  # macOS
            process.nice(-20)  # Highest priority
    except Exception as e:
        print(f"Could not set process priority: {e}")
    
    # Initialize evaluator with faster training parameters
    evaluator = EVChargingEvaluator(
        num_sessions=2,  # Reduced number of sessions
        timesteps_per_session=10000,  # Reduced timesteps
        eval_freq=1000,
        eval_episodes=5,  # Reduced evaluation episodes
        num_envs=num_cores  # Use all available CPU cores
    )
    
    # Set multiprocessing start method to 'spawn' for better stability
    if sys.platform == 'darwin':  # macOS
        multiprocessing.set_start_method('spawn')
    
    # Run training sessions with error handling
    print("Starting training sessions with optimized parameters...")
    try:
        training_results = evaluator.run_training_sessions()
        
        # Generate visualizations
        print("\nGenerating visualizations...")
        evaluator.plot_results(training_results)
        
        # Save results
        print("\nSaving results...")
        evaluator.save_results(training_results)
        
        # Print summary
        evaluator.print_summary(training_results)
        
        print("\nTraining completed successfully!")
        
    except Exception as e:
        print(f"Error occurred during training: {e}")
        print("Training interrupted.")

if __name__ == "__main__":
    main() 