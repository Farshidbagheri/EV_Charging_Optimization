from typing import Dict, Any
import os
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed
from src.environment.ev_charging_env import EVChargingEnv

# Check if MPS (Apple GPU) is available and set device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    # Enable GPU optimizations
    torch.backends.mps.enable_ddp = True
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

class TrainingCallback(BaseCallback):
    """Custom callback for logging training metrics."""
    
    def __init__(self, check_freq: int, save_path: str, verbose: int = 1):
        super(TrainingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf
    
    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
    
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Get current statistics
            mean_reward = np.mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
            mean_length = np.mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])
            
            # Log to tensorboard
            self.logger.record("rollout/ep_rew_mean", mean_reward)
            self.logger.record("rollout/ep_len_mean", mean_length)
            
            # Save best model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(os.path.join(self.save_path, "best_model"))
                
                if self.verbose > 0:
                    print(f"Saving new best model with mean reward: {mean_reward:.2f}")
        
        return True

class EVChargingAgent:
    """PPO agent for EV charging control."""
    
    def __init__(self, env):
        """Initialize the agent."""
        # Set device to MPS (Apple GPU) if available
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Enable maximum GPU utilization
        if torch.backends.mps.is_available():
            torch.backends.mps.enable_ddp = True
            torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for faster training
            torch.backends.cudnn.benchmark = True  # Enable cudnn autotuner
        
        # Create environment factory function
        def make_env():
            return EVChargingEnv(load_factor=1.0, price_factor=1.0)
        
        # Vectorize environment for parallel processing
        env = DummyVecEnv([make_env for _ in range(8)])  # Use 8 parallel environments
        
        # Define policy network architecture optimized for faster training
        policy_kwargs = {
            'net_arch': dict(
                pi=[512, 256],  # Larger network for better GPU utilization
                vf=[512, 256]
            ),
            'activation_fn': torch.nn.ReLU,
            'optimizer_class': torch.optim.Adam,
            'optimizer_kwargs': {'eps': 1e-5, 'weight_decay': 1e-5},  # Add weight decay for better stability
            'normalize_images': False
        }
        
        # Initialize PPO agent with optimized parameters
        self.model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=5e-4,  # Increased for faster learning
            n_steps=2048,  # Increased for better parallelization
            batch_size=512,  # Increased for better GPU utilization
            n_epochs=5,  # Reduced epochs but increased batch size
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            normalize_advantage=True,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_sde=False,
            sde_sample_freq=-1,
            target_kl=None,
            tensorboard_log="logs/",
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=None,
            device=device,
            _init_setup_model=True
        )
    
    def train(self, total_timesteps: int, log_path: str = None):
        """Train the agent with maximum resource utilization."""
        # Create callback for saving best model
        checkpoint_callback = CheckpointCallback(
            save_freq=5000,  # Save less frequently to reduce I/O overhead
            save_path=log_path,
            name_prefix="ppo_model",
            save_replay_buffer=True,
            save_vecnormalize=True
        )
        
        # Train the agent with maximum resource utilization
        try:
            self.model.learn(
                total_timesteps=total_timesteps,
                callback=checkpoint_callback,
                progress_bar=True,  # Enable progress bar for monitoring
                reset_num_timesteps=False,  # Continue training from previous state
                tb_log_name="PPO"  # TensorBoard logging
            )
        except Exception as e:
            print(f"Training error occurred: {e}")
            print("Training failed. Please check the error message above.")
            raise  # Re-raise the exception to handle it in the calling code
    
    def predict(self, observation, deterministic=True):
        """Make a prediction based on the current observation."""
        return self.model.predict(observation, deterministic=deterministic)
    
    def save(self, path: str):
        """Save the agent to the specified path."""
        self.model.save(path)
    
    def load(self, path: str):
        """Load the agent from the specified path."""
        self.model = PPO.load(path)
    
    def evaluate(self, n_episodes: int = 10) -> Dict[str, float]:
        """Evaluate the agent's performance."""
        rewards = []
        episode_lengths = []
        
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                action, _ = self.predict(obs)
                obs, reward, done, _, _ = self.env.step(action)
                episode_reward += reward
                episode_length += 1
            
            rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        return {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "mean_episode_length": np.mean(episode_lengths),
            "std_episode_length": np.std(episode_lengths)
        } 