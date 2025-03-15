import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

def normalize_data(data: np.ndarray, min_val: float = None, max_val: float = None) -> Tuple[np.ndarray, float, float]:
    """
    Normalize data to range [0, 1]
    
    Args:
        data: Input data array
        min_val: Optional minimum value for normalization
        max_val: Optional maximum value for normalization
    
    Returns:
        Tuple of (normalized_data, min_val, max_val)
    """
    if min_val is None:
        min_val = np.min(data)
    if max_val is None:
        max_val = np.max(data)
    
    normalized = (data - min_val) / (max_val - min_val)
    return normalized, min_val, max_val

def plot_training_history(
    rewards: List[float],
    avg_window: int = 100,
    title: str = "Training History"
) -> None:
    """
    Plot training rewards history
    
    Args:
        rewards: List of episode rewards
        avg_window: Window size for moving average
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, alpha=0.3, label='Raw Rewards')
    
    # Calculate moving average
    if len(rewards) >= avg_window:
        moving_avg = np.convolve(rewards, np.ones(avg_window)/avg_window, mode='valid')
        plt.plot(range(avg_window-1, len(rewards)), moving_avg, label=f'{avg_window}-Episode Moving Average')
    
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.show()

def calculate_price_factors(
    time_of_day: float,
    current_load: float,
    base_price: float = 0.5
) -> Dict[str, float]:
    """
    Calculate various price factors for electricity
    
    Args:
        time_of_day: Hour of the day (0-24)
        current_load: Current grid load (0-1)
        base_price: Base electricity price
    
    Returns:
        Dictionary containing price factors
    """
    # Time of day factor
    peak_hours = (14 <= time_of_day <= 20)
    time_factor = 2.0 if peak_hours else 1.0
    
    # Load factor
    load_factor = 1 + 0.2 * current_load
    
    # Calculate final price
    final_price = min(1.0, base_price * time_factor * load_factor)
    
    return {
        'base_price': base_price,
        'time_factor': time_factor,
        'load_factor': load_factor,
        'final_price': final_price
    }

def create_time_features(time_of_day: float) -> Dict[str, float]:
    """
    Create cyclical time features
    
    Args:
        time_of_day: Hour of the day (0-24)
    
    Returns:
        Dictionary containing sin and cos features
    """
    # Convert time to radians
    time_rad = (time_of_day * 2 * np.pi) / 24
    
    return {
        'time_sin': np.sin(time_rad),
        'time_cos': np.cos(time_rad)
    } 