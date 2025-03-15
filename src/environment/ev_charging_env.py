import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class ChargingSessionStatus(Enum):
    WAITING = 0
    CHARGING = 1
    COMPLETED = 2

@dataclass
class ChargingSession:
    id: int
    battery_soc: float
    target_soc: float
    arrival_time: float
    status: ChargingSessionStatus
    charging_rate: float = 0.0
    completion_time: Optional[float] = None

class EVChargingEnv(gym.Env):
    """
    Enhanced Electric Vehicle Charging Environment
    
    This environment simulates an EV charging station with multiple charging sessions,
    queue management, and grid load balancing.
    """
    
    def __init__(self, load_factor=1.0, price_factor=1.0):
        super().__init__()
        
        # Environment parameters
        self.max_queue_length = 20
        self.max_charging_time = 8  # hours
        self.num_charging_stations = 10
        self.max_power_per_station = 50  # kW
        self.base_load = 500  # kW
        self.load_factor = load_factor
        self.price_factor = price_factor
        
        # State space: [queue_length, available_stations, current_load, electricity_price, time_of_day]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0]),
            high=np.array([self.max_queue_length, self.num_charging_stations, 
                          self.base_load * 2, 1.0, 24]),
            dtype=np.float32
        )
        
        # Action space: [num_vehicles_to_charge]
        self.action_space = spaces.Discrete(self.num_charging_stations + 1)
        
        # Initialize state variables
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset state variables
        self.time_step = 0
        self.waiting_queue = []
        self.charging_stations = [None] * self.num_charging_stations
        self.completed_sessions = []
        self.current_load = self.base_load
        self.electricity_price = self._calculate_electricity_price()
        
        # Generate initial state
        observation = self._get_observation()
        
        return observation, {}
    
    def step(self, action):
        """Execute one time step within the environment."""
        # Update time
        self.time_step = (self.time_step + 1) % 24
        
        # Process action
        if isinstance(action, tuple):
            action = action[0]  # Extract the action from the tuple
        num_to_charge = int(action)  # Convert action to integer
        reward = 0
        
        # Update charging stations
        completed_charges = 0
        for i in range(len(self.charging_stations)):
            if self.charging_stations[i] is not None:
                session = self.charging_stations[i]
                # Update charging progress
                session.charging_rate = self.max_power_per_station
                session.battery_soc = min(1.0, session.battery_soc + 0.1)  # Simplified charging model
                
                if session.battery_soc >= session.target_soc:
                    # Charging complete
                    session.status = ChargingSessionStatus.COMPLETED
                    session.completion_time = self.time_step
                    self.completed_sessions.append(session)
                    self.charging_stations[i] = None
                    completed_charges += 1
        
        # Start new charging sessions
        available_stations = self.charging_stations.count(None)
        num_to_charge = min(num_to_charge, len(self.waiting_queue), available_stations)
        
        for _ in range(num_to_charge):
            # Find empty station
            station_idx = self.charging_stations.index(None)
            # Start charging
            session = self.waiting_queue.pop(0)
            session.status = ChargingSessionStatus.CHARGING
            self.charging_stations[station_idx] = session
        
        # Add new vehicles to queue
        new_sessions = self._generate_new_vehicles()
        self.waiting_queue.extend(new_sessions)
        
        # Ensure queue doesn't exceed maximum length
        if len(self.waiting_queue) > self.max_queue_length:
            # Penalize for exceeding queue capacity
            reward -= 10 * (len(self.waiting_queue) - self.max_queue_length)
            # Remove excess vehicles
            self.waiting_queue = self.waiting_queue[:self.max_queue_length]
        
        # Update grid load and electricity price
        self.current_load = self._calculate_grid_load()
        self.electricity_price = self._calculate_electricity_price()
        
        # Calculate reward
        reward += self._calculate_reward(completed_charges, num_to_charge)
        
        # Get observation
        observation = self._get_observation()
        
        # Check if episode is done
        done = self.time_step == 23
        
        # Additional info
        info = {
            'queue_length': len(self.waiting_queue),
            'grid_load': self.current_load,
            'price': self.electricity_price,
            'completed_charges': completed_charges
        }
        
        return observation, reward, done, False, info
    
    def _get_observation(self):
        """Get current state observation."""
        return np.array([
            len(self.waiting_queue),
            self.charging_stations.count(None),
            self.current_load,
            self.electricity_price,
            self.time_step
        ], dtype=np.float32)
    
    def _generate_new_vehicles(self):
        """Generate new vehicles arriving at the charging station."""
        # Time-dependent arrival rate
        if 6 <= self.time_step < 10:  # Morning peak
            arrival_rate = 3
        elif 16 <= self.time_step < 20:  # Evening peak
            arrival_rate = 4
        else:  # Off-peak
            arrival_rate = 1
        
        # Adjust arrival rate based on load factor
        arrival_rate *= self.load_factor
        
        # Generate vehicles
        num_new_vehicles = np.random.poisson(arrival_rate)
        new_sessions = []
        
        for i in range(num_new_vehicles):
            session = ChargingSession(
                id=len(self.completed_sessions) + len(self.waiting_queue) + i,
                battery_soc=np.random.uniform(0.2, 0.4),  # Initial battery level
                target_soc=np.random.uniform(0.8, 1.0),   # Target battery level
                arrival_time=self.time_step,
                status=ChargingSessionStatus.WAITING
            )
            new_sessions.append(session)
        
        return new_sessions
    
    def _calculate_grid_load(self):
        """Calculate current grid load."""
        # Base load
        load = self.base_load
        
        # Add load from charging vehicles
        for station in self.charging_stations:
            if station is not None:
                load += station.charging_rate
        
        return load
    
    def _calculate_electricity_price(self):
        """Calculate electricity price based on time of day and grid load."""
        # Base price component (time-dependent)
        if 6 <= self.time_step < 10 or 16 <= self.time_step < 20:  # Peak hours
            base_price = 0.8
        else:  # Off-peak hours
            base_price = 0.4
        
        # Load-dependent component
        load_ratio = self.current_load / (self.base_load * 2)
        load_price = 0.2 * load_ratio
        
        # Combine components and apply price factor
        price = (base_price + load_price) * self.price_factor
        
        return min(max(price, 0), 1)  # Ensure price is between 0 and 1
    
    def _calculate_reward(self, completed_charges, num_started_charges):
        """Calculate reward based on charging operations and grid status."""
        reward = 0
        
        # Reward for completed charging sessions
        reward += 50 * completed_charges
        
        # Reward for starting new charging sessions
        reward += 20 * num_started_charges
        
        # Penalty for waiting vehicles
        reward -= 5 * len(self.waiting_queue)
        
        # Penalty for high grid load
        if self.current_load > self.base_load * 1.5:
            reward -= 10 * (self.current_load / self.base_load - 1.5)
        
        # Penalty for high electricity price
        if self.electricity_price > 0.8:
            reward -= 10 * (self.electricity_price - 0.8)
        
        return reward
    
    def render(self):
        """Render the environment (not implemented)."""
        pass 