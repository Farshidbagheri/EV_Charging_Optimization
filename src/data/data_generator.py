import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

class EVChargingDataGenerator:
    """Generates synthetic EV charging session data with realistic patterns."""
    
    def __init__(
        self,
        num_sessions: int = 1000,
        time_interval_minutes: int = 15,
        base_price: float = 0.5,
        start_date: str = "2024-01-01"
    ):
        self.num_sessions = num_sessions
        self.time_interval = time_interval_minutes
        self.base_price = base_price
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        
        # Define charging patterns
        self.peak_hours = [(7, 9), (17, 19)]  # Morning and evening peaks
        self.weekend_multiplier = 0.7  # Less charging on weekends
        self.seasonal_factors = {
            'spring': 1.0,
            'summer': 1.2,  # More charging in summer
            'fall': 0.9,
            'winter': 0.8   # Less charging in winter
        }
    
    def _get_season(self, date: datetime) -> str:
        """Determine the season based on date."""
        month = date.month
        if 3 <= month <= 5:
            return 'spring'
        elif 6 <= month <= 8:
            return 'summer'
        elif 9 <= month <= 11:
            return 'fall'
        else:
            return 'winter'
    
    def _generate_time_features(self) -> Tuple[List[datetime], List[float], List[float]]:
        """Generate timestamps and cyclical time features."""
        timestamps = []
        hour_sin = []
        hour_cos = []
        day_sin = []
        day_cos = []
        
        current_date = self.start_date
        for _ in range(self.num_sessions):
            # Add random minutes (0-1440) to create variation
            minutes = np.random.randint(0, 1440)
            timestamp = current_date + timedelta(minutes=minutes)
            
            # Convert to cyclical features
            hour = timestamp.hour
            day = timestamp.weekday()
            
            # Hour cyclical encoding
            hour_sin.append(np.sin(2 * np.pi * hour / 24))
            hour_cos.append(np.cos(2 * np.pi * hour / 24))
            
            # Day cyclical encoding
            day_sin.append(np.sin(2 * np.pi * day / 7))
            day_cos.append(np.cos(2 * np.pi * day / 7))
            
            timestamps.append(timestamp)
            current_date += timedelta(days=1)
        
        return timestamps, hour_sin, hour_cos, day_sin, day_cos
    
    def _generate_charging_patterns(self, timestamps: List[datetime]) -> Tuple[List[float], List[float]]:
        """Generate realistic charging patterns based on time and season."""
        battery_levels = []
        charging_rates = []
        
        for timestamp in timestamps:
            # Get season and time-based factors
            season = self._get_season(timestamp)
            hour = timestamp.hour
            is_weekend = timestamp.weekday() >= 5
            
            # Base charging rate
            base_rate = np.random.normal(0.5, 0.1)  # Normal distribution around 0.5
            
            # Apply time-based adjustments
            for peak_start, peak_end in self.peak_hours:
                if peak_start <= hour <= peak_end:
                    base_rate *= 1.2  # Increase charging during peak hours
            
            # Apply weekend adjustment
            if is_weekend:
                base_rate *= self.weekend_multiplier
            
            # Apply seasonal adjustment
            base_rate *= self.seasonal_factors[season]
            
            # Generate battery level (lower in winter)
            if season == 'winter':
                battery_level = np.random.uniform(0.1, 0.3)
            else:
                battery_level = np.random.uniform(0.2, 0.4)
            
            # Ensure charging rate is within [0, 1]
            charging_rate = np.clip(base_rate, 0, 1)
            
            battery_levels.append(battery_level)
            charging_rates.append(charging_rate)
        
        return battery_levels, charging_rates
    
    def _generate_grid_features(self, timestamps: List[datetime], charging_rates: List[float]) -> Tuple[List[float], List[float]]:
        """Generate grid load and electricity prices."""
        grid_loads = []
        prices = []
        
        for timestamp, charging_rate in zip(timestamps, charging_rates):
            hour = timestamp.hour
            season = self._get_season(timestamp)
            
            # Base grid load
            base_load = np.random.normal(0.5, 0.1)
            
            # Add charging impact
            charging_impact = charging_rate * 0.3
            
            # Add time-based variation
            for peak_start, peak_end in self.peak_hours:
                if peak_start <= hour <= peak_end:
                    base_load *= 1.3
            
            # Add seasonal variation
            if season == 'summer':
                base_load *= 1.2  # Higher load in summer
            elif season == 'winter':
                base_load *= 1.1  # Slightly higher in winter
            
            # Calculate price based on load and time
            base_price = self.base_price
            price_multiplier = 1 + 0.5 * base_load  # Price increases with load
            
            for peak_start, peak_end in self.peak_hours:
                if peak_start <= hour <= peak_end:
                    price_multiplier *= 1.5  # Peak pricing
            
            # Add some random noise
            price_noise = np.random.normal(0, 0.05)
            
            # Ensure values are within bounds
            grid_load = np.clip(base_load + charging_impact, 0, 1)
            price = np.clip(base_price * price_multiplier + price_noise, 0, 1)
            
            grid_loads.append(grid_load)
            prices.append(price)
        
        return grid_loads, prices
    
    def generate_charging_sessions(self) -> pd.DataFrame:
        """Generate complete charging session dataset."""
        # Generate time features
        timestamps, hour_sin, hour_cos, day_sin, day_cos = self._generate_time_features()
        
        # Generate charging patterns
        battery_levels, charging_rates = self._generate_charging_patterns(timestamps)
        
        # Generate grid features
        grid_loads, prices = self._generate_grid_features(timestamps, charging_rates)
        
        # Create DataFrame
        data = {
            'timestamp': timestamps,
            'hour_sin': hour_sin,
            'hour_cos': hour_cos,
            'day_sin': day_sin,
            'day_cos': day_cos,
            'battery_level': battery_levels,
            'charging_rate': charging_rates,
            'grid_load': grid_loads,
            'electricity_price': prices
        }
        
        return pd.DataFrame(data)

class DataPreprocessor:
    """Preprocesses the generated charging session data."""
    
    def __init__(self):
        self.scalers = {}
    
    def preprocess(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Preprocess the data and return processed DataFrame and scalers."""
        # Create a copy to avoid modifying original data
        processed_data = data.copy()
        
        # Extract numerical features
        numerical_features = ['battery_level', 'charging_rate', 'grid_load', 'electricity_price']
        
        # Normalize numerical features
        for feature in numerical_features:
            min_val = processed_data[feature].min()
            max_val = processed_data[feature].max()
            processed_data[feature] = (processed_data[feature] - min_val) / (max_val - min_val)
            self.scalers[feature] = {'min': min_val, 'max': max_val}
        
        return processed_data, self.scalers 