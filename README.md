# EV Charging Optimization System

Advanced electric vehicle charging management system using reinforcement learning to optimize charging schedules and balance grid load.

## Overview

This project utilizes reinforcement learning (RL) for optimizing electric vehicle (EV) charging strategies, considering grid load, pricing, and dynamic demand. The system implements a custom Gym environment and uses state-of-the-art RL algorithms to manage charging schedules efficiently.

## Features

### Core Functionality
- Custom Gym environment for EV charging simulation
- Load-based and time-based pricing integration
- Optimized reinforcement learning agent
- Real-time grid load balancing
- Dynamic pricing optimization
- Queue management system

### Technical Implementation
- Multi-station parallel processing
- Custom reward shaping
- State/action space optimization
- Vectorized environment support
- Comprehensive metrics tracking

## Performance Results

- **Charging Efficiency**: 70.33% completion rate (±14.12%)
- **Queue Management**: 5.16 average length (±1.53 vehicles)
- **Grid Stability**: 812.61 kW average load (±35.86 kW)
- **Price Optimization**: 0.70 average price unit (±0.007)
- **System Performance**: 44.46 mean reward (±12.27)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Farshidbagheri/EV_Charging_Optimization.git
cd EV_Charging_Optimization
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training
```bash
python src/train.py
```

The training process:
- Initializes the custom Gym environment
- Configures the RL agent with optimized parameters
- Trains using parallel environment processing
- Saves checkpoints and performance metrics

### Evaluation
```bash
python src/evaluate.py
```

## Project Structure
```
EV_Charging_Optimization/
│── src/                  # Source code directory
│   ├── env/             # Custom environment for charging stations
│   ├── models/          # RL models and training scripts
│   ├── data/            # Sample datasets
│   ├── utils/           # Utility scripts
│── results/             # Evaluation metrics and plots
│── docs/                # Documentation files
│── notebooks/           # Analysis notebooks
│── README.md            # Project overview
│── requirements.txt     # Dependencies
│── LICENSE             # License information
│── .gitignore          # Git ignore rules
```

## Technical Details

### Environment Configuration
- Number of charging stations: 10
- Queue capacity: 20 vehicles
- Maximum charging rate: 50 kW per station
- Base grid load: 500 kW
- Dynamic arrival patterns based on time of day

### Model Architecture
- Policy network: MLP [512, 256]
- Parallel environments: 8
- Batch size: 512
- Learning rate: 5e-4
- Optimized hyperparameters for stability

## Performance Metrics

1. **Charging Efficiency**
   - Session completion rate
   - Average charging time
   - Energy delivery rate

2. **Queue Management**
   - Average wait times
   - Queue length distribution
   - Service rate analysis

3. **Grid Impact**
   - Load distribution
   - Peak demand reduction
   - Grid stability metrics

4. **Economic Performance**
   - Price optimization
   - Revenue analysis
   - Cost efficiency metrics

## Developer

**Farshid Bagheri Saravi**
- Lead Developer & System Architect
- Machine Learning Engineer
- Contact: [GitHub Profile](https://github.com/Farshidbagheri)

## Copyright

Copyright © 2024 Farshid Bagheri Saravi. All rights reserved.

This software and associated documentation files are proprietary and confidential.

