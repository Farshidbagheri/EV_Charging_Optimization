# AI-Powered EV Charging Station Management

An intelligent system for managing electric vehicle charging stations using reinforcement learning to optimize charging schedules, reduce waiting times, and balance grid load.

## 🚀 Features

- Reinforcement learning-based charging station management
- Real-time grid load balancing
- Dynamic pricing based on demand and time of day
- Queue management optimization
- Multi-station parallel charging coordination
- Performance monitoring and visualization

## 📊 Key Results

- **Completion Rate**: 70.33% ± 14.12%
- **Mean Queue Length**: 5.16 ± 1.53 vehicles
- **Grid Load**: 812.61 ± 35.86 kW
- **Electricity Price**: 0.70 ± 0.007 units
- **Mean Reward**: 44.46 ± 12.27

## 🛠 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AI_Electric_Vehicles.git
cd AI_Electric_Vehicles
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

## 💻 Usage

### Training the Model

```bash
python -m src.train
```

The training script will:
- Initialize the environment with optimized parameters
- Train the PPO agent using parallel environments
- Save the trained model and evaluation metrics
- Generate performance visualizations

### Evaluating the Model

```bash
python -m src.evaluate
```

## 🏗 Project Structure

```
AI_Electric_Vehicles/
├── src/
│   ├── agents/
│   │   └── ppo_agent.py
│   ├── environment/
│   │   └── ev_charging_env.py
│   ├── train.py
│   └── evaluate.py
├── models/
│   └── final_model.zip
├── results/
│   ├── evaluation_metrics.json
│   └── training_evaluation.png
├── logs/
│   └── PPO_tensorboard_logs/
├── requirements.txt
└── README.md
```

## 🔧 Environment Configuration

The charging station environment (`EVChargingEnv`) simulates:
- 10 charging stations
- Maximum queue length of 20 vehicles
- Dynamic arrival rates based on time of day
- Variable charging rates up to 50 kW per station
- Base grid load of 500 kW

## 📈 Performance Metrics

The system is evaluated on multiple metrics:
1. **Charging Completion Rate**: Percentage of successfully completed charging sessions
2. **Queue Management**: Average and maximum queue lengths
3. **Grid Load Balancing**: Distribution and peaks of power consumption
4. **Price Optimization**: Dynamic pricing effectiveness
5. **Waiting Times**: Average and maximum waiting times for vehicles

## 🤖 Model Architecture

The PPO agent uses:
- MLP Policy with [512, 256] units
- Parallel environment processing (8 environments)
- Batch size of 512 for optimal GPU utilization
- Learning rate of 5e-4
- Optimized hyperparameters for stable training

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions and feedback, please open an issue in the GitHub repository.

