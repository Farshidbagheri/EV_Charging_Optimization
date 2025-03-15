# EV Charging Optimization System

Advanced electric vehicle charging management system using reinforcement learning to optimize charging schedules and balance grid load.

## Features

- Intelligent charging station management
- Real-time grid load balancing
- Dynamic pricing optimization
- Queue management system
- Multi-station coordination
- Performance monitoring

## Key Results

- Completion Rate: 70.33% ± 14.12%
- Mean Queue Length: 5.16 ± 1.53 vehicles
- Grid Load: 812.61 ± 35.86 kW
- Electricity Price: 0.70 ± 0.007 units
- Mean Reward: 44.46 ± 12.27

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
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
python -m src.train
```

This will:
- Initialize the environment
- Train the optimization agent
- Save model checkpoints
- Generate performance metrics

### Evaluation

```bash
python -m src.evaluate
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

## Environment Configuration

The charging environment includes:
- 10 charging stations
- Maximum queue: 20 vehicles
- Dynamic arrival patterns
- Variable charging rates (50 kW max)
- Base grid load: 500 kW

## Performance Metrics

1. Charging Completion Rate
2. Queue Management
3. Grid Load Balancing
4. Price Optimization
5. Waiting Time Analysis

## Model Architecture

- Policy: MLP [512, 256]
- Parallel processing (8 environments)
- Batch size: 512
- Learning rate: 5e-4
- Optimized hyperparameters

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions and feedback, please open an issue in the GitHub repository.

