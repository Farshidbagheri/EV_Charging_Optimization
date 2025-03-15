# EV Charging Optimization System Deployment Guide

This guide provides instructions for deploying the EV Charging Optimization System in a production environment.

## System Requirements

### Hardware Requirements
- CPU: 4+ cores
- RAM: 8GB minimum
- Storage: 20GB minimum
- GPU: Optional, but recommended for faster inference

### Software Requirements
- Python 3.8+
- CUDA 11.0+ (if using GPU)
- Operating System: Linux/Unix (recommended), Windows, macOS

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd AI_Electric_Vehicles
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements_deployment.txt
```

## Model Deployment

1. Ensure you have a trained model in the `models` directory:
   - `final_model.zip`: The trained PPO model
   - `vec_normalize.pkl`: The observation normalizer

2. The model will be automatically loaded when starting the API server.

## API Deployment

1. Start the FastAPI server:
```bash
python -m src.deployment.api
```

2. The API will be available at:
   - Main API: http://localhost:8000
   - Interactive Documentation: http://localhost:8000/docs
   - Alternative Documentation: http://localhost:8000/redoc

### API Endpoints

1. Make Charging Decision
   - Endpoint: POST /make_decision
   - Input: Charging state and optional constraints
   - Output: Charging decision with actions and metadata

2. Get Decision History
   - Endpoint: GET /decision_history
   - Optional parameter: limit (number of decisions to return)
   - Output: List of past charging decisions

3. Get Model Information
   - Endpoint: GET /model_info
   - Output: Model metadata and configuration

4. Save Decision History
   - Endpoint: POST /save_history
   - Output: Confirmation of saved history file

## Web UI Deployment

1. Start the Streamlit UI:
```bash
streamlit run src/deployment/ui.py
```

2. Access the UI at http://localhost:8501

## Production Deployment Considerations

### Security
1. Enable HTTPS using a reverse proxy (e.g., Nginx)
2. Implement authentication and authorization
3. Use environment variables for sensitive data
4. Regular security updates

### Monitoring
1. Enable logging for all components
2. Set up monitoring for:
   - API response times
   - Model inference latency
   - System resource usage
   - Error rates

### Scaling
1. Use a process manager (e.g., Supervisor)
2. Consider containerization (Docker)
3. Implement load balancing for multiple instances
4. Use a database for persistent storage

### Backup and Recovery
1. Regular model backups
2. Database backups
3. Configuration backups
4. Recovery procedures documentation

## Example Deployment Script

```bash
#!/bin/bash

# Environment setup
export PYTHONPATH=/path/to/AI_Electric_Vehicles
export CUDA_VISIBLE_DEVICES=0  # If using GPU

# Start API server
nohup python -m src.deployment.api > api.log 2>&1 &

# Start Web UI
nohup streamlit run src/deployment/ui.py > ui.log 2>&1 &

# Monitor logs
tail -f api.log ui.log
```

## Troubleshooting

### Common Issues
1. Model Loading Errors
   - Check model file permissions
   - Verify model file integrity
   - Ensure correct CUDA version

2. API Connection Issues
   - Check firewall settings
   - Verify port availability
   - Check network connectivity

3. Performance Issues
   - Monitor system resources
   - Check for memory leaks
   - Optimize batch sizes

### Logging
- API logs: `api.log`
- UI logs: `ui.log`
- Model logs: `models/model.log`

## Maintenance

### Regular Tasks
1. Update dependencies
2. Monitor system performance
3. Backup data and models
4. Review and update documentation

### Emergency Procedures
1. System failure recovery
2. Data corruption handling
3. Security incident response
4. Backup restoration

## Support

For technical support:
- Email: support@example.com
- Documentation: https://docs.example.com
- Issue Tracker: https://github.com/example/issues 