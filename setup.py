from setuptools import setup, find_packages

setup(
    name="ev_charging_optimization",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.29.1",
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "matplotlib>=3.7.1",
        "seaborn>=0.12.2",
        "torch>=2.1.0",
        "stable-baselines3>=2.2.1",
        "tensorboard>=2.14.0",
        "tqdm>=4.65.0",
    ],
    python_requires=">=3.8",
) 