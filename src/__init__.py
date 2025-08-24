"""
Cross-Market 30D Directional Forecasting project.

This package implements an end-to-end ML trading research pipeline that:
- Predicts 30-day forward direction (up/down) for ~60 tickers across global markets
- Produces vectorized portfolio simulations with threshold analysis  
- Provides comprehensive model evaluation and performance metrics

Main modules:
- config: Configuration and constants
- data_loader: Data ingestion from Yahoo Finance
- feature_engineering: Technical indicators and feature creation
- models: ML models (XGBoost, RandomForest, LogisticRegression)
- simulation: Trading simulation and backtesting
- evaluation: Performance metrics and reporting
"""

__version__ = "0.1.0"
__author__ = "Stock Markets Analytics Zoomcamp Project"

# Import main classes for convenience
from .config import *
from .data_loader import DataLoader, download_all_data

__all__ = [
    'DataLoader',
    'download_all_data',
    # Config exports will be available via import *
]