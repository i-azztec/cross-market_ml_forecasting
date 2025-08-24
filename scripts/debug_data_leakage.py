#!/usr/bin/env python3
"""
Debug script to identify data leakage in the model.
Analyze feature importance and detect problematic features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle

# Set matplotlib backend to avoid tkinter threading issues
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_latest_models():
    """Load the latest trained models."""
    models_dir = Path("data/models")
    
    # Find latest models
    model_files = {
        'logistic_regression': None,
        'random_forest': None, 
        'xgboost': None
    }
    
    for model_type in model_files.keys():
        pattern = f"{model_type}_*.pkl"
        files = list(models_dir.glob(pattern))
        if files:
            # Get most recent
            latest = max(files, key=lambda x: x.stat().st_mtime)
            model_files[model_type] = latest
            
    return model_files

def analyze_feature_importance(model, X, y, feature_names, model_name):
    """Analyze feature importance for a model."""
    logger.info(f"Analyzing feature importance for {model_name}")
    
    results = {}
    
    # Tree-based models have feature_importances_
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        results['builtin_importance'] = dict(zip(feature_names, importance))
        
    # Calculate permutation importance
    logger.info(f"Calculating permutation importance for {model_name}")
    perm_importance = permutation_importance(
        model, X, y, n_repeats=5, random_state=42, n_jobs=-1
    )
    results['permutation_importance'] = dict(zip(feature_names, perm_importance.importances_mean))
    
    return results

def check_feature_correlations_with_target(df, target_col):
    """Check correlation between features and target."""
    logger.info("Checking feature correlations with target")
    
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col != target_col and not col.startswith('y_') and not col.startswith('ret_')]
    
    correlations = {}
    target_values = df[target_col]
    
    for col in feature_cols:
        if col in df.columns:
            corr = df[col].corr(target_values)
            if not np.isnan(corr):
                correlations[col] = abs(corr)  # Use absolute correlation
                
    return correlations

def detect_future_leakage_features(df):
    """Detect features that might contain future information."""
    logger.info("Detecting potential future leakage features")
    
    suspicious_features = []
    
    # Check for features that perfectly predict the target
    target_col = 'is_positive_growth_30d_future'
    if target_col in df.columns:
        for col in df.columns:
            if col != target_col and not col.startswith('symbol'):
                # Check if feature perfectly correlates with target
                try:
                    corr = abs(df[col].corr(df[target_col]))
                    if corr > 0.95:  # Very high correlation
                        suspicious_features.append((col, corr))
                except:
                    continue
                    
    return suspicious_features

def main():
    """Main analysis function."""
    logger.info("Starting data leakage analysis...")
    
    # Load data
    logger.info("Loading processed dataset...")
    df = pd.read_parquet("data/processed/dataset.parquet")
    logger.info(f"Dataset shape: {df.shape}")
    
    # Load predictions
    pred_files = list(Path("data/reports").glob("predictions_*.parquet"))
    if pred_files:
        latest_pred = max(pred_files, key=lambda x: x.stat().st_mtime)
        pred_df = pd.read_parquet(latest_pred)
        logger.info(f"Loaded predictions from {latest_pred}")
    else:
        logger.error("No predictions file found")
        return
    
    # Check feature correlations
    target_col = 'is_positive_growth_30d_future'
    correlations = check_feature_correlations_with_target(df, target_col)
    
    # Sort by correlation strength
    sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    
    print("\n=== TOP FEATURES BY CORRELATION WITH TARGET ===")
    for i, (feature, corr) in enumerate(sorted_corr[:20]):
        print(f"{i+1:2d}. {feature:30s} | Correlation: {corr:.4f}")
    
    # Detect suspicious features
    suspicious = detect_future_leakage_features(df)
    
    print("\n=== SUSPICIOUS FEATURES (High correlation > 0.95) ===")
    if suspicious:
        for feature, corr in sorted(suspicious, key=lambda x: x[1], reverse=True):
            print(f"  {feature:30s} | Correlation: {corr:.4f}")
    else:
        print("  No suspicious features found")
    
    # Load and analyze models
    model_files = load_latest_models()
    
    # Prepare data for model analysis
    feature_cols = [col for col in df.columns if col not in [
        'symbol', 'is_positive_growth_30d_future', 'y_30d', 
        'ret_30d', 'growth_future_30d', 'ret_1d', 'ret_5d', 'ret_10d', 'ret_20d'
    ]]
    
    X = df[feature_cols].fillna(0)
    y = df[target_col].fillna(0).astype(int)
    
    # Remove rows with NaN in target
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]
    
    print(f"\n=== MODEL FEATURE IMPORTANCE ANALYSIS ===")
    print(f"Using {len(feature_cols)} features, {len(X)} samples")
    
    # Analyze each model
    for model_name, model_file in model_files.items():
        if model_file and model_file.exists():
            print(f"\n--- {model_name.upper()} ---")
            
            try:
                # Load model
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
                
                # Analyze importance
                importance_results = analyze_feature_importance(
                    model, X, y, feature_cols, model_name
                )
                
                # Print top features
                if 'builtin_importance' in importance_results:
                    print("Top 10 features by built-in importance:")
                    builtin = importance_results['builtin_importance']
                    sorted_builtin = sorted(builtin.items(), key=lambda x: x[1], reverse=True)
                    for i, (feat, imp) in enumerate(sorted_builtin[:10]):
                        print(f"  {i+1:2d}. {feat:25s} | {imp:.4f}")
                
                if 'permutation_importance' in importance_results:
                    print("Top 10 features by permutation importance:")
                    perm = importance_results['permutation_importance']
                    sorted_perm = sorted(perm.items(), key=lambda x: x[1], reverse=True)
                    for i, (feat, imp) in enumerate(sorted_perm[:10]):
                        print(f"  {i+1:2d}. {feat:25s} | {imp:.4f}")
                        
            except Exception as e:
                logger.error(f"Error analyzing {model_name}: {e}")
    
    # Check specific suspicious patterns
    print(f"\n=== CHECKING FOR SPECIFIC LEAKAGE PATTERNS ===")
    
    # Check for future data in features
    future_features = [col for col in df.columns if 'future' in col.lower() and col != target_col]
    if future_features:
        print(f"Features with 'future' in name: {future_features}")
    
    # Check for return features that shouldn't be there
    return_features = [col for col in feature_cols if 'ret_' in col]
    if return_features:
        print(f"Return features in model: {return_features}")
    
    # Check date-time features
    date_features = [col for col in feature_cols if any(x in col.lower() for x in ['date', 'time', 'day', 'month', 'year'])]
    if date_features:
        print(f"Date/time features: {date_features}")
    
    print(f"\n=== RECOMMENDATIONS ===")
    print("1. Remove features with correlation > 0.95 with target")
    print("2. Remove any 'ret_' features from model features") 
    print("3. Ensure no future data leaks into past predictions")
    print("4. Check shift operations in feature engineering")

if __name__ == "__main__":
    main()