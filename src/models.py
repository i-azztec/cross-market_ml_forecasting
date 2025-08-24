"""
ML Models module for Cross-Market 30D Directional Forecasting project.

Implements training, validation and prediction for:
- XGBoost Classifier (primary model)
- Random Forest Classifier (baseline)
- Logistic Regression (simple baseline)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb

from config import (
    MODEL_PARAMS, MODELS_DIR, REPORTS_DIR,
    TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT,
    PREDICTION_HORIZON_DAYS
)

class ModelTrainer:
    """
    Trains and evaluates ML models for directional forecasting.
    
    Features:
    - Temporal train/validation/test splits
    - Model training with hyperparameter optimization
    - Performance evaluation and model comparison
    - Model persistence and loading
    """
    
    def __init__(self, models_dir: str = None):
        """Initialize ModelTrainer."""
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Store trained models and scalers
        self.models = {}
        self.scalers = {}
        self.model_metadata = {}
        
    def prepare_data(self, 
                    df: pd.DataFrame,
                    feature_columns: List[str] = None,
                    target_column: str = 'y_30d') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare data with temporal train/validation/test splits.
        
        Args:
            df: Input dataframe with features and targets
            feature_columns: List of feature column names (if None, auto-detect)
            target_column: Target column name
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        self.logger.info("Preparing data splits...")
        
        # Remove rows with missing target
        df_clean = df.dropna(subset=[target_column]).copy()
        self.logger.info(f"Cleaned data shape: {df_clean.shape}")
        
        # Auto-detect feature columns if not provided
        if feature_columns is None:
            # Comprehensive exclusion list to prevent data leakage
            exclude_cols = {
                # Meta columns
                'symbol', 'market',
                
                # Target variables (ALL variants to prevent leakage)
                target_column, 'y_30d', 'is_positive_growth_30d_future',
                
                # Future information (anything with 'future' in name)
                'growth_future_30d', 'growth_future_1d', 'growth_future_5d', 
                'growth_future_10d', 'growth_future_20d',
                
                # Return variables (contain future info or are derived from target)
                'ret_30d', 'ret_1d', 'ret_5d', 'ret_10d', 'ret_20d',
                
                # Derived return features (also contain future info)
                'ret_20d_zscore', 'ret_20d_rank', 'ret_30d_zscore', 'ret_30d_rank',
                'ret_1d_zscore', 'ret_1d_rank', 'ret_5d_zscore', 'ret_5d_rank',
                'ret_10d_zscore', 'ret_10d_rank',
                
                # Raw price data (use technical indicators instead)
                'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'
            }
            feature_columns = [col for col in df_clean.columns if col not in exclude_cols]
        
        self.logger.info(f"Using {len(feature_columns)} features")
        
        # Sort by date for temporal splits
        if not isinstance(df_clean.index, pd.DatetimeIndex):
            df_clean = df_clean.sort_values('date') if 'date' in df_clean.columns else df_clean.sort_index()
        else:
            df_clean = df_clean.sort_index()
        
        # Calculate split indices
        n_samples = len(df_clean)
        train_end_idx = int(n_samples * TRAIN_SPLIT)
        val_end_idx = int(n_samples * (TRAIN_SPLIT + VALIDATION_SPLIT))
        
        # Create splits
        train_df = df_clean.iloc[:train_end_idx].copy()
        val_df = df_clean.iloc[train_end_idx:val_end_idx].copy()
        test_df = df_clean.iloc[val_end_idx:].copy()
        
        # Log split info
        self.logger.info(f"Train: {len(train_df)} samples ({train_df.index.min()} to {train_df.index.max()})")
        self.logger.info(f"Valid: {len(val_df)} samples ({val_df.index.min()} to {val_df.index.max()})")
        self.logger.info(f"Test: {len(test_df)} samples ({test_df.index.min()} to {test_df.index.max()})")
        
        # Check class balance
        train_balance = train_df[target_column].value_counts(normalize=True)
        self.logger.info(f"Train class balance: {train_balance.to_dict()}")
        
        # Store feature columns for later use
        self.feature_columns = feature_columns
        self.target_column = target_column
        
        return train_df, val_df, test_df
    
    def train_xgboost(self, 
                     train_df: pd.DataFrame, 
                     val_df: pd.DataFrame,
                     params: Dict = None) -> Dict[str, Any]:
        """
        Train XGBoost model with early stopping.
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            params: Model parameters (uses config defaults if None)
            
        Returns:
            Dictionary with model and training info
        """
        self.logger.info("Training XGBoost model...")
        
        # Use default params if not provided
        if params is None:
            params = MODEL_PARAMS['xgboost'].copy()
        
        # Prepare features and targets
        X_train = train_df[self.feature_columns].fillna(0)
        y_train = train_df[self.target_column]
        X_val = val_df[self.feature_columns].fillna(0)
        y_val = val_df[self.target_column]
        
        # Calculate class weights for imbalanced data
        pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        params['scale_pos_weight'] = pos_weight
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Set up evaluation
        evallist = [(dtrain, 'train'), (dval, 'eval')]
        
        # Train model
        model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=params.get('n_estimators', 400),
            evals=evallist,
            early_stopping_rounds=params.get('early_stopping_rounds', 50),
            verbose_eval=False
        )
        
        # Make predictions for evaluation
        y_train_pred = model.predict(dtrain)
        y_val_pred = model.predict(dval)
        
        # Calculate metrics
        train_auc = roc_auc_score(y_train, y_train_pred)
        val_auc = roc_auc_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, (y_val_pred > 0.5).astype(int))
        
        self.logger.info(f"XGBoost - Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}, Val F1: {val_f1:.4f}")
        
        # Store model and metadata
        model_info = {
            'model': model,
            'model_type': 'xgboost',
            'params': params,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'train_auc': train_auc,
            'val_auc': val_auc,
            'val_f1': val_f1,
            'feature_importance': dict(zip(self.feature_columns, model.get_score(importance_type='weight').values())),
            'training_date': datetime.now().isoformat()
        }
        
        self.models['xgboost'] = model
        self.model_metadata['xgboost'] = model_info
        
        return model_info
    
    def train_random_forest(self, 
                           train_df: pd.DataFrame, 
                           val_df: pd.DataFrame,
                           params: Dict = None) -> Dict[str, Any]:
        """
        Train Random Forest model.
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            params: Model parameters (uses config defaults if None)
            
        Returns:
            Dictionary with model and training info
        """
        self.logger.info("Training Random Forest model...")
        
        # Use default params if not provided
        if params is None:
            params = MODEL_PARAMS['random_forest'].copy()
        
        # Prepare features and targets
        X_train = train_df[self.feature_columns].fillna(0)
        y_train = train_df[self.target_column]
        X_val = val_df[self.feature_columns].fillna(0)
        y_val = val_df[self.target_column]
        
        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Make predictions for evaluation
        y_train_pred = model.predict_proba(X_train)[:, 1]
        y_val_pred = model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        train_auc = roc_auc_score(y_train, y_train_pred)
        val_auc = roc_auc_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, (y_val_pred > 0.5).astype(int))
        
        self.logger.info(f"RandomForest - Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}, Val F1: {val_f1:.4f}")
        
        # Store model and metadata
        model_info = {
            'model': model,
            'model_type': 'random_forest',
            'params': params,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'train_auc': train_auc,
            'val_auc': val_auc,
            'val_f1': val_f1,
            'feature_importance': dict(zip(self.feature_columns, model.feature_importances_)),
            'training_date': datetime.now().isoformat()
        }
        
        self.models['random_forest'] = model
        self.model_metadata['random_forest'] = model_info
        
        return model_info
    
    def train_logistic_regression(self, 
                                 train_df: pd.DataFrame, 
                                 val_df: pd.DataFrame,
                                 params: Dict = None) -> Dict[str, Any]:
        """
        Train Logistic Regression model with scaling.
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            params: Model parameters (uses config defaults if None)
            
        Returns:
            Dictionary with model and training info
        """
        self.logger.info("Training Logistic Regression model...")
        
        # Use default params if not provided
        if params is None:
            params = MODEL_PARAMS['logistic_regression'].copy()
        
        # Prepare features and targets
        X_train = train_df[self.feature_columns].fillna(0)
        y_train = train_df[self.target_column]
        X_val = val_df[self.feature_columns].fillna(0)
        y_val = val_df[self.target_column]
        
        # Scale features for logistic regression
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Train model
        model = LogisticRegression(**params)
        model.fit(X_train_scaled, y_train)
        
        # Make predictions for evaluation
        y_train_pred = model.predict_proba(X_train_scaled)[:, 1]
        y_val_pred = model.predict_proba(X_val_scaled)[:, 1]
        
        # Calculate metrics
        train_auc = roc_auc_score(y_train, y_train_pred)
        val_auc = roc_auc_score(y_val, y_val_pred)
        val_f1 = f1_score(y_val, (y_val_pred > 0.5).astype(int))
        
        self.logger.info(f"LogisticRegression - Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}, Val F1: {val_f1:.4f}")
        
        # Store model, scaler and metadata
        model_info = {
            'model': model,
            'scaler': scaler,
            'model_type': 'logistic_regression',
            'params': params,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'train_auc': train_auc,
            'val_auc': val_auc,
            'val_f1': val_f1,
            'feature_importance': dict(zip(self.feature_columns, np.abs(model.coef_[0]))),
            'training_date': datetime.now().isoformat()
        }
        
        self.models['logistic_regression'] = model
        self.scalers['logistic_regression'] = scaler
        self.model_metadata['logistic_regression'] = model_info
        
        return model_info
    
    def train_all_models(self, 
                        train_df: pd.DataFrame, 
                        val_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train all models in logical order: LogisticRegression → RandomForest → XGBoost.
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            
        Returns:
            Dictionary with all model results
        """
        self.logger.info("Training all models in order: LogisticRegression → RandomForest → XGBoost")
        
        results = {}
        
        # Train models in logical order from simple to complex
        results['logistic_regression'] = self.train_logistic_regression(train_df, val_df)
        results['random_forest'] = self.train_random_forest(train_df, val_df)
        results['xgboost'] = self.train_xgboost(train_df, val_df)
        
        # Create comparison summary
        comparison = pd.DataFrame({
            model_name: {
                'Train AUC': info['train_auc'],
                'Val AUC': info['val_auc'],
                'Val F1': info['val_f1']
            }
            for model_name, info in results.items()
        }).T
        
        self.logger.info("Model comparison:")
        self.logger.info(f"\n{comparison}")
        
        results['comparison'] = comparison
        return results
    
    def predict(self, 
               df: pd.DataFrame, 
               model_name: str) -> np.ndarray:
        """
        Make predictions with a trained model.
        
        Args:
            df: Input dataframe
            model_name: Name of the model to use
            
        Returns:
            Array of prediction probabilities
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available: {list(self.models.keys())}")
        
        model = self.models[model_name]
        X = df[self.feature_columns].fillna(0)
        
        if model_name == 'xgboost':
            dtest = xgb.DMatrix(X)
            predictions = model.predict(dtest)
        elif model_name == 'logistic_regression':
            # Apply scaling for logistic regression
            scaler = self.scalers[model_name]
            X_scaled = scaler.transform(X)
            predictions = model.predict_proba(X_scaled)[:, 1]
        else:
            # Random forest and other sklearn models
            predictions = model.predict_proba(X)[:, 1]
        
        return predictions
    
    def evaluate_on_test(self, 
                        test_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all trained models on test set.
        
        Args:
            test_df: Test dataframe
            
        Returns:
            Dictionary with test metrics for each model
        """
        self.logger.info("Evaluating models on test set...")
        
        test_results = {}
        y_test = test_df[self.target_column]
        
        for model_name in self.models.keys():
            y_pred_proba = self.predict(test_df, model_name)
            y_pred_binary = (y_pred_proba > 0.5).astype(int)
            
            metrics = {
                'test_auc': roc_auc_score(y_test, y_pred_proba),
                'test_f1': f1_score(y_test, y_pred_binary),
                'test_precision': precision_score(y_test, y_pred_binary),
                'test_recall': recall_score(y_test, y_pred_binary),
                'test_accuracy': accuracy_score(y_test, y_pred_binary)
            }
            
            test_results[model_name] = metrics
            self.logger.info(f"{model_name} test metrics: {metrics}")
        
        return test_results
    
    def save_models(self, timestamp: str = None) -> str:
        """
        Save all trained models and metadata to disk.
        
        Args:
            timestamp: Optional timestamp string (auto-generated if None)
            
        Returns:
            Timestamp string used for filenames
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.logger.info(f"Saving models with timestamp: {timestamp}")
        
        for model_name, model in self.models.items():
            # Save model
            model_file = self.models_dir / f"{model_name}_{timestamp}.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)
            
            # Save scaler if exists
            if model_name in self.scalers:
                scaler_file = self.models_dir / f"{model_name}_scaler_{timestamp}.pkl"
                with open(scaler_file, 'wb') as f:
                    pickle.dump(self.scalers[model_name], f)
            
            # Save metadata
            metadata_file = self.models_dir / f"{model_name}_metadata_{timestamp}.json"
            metadata = self.model_metadata[model_name].copy()
            # Remove non-serializable objects
            metadata.pop('model', None)
            metadata.pop('scaler', None)
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.info(f"Saved {model_name} to {model_file}")
        
        return timestamp
    
    def load_models(self, timestamp: str) -> None:
        """
        Load trained models from disk.
        
        Args:
            timestamp: Timestamp string for model files
        """
        self.logger.info(f"Loading models with timestamp: {timestamp}")
        
        model_names = ['xgboost', 'random_forest', 'logistic_regression']
        
        for model_name in model_names:
            model_file = self.models_dir / f"{model_name}_{timestamp}.pkl"
            metadata_file = self.models_dir / f"{model_name}_metadata_{timestamp}.json"
            
            if model_file.exists() and metadata_file.exists():
                # Load model
                with open(model_file, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
                
                # Load metadata
                with open(metadata_file, 'r') as f:
                    self.model_metadata[model_name] = json.load(f)
                
                # Load scaler if exists
                scaler_file = self.models_dir / f"{model_name}_scaler_{timestamp}.pkl"
                if scaler_file.exists():
                    with open(scaler_file, 'rb') as f:
                        self.scalers[model_name] = pickle.load(f)
                
                # Restore feature columns from metadata
                self.feature_columns = self.model_metadata[model_name]['feature_columns']
                self.target_column = self.model_metadata[model_name]['target_column']
                
                self.logger.info(f"Loaded {model_name}")
    
    def get_feature_importance(self, 
                              model_name: str, 
                              top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance for a trained model.
        
        Args:
            model_name: Name of the model
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance
        """
        if model_name not in self.model_metadata:
            raise ValueError(f"Model {model_name} not found")
        
        importance_dict = self.model_metadata[model_name]['feature_importance']
        
        importance_df = pd.DataFrame(
            list(importance_dict.items()),
            columns=['feature', 'importance']
        ).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df

def create_predictions_dataframe(df: pd.DataFrame, 
                                model_trainer: ModelTrainer,
                                model_names: List[str] = None) -> pd.DataFrame:
    """
    Create predictions dataframe for simulation.
    
    Args:
        df: Input dataframe with features
        model_trainer: Trained ModelTrainer instance
        model_names: List of model names (uses all if None)
        
    Returns:
        DataFrame with predictions for each model
    """
    if model_names is None:
        model_names = list(model_trainer.models.keys())
    
    # Start with basic info needed for simulation
    # Include both legacy ret_30d and original notebook growth_future_30d
    preds_df = df[['symbol', model_trainer.target_column, 'ret_30d', 'growth_future_30d']].copy()
    preds_df = preds_df.rename(columns={model_trainer.target_column: 'y_true'})
    
    # Add predictions for each model
    for model_name in model_names:
        preds_df[f'y_proba_{model_name}'] = model_trainer.predict(df, model_name)
    
    return preds_df