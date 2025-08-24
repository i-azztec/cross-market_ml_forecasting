"""
Main CLI entry point for Cross-Market 30D Directional Forecasting project.

Usage:
    python run.py --stage all                    # Run full pipeline
    python run.py --stage ingest                 # Download data only
    python run.py --stage ingest --test-mode     # Test with subset of tickers
"""

import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import json
import re
import glob

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.config import ALL_TICKERS, MACRO_INDICATORS, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR
from src.data_loader import DataLoader
from src.feature_engineering_simple import create_unified_dataset_simple
from src.models import ModelTrainer, create_predictions_dataframe
from src.simulation import TradingSimulator, SimulationReporter, run_complete_simulation
from src.evaluation import PerformanceEvaluator

def setup_logging(level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('pipeline.log')
        ]
    )

def run_ingest_stage(test_mode: bool = False, force_download: bool = False):
    """Run data ingestion stage."""
    logger = logging.getLogger(__name__)
    logger.info("Starting data ingestion stage...")
    
    # Use subset for testing
    if test_mode:
        # Expanded test set: 15 representative tickers across different sectors
        tickers = [
            # Top US Tech (5)
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA',
            # Finance & Healthcare (4)
            'JPM', 'UNH', 'JNJ', 'PFE',
            # Consumer & Industrial (3)
            'PG', 'KO', 'HD',
            # Sector ETFs (3)
            'SPY', 'QQQ', 'XLF'
        ]
        macros = {'vix': '^VIX', 'tnx10y': '^TNX'}  # Key indicators
        logger.info("Running in TEST MODE with 15 tickers and 2 macro indicators")
    else:
        tickers = ALL_TICKERS
        macros = MACRO_INDICATORS
        logger.info(f"Running with {len(tickers)} tickers and {len(macros)} macro indicators")
    
    # Initialize data loader
    loader = DataLoader()
    
    try:
        # Download all data
        ticker_data, macro_data = loader.load_all_data(
            tickers=tickers,
            indicators=macros,
            force_download=force_download
        )
        
        # Save raw data summary
        summary = loader.get_download_summary()
        
        if not ticker_data.empty:
            # Save combined ticker data
            ticker_file = PROCESSED_DATA_DIR / "raw_tickers.parquet"
            ticker_data.to_parquet(ticker_file)
            logger.info(f"Saved ticker data to {ticker_file}")
            
        if not macro_data.empty:
            # Save macro data  
            macro_file = PROCESSED_DATA_DIR / "raw_macro.parquet"
            macro_data.to_parquet(macro_file)
            logger.info(f"Saved macro data to {macro_file}")
        
        # Print summary
        logger.info("=== DOWNLOAD SUMMARY ===")
        logger.info(f"Ticker data shape: {ticker_data.shape}")
        logger.info(f"Macro data shape: {macro_data.shape}")
        
        if summary['failed_tickers']:
            logger.warning(f"Failed tickers: {', '.join(summary['failed_tickers'])}")
        
        if summary['failed_macros']:
            logger.warning(f"Failed macros: {', '.join(summary['failed_macros'])}")
        
        logger.info("Data ingestion completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        return False

def run_features_stage():
    """Run feature engineering stage."""
    logger = logging.getLogger(__name__)
    logger.info("Starting feature engineering stage...")
    
    try:
        # Load raw data
        ticker_file = PROCESSED_DATA_DIR / "raw_tickers.parquet"
        macro_file = PROCESSED_DATA_DIR / "raw_macro.parquet"
        
        if not ticker_file.exists():
            logger.error(f"Ticker data not found: {ticker_file}")
            logger.error("Please run data ingestion first: python run.py --stage ingest")
            return False
            
        if not macro_file.exists():
            logger.error(f"Macro data not found: {macro_file}")
            logger.error("Please run data ingestion first: python run.py --stage ingest")
            return False
        
        # Load data
        logger.info("Loading raw data...")
        ticker_data = pd.read_parquet(ticker_file)
        macro_data = pd.read_parquet(macro_file)
        
        # Create unified dataset with all features
        logger.info("Creating unified dataset with technical features...")
        unified_dataset = create_unified_dataset_simple(
            ticker_data=ticker_data,
            macro_data=macro_data,
            save_path=PROCESSED_DATA_DIR / "dataset.parquet"
        )
        
        # Print summary
        logger.info("=== FEATURE ENGINEERING SUMMARY ===")
        logger.info(f"Final dataset shape: {unified_dataset.shape}")
        logger.info(f"Date range: {unified_dataset.index.min()} to {unified_dataset.index.max()}")
        logger.info(f"Unique symbols: {unified_dataset['symbol'].nunique()}")
        
        # Feature categories
        all_cols = set(unified_dataset.columns)
        base_cols = {'symbol', 'market', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'}
        feature_cols = all_cols - base_cols
        
        logger.info(f"Technical features created: {len(feature_cols)}")
        logger.info(f"Sample features: {list(sorted(feature_cols))[:10]}...")
        
        # Check for target variables
        target_cols = [c for c in unified_dataset.columns if c.startswith('y_') or c.startswith('ret_')]
        logger.info(f"Target variables: {target_cols}")
        
        logger.info("Feature engineering completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_train_stage():
    """Run model training stage."""
    logger = logging.getLogger(__name__)
    logger.info("Starting model training stage...")
    
    try:
        # Load processed dataset
        dataset_file = PROCESSED_DATA_DIR / "dataset.parquet"
        
        if not dataset_file.exists():
            logger.error(f"Dataset not found: {dataset_file}")
            logger.error("Please run feature engineering first: python run.py --stage features")
            return False
        
        # Load dataset
        logger.info("Loading processed dataset...")
        dataset = pd.read_parquet(dataset_file)
        
        # Initialize model trainer
        trainer = ModelTrainer()
        
        # Prepare data splits
        logger.info("Preparing train/validation/test splits...")
        train_df, val_df, test_df = trainer.prepare_data(dataset)
        
        # Train all models
        logger.info("Training all models...")
        training_results = trainer.train_all_models(train_df, val_df)
        
        # Evaluate on test set
        logger.info("Evaluating models on test set...")
        test_results = trainer.evaluate_on_test(test_df)
        
        # Save models
        timestamp = trainer.save_models()
        
        # Create predictions for simulation
        logger.info("Creating predictions for simulation...")
        predictions_df = create_predictions_dataframe(test_df, trainer)
        
        # Create timestamped results directory and save predictions
        results_dir = REPORTS_DIR / f"results_{timestamp}"
        results_dir.mkdir(exist_ok=True)
        pred_file = results_dir / f"predictions_{timestamp}.parquet"
        predictions_df.to_parquet(pred_file)
        logger.info(f"Saved predictions to {pred_file}")
        
        # Print summary
        logger.info("=== MODEL TRAINING SUMMARY ===")
        logger.info(f"Models trained: {list(training_results.keys())}")
        logger.info(f"Test set size: {len(test_df)}")
        logger.info(f"Models saved with timestamp: {timestamp}")
        
        # Print test performance
        logger.info("Test performance:")
        for model_name, metrics in test_results.items():
            logger.info(f"  {model_name}: AUC={metrics['test_auc']:.3f}, F1={metrics['test_f1']:.3f}")
        
        logger.info("Model training completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_simulate_stage():
    """Run simulation stage."""
    logger = logging.getLogger(__name__)
    logger.info("Starting simulation stage...")
    
    try:
        # Find latest predictions file in timestamped results directories
        pred_files = []
        results_dirs = list(REPORTS_DIR.glob("results_*"))
        
        for results_dir in results_dirs:
            pred_files.extend(results_dir.glob("predictions_*.parquet"))
        
        # Also check for old predictions files in the main reports directory (for backward compatibility)
        pred_files.extend(REPORTS_DIR.glob("predictions_*.parquet"))
        
        if not pred_files:
            logger.error("No predictions file found")
            logger.error("Please run model training first: python run.py --stage train")
            return False
        
        # Use most recent predictions file
        pred_file = max(pred_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Loading predictions from {pred_file}")
        
        # Load predictions
        predictions_df = pd.read_parquet(pred_file)
        
        # Extract timestamp from predictions file for consistency
        # predictions_YYYYMMDD_HHMMSS.parquet -> YYYYMMDD_HHMMSS
        pred_filename = pred_file.stem  # predictions_20250821_080330
        pred_timestamp = '_'.join(pred_filename.split('_')[1:])  # 20250821_080330
        
        # Try to load the latest trained models for feature importance
        model_trainer = None
        try:
            from src.models import ModelTrainer
            trainer_tmp = ModelTrainer()
            
            # Find latest model files
            model_files = list(MODELS_DIR.glob("*_*.pkl"))
            if model_files:
                logger.info(f"Looking for models with predictions timestamp: {pred_timestamp}")
                
                # First try to load models with matching timestamp
                trainer_tmp.models = {}
                loaded_models = []
                for model_name in ['logistic_regression', 'random_forest', 'xgboost']:
                    model_file = MODELS_DIR / f"{model_name}_{pred_timestamp}.pkl"
                    if model_file.exists():
                        import pickle
                        with open(model_file, 'rb') as f:
                            model = pickle.load(f)
                        trainer_tmp.models[model_name] = model
                        loaded_models.append(model_name)
                
                # If no models found with predictions timestamp, use latest available
                if not loaded_models:
                    logger.info("No models found with predictions timestamp, looking for latest models")
                    
                    # Find all full timestamps and pick the latest
                    timestamps = set()
                    for f in model_files:
                        # Extract full timestamp YYYYMMDD_HHMMSS
                        match = re.search(r'(\d{8}_\d{6})', f.name)
                        if match:
                            timestamps.add(match.group(1))
                    
                    if timestamps:
                        latest_timestamp = max(timestamps)
                        logger.info(f"Using latest available model timestamp: {latest_timestamp}")
                        
                        for model_name in ['logistic_regression', 'random_forest', 'xgboost']:
                            model_file = MODELS_DIR / f"{model_name}_{latest_timestamp}.pkl"
                            if model_file.exists():
                                import pickle
                                with open(model_file, 'rb') as f:
                                    model = pickle.load(f)
                                trainer_tmp.models[model_name] = model
                                loaded_models.append(model_name)
                
                logger.info(f"Loaded models: {loaded_models}")
                
                # Load feature names from metadata file using the same timestamp as models
                trainer_tmp.feature_names_ = None
                timestamp_to_use = latest_timestamp if 'latest_timestamp' in locals() else pred_timestamp
                
                for model_name in ['random_forest', 'xgboost', 'logistic_regression']:
                    metadata_file = MODELS_DIR / f"{model_name}_metadata_{timestamp_to_use}.json"
                    if metadata_file.exists():
                        logger.info(f"Loading feature names from {metadata_file}")
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            feature_names = metadata.get('feature_names', None) or metadata.get('feature_columns', None)
                            if feature_names is not None:
                                trainer_tmp.feature_names_ = feature_names
                                logger.info(f"Loaded {len(feature_names)} feature names from {model_name} metadata")
                
                if trainer_tmp.feature_names_ is None:
                    logger.warning("No feature names found in metadata files")
                
                if trainer_tmp.models:
                    model_trainer = trainer_tmp
                    logger.info(f"Loaded {len(trainer_tmp.models)} models for feature importance analysis")
                else:
                    logger.warning("No models found for feature importance analysis")
        except Exception as e:
            logger.warning(f"Could not load models for feature importance: {e}")
        
        # Initialize comprehensive evaluator with timestamped results
        evaluator = PerformanceEvaluator(run_timestamp=pred_timestamp)
        
        # Create timestamped results directory using the same timestamp
        results_dir = REPORTS_DIR / f"results_{pred_timestamp}"
        results_dir.mkdir(exist_ok=True)
        
        # Copy predictions file to the results directory for completeness
        predictions_copy = results_dir / f"predictions_{pred_timestamp}.parquet"
        if not predictions_copy.exists():
            import shutil
            shutil.copy2(pred_file, predictions_copy)
            logger.info(f"Copied predictions to {predictions_copy}")
        
        # Run complete simulation analysis
        logger.info("Running complete simulation analysis...")
        simulation_results = run_complete_simulation(predictions_df)
        
        # Extract model names from predictions columns
        model_names = [col.replace('y_proba_', '') for col in predictions_df.columns 
                      if col.startswith('y_proba_')]
        
        # Combine ML and financial metrics
        logger.info("Calculating comprehensive metrics...")
        combined_metrics = evaluator.combine_metrics(
            predictions_df, 
            model_names,
            investment_per_signal=100,
            transaction_cost=0.001,
            min_trades_threshold=50  # Minimum trades for valid statistics
        )
        
        # Generate summary report
        summary_report = evaluator.generate_summary_report(combined_metrics)
        
        # Create visualization plots
        evaluator.create_visualization_plots(combined_metrics, predictions_df, model_trainer)
        
        # Generate reports and plots
        logger.info("Creating simulation reports...")
        
        # 1. Threshold analysis summary
        if simulation_results['threshold_analysis']:
            for model_name, threshold_df in simulation_results['threshold_analysis'].items():
                logger.info(f"Threshold Analysis - {model_name}:")
                logger.info(f"\n{threshold_df.to_string(index=False)}")
        
        # 2. Model comparison
        if simulation_results['model_comparison'] is not None:
            logger.info("Model Comparison at Default Threshold:")
            logger.info(f"\n{simulation_results['model_comparison'].to_string(index=False)}")
        
        # 3. Export results using the same timestamp as predictions
        # results_dir is already created above with the correct timestamp
        
        # Export summary metrics
        if simulation_results['summary_metrics'] is not None:
            summary_file = results_dir / f"simulation_summary_{pred_timestamp}.csv"
            simulation_results['summary_metrics'].to_csv(summary_file, index=False)
            logger.info(f"Exported simulation summary to {summary_file}")
        
        # Export best configurations
        best_configs_file = results_dir / f"best_configs_{pred_timestamp}.json"
        with open(best_configs_file, 'w') as f:
            # Convert numpy types for JSON serialization
            best_configs_json = {}
            for model, config in simulation_results['best_configs'].items():
                best_configs_json[model] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                          for k, v in config.items()}
            json.dump(best_configs_json, f, indent=2)
        logger.info(f"Exported best configurations to {best_configs_file}")
        
        # Print summary
        logger.info("=== SIMULATION SUMMARY ===")
        logger.info(f"Models analyzed: {list(simulation_results['best_configs'].keys())}")
        
        logger.info("Best configurations by Sharpe ratio:")
        for model_name, config in simulation_results['best_configs'].items():
            logger.info(f"  {model_name}: threshold={config['threshold']:.2f}, "
                       f"Sharpe={config['sharpe_ratio']:.3f}, CAGR={config['cagr']:.1%}")
        
        logger.info("Simulation completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_all_stages(test_mode: bool = False, force_download: bool = False):
    """Run all pipeline stages."""
    logger = logging.getLogger(__name__)
    logger.info("Starting full pipeline...")
    
    # Stage 1: Data ingestion
    if not run_ingest_stage(test_mode, force_download):
        logger.error("Pipeline failed at ingestion stage")
        return False
    
    # Stage 2: Feature engineering
    if not run_features_stage():
        logger.error("Pipeline failed at feature engineering stage")
        return False
    
    # Stage 3: Model training
    if not run_train_stage():
        logger.error("Pipeline failed at model training stage")
        return False
    
    # Stage 4: Simulation
    if not run_simulate_stage():
        logger.error("Pipeline failed at simulation stage")
        return False
    
    logger.info("Full pipeline completed successfully!")
    return True

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Cross-Market 30D Directional Forecasting Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py --stage all                    # Run full pipeline
  python run.py --stage ingest --test-mode     # Test data download
  python run.py --stage ingest --force         # Force re-download
        """
    )
    
    parser.add_argument(
        '--stage',
        choices=['all', 'ingest', 'features', 'train', 'simulate'],
        default='all',
        help='Pipeline stage to run (default: all)'
    )
    
    parser.add_argument(
        '--test-mode',
        action='store_true',
        help='Run with subset of data for testing'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-download/rebuild (ignore cache)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = getattr(logging, args.log_level)
    setup_logging(log_level)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting pipeline with stage: {args.stage}")
    
    # Run requested stage
    success = False
    
    if args.stage == 'all':
        success = run_all_stages(args.test_mode, args.force)
    elif args.stage == 'ingest':
        success = run_ingest_stage(args.test_mode, args.force)
    elif args.stage == 'features':
        success = run_features_stage()
    elif args.stage == 'train':
        success = run_train_stage()
    elif args.stage == 'simulate':
        success = run_simulate_stage()
    
    if success:
        logger.info("Pipeline completed successfully!")
        sys.exit(0)
    else:
        logger.error("Pipeline failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()