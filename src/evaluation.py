"""
Comprehensive evaluation module for Cross-Market 30D Directional Forecasting project.

Integrates ML and financial metrics into unified reporting system.
Follows original notebook patterns for metric calculation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json

# Set matplotlib backend to avoid tkinter threading issues
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

from config import REPORTS_DIR, TIMESTAMP_FORMAT

class PerformanceEvaluator:
    """
    Comprehensive performance evaluation combining ML and financial metrics.
    
    Features:
    - ML classification metrics (ROC-AUC, F1, precision, recall)
    - Financial performance metrics (CAGR, Sharpe, Max Drawdown)
    - Threshold sensitivity analysis
    - Per-model and comparative analysis
    - Report generation with plots and summaries
    """
    
    def __init__(self, run_timestamp: Optional[str] = None):
        """
        Initialize PerformanceEvaluator.
        
        Args:
            run_timestamp: Timestamp for this evaluation run
        """
        self.run_timestamp = run_timestamp or datetime.now().strftime(TIMESTAMP_FORMAT)
        self.logger = logging.getLogger(__name__)
        
        # Create timestamped results directory
        self.results_dir = REPORTS_DIR / f"results_{self.run_timestamp}"
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize result storage
        self.ml_metrics = {}
        self.financial_metrics = {}
        self.threshold_analysis = {}
        
    def calculate_ml_metrics(self, 
                           y_true: np.ndarray, 
                           y_proba: np.ndarray, 
                           model_name: str,
                           thresholds: List[float] = [0.5, 0.55, 0.6, 0.65, 0.7]) -> Dict[str, float]:
        """
        Calculate comprehensive ML classification metrics following original notebook approach.
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities
            model_name: Name of the model
            thresholds: List of probability thresholds to evaluate
            
        Returns:
            Dictionary with ML metrics
        """
        metrics = {
            'model_name': model_name,
            'roc_auc': roc_auc_score(y_true, y_proba),
            'n_samples': len(y_true),
            'class_balance': y_true.mean()
        }
        
        # Calculate metrics for each threshold
        threshold_metrics = []
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            # Basic classification metrics
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / len(y_true)
            
            # Signal statistics (following original notebook approach)
            signal_rate = y_pred.sum() / len(y_pred)  # How often we generate signals
            hit_rate = y_true[y_pred == 1].mean() if y_pred.sum() > 0 else 0  # Success rate of our signals
            
            threshold_metrics.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'accuracy': accuracy,
                'signal_rate': signal_rate,
                'hit_rate': hit_rate,
                'n_signals': y_pred.sum(),
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
            })
        
        metrics['threshold_analysis'] = threshold_metrics
        self.ml_metrics[model_name] = metrics
        
        self.logger.info(f"ML metrics calculated for {model_name}: ROC-AUC={metrics['roc_auc']:.3f}")
        return metrics
    
    def calculate_financial_metrics(self,
                                  predictions_df: pd.DataFrame,
                                  model_name: str,
                                  investment_per_signal: float = 100,
                                  transaction_cost: float = 0.001,
                                  thresholds: List[float] = [0.5, 0.55, 0.6, 0.65, 0.7]) -> Dict[str, float]:
        """
        Calculate financial performance metrics following original notebook simulation approach.
        
        Args:
            predictions_df: DataFrame with predictions and returns
            model_name: Name of the model
            investment_per_signal: Fixed investment per signal ($)
            transaction_cost: Transaction cost rate (e.g., 0.001 = 0.1%)
            thresholds: List of probability thresholds to evaluate
            
        Returns:
            Dictionary with financial metrics
        """
        y_proba_col = f'y_proba_{model_name}'
        if y_proba_col not in predictions_df.columns:
            self.logger.error(f"Probability column {y_proba_col} not found")
            return {}
        
        financial_results = []
        
        for threshold in thresholds:
            # Generate signals following original notebook approach
            signals = (predictions_df[y_proba_col] >= threshold).astype(int)
            trades_df = predictions_df[signals == 1].copy()
            
            if len(trades_df) == 0:
                # No trades generated
                financial_results.append({
                    'threshold': threshold,
                    'total_trades': 0,
                    'cagr': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'hit_rate': 0.0,
                    'profit_factor': 0.0,
                    'total_return': 0.0,
                    'avg_trade_return': 0.0
                })
                continue
            
            # Calculate trade P&L following original notebook approach:
            # sim1_gross_rev = prediction * 100 * (growth_future_30d - 1)
            # sim1_fees = -prediction * 100 * 0.002  # 0.2% total transaction cost
            # sim1_net_rev = gross_rev + fees
            
            trades_df['gross_pnl'] = investment_per_signal * (trades_df['growth_future_30d'] - 1)
            trades_df['transaction_costs'] = -investment_per_signal * (transaction_cost * 2)  # Buy + sell
            trades_df['net_pnl'] = trades_df['gross_pnl'] + trades_df['transaction_costs']
            trades_df['trade_success'] = (trades_df['growth_future_30d'] > 1.0).astype(int)
            
            # Aggregate daily P&L
            daily_pnl = trades_df.groupby(trades_df.index)['net_pnl'].sum().sort_index()
            
            # Calculate performance metrics following original notebook approach
            total_trades = len(trades_df)
            total_net_pnl = trades_df['net_pnl'].sum()
            total_gross_pnl = trades_df['gross_pnl'].sum()
            hit_rate = trades_df['trade_success'].mean()
            
            # CAGR calculation (following original notebook)
            # Estimate required capital based on daily investment patterns
            daily_trades = trades_df.groupby(trades_df.index).size()
            avg_daily_trades = daily_trades.mean()
            q75_daily_trades = daily_trades.quantile(0.75)
            estimated_capital = investment_per_signal * 30 * q75_daily_trades  # 30 days holding period
            
            if estimated_capital > 0:
                total_return = total_net_pnl / estimated_capital
                
                # Time period calculation
                start_date = trades_df.index.min()
                end_date = trades_df.index.max()
                years = (end_date - start_date).days / 365.25
                
                if years > 0:
                    cagr = ((estimated_capital + total_net_pnl) / estimated_capital) ** (1/years) - 1
                else:
                    cagr = 0.0
            else:
                total_return = 0.0
                cagr = 0.0
            
            # Sharpe ratio calculation
            if len(daily_pnl) > 1:
                daily_returns = daily_pnl / estimated_capital if estimated_capital > 0 else daily_pnl / investment_per_signal
                sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
            else:
                sharpe_ratio = 0.0
            
            # Max drawdown calculation
            cumulative_pnl = daily_pnl.cumsum()
            running_max = cumulative_pnl.expanding().max()
            drawdown = (cumulative_pnl - running_max) / estimated_capital if estimated_capital > 0 else (cumulative_pnl - running_max)
            max_drawdown = drawdown.min()
            
            # Profit factor
            winning_trades = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].sum()
            losing_trades = abs(trades_df[trades_df['net_pnl'] < 0]['net_pnl'].sum())
            profit_factor = winning_trades / losing_trades if losing_trades > 0 else np.inf if winning_trades > 0 else 0
            
            financial_results.append({
                'threshold': threshold,
                'total_trades': total_trades,
                'cagr': cagr * 100,  # Convert to percentage
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown * 100,  # Convert to percentage
                'hit_rate': hit_rate * 100,  # Convert to percentage
                'profit_factor': profit_factor,
                'total_return': total_return * 100,  # Convert to percentage
                'avg_trade_return': trades_df['net_pnl'].mean(),
                'estimated_capital': estimated_capital,
                'total_net_pnl': total_net_pnl,
                'avg_daily_trades': avg_daily_trades
            })
        
        # Store results
        financial_metrics = {
            'model_name': model_name,
            'threshold_analysis': financial_results
        }
        
        self.financial_metrics[model_name] = financial_metrics
        
        # Log best performance
        best_by_sharpe = max(financial_results, key=lambda x: x['sharpe_ratio'])
        self.logger.info(f"Financial metrics calculated for {model_name}: Best Sharpe={best_by_sharpe['sharpe_ratio']:.2f} at threshold={best_by_sharpe['threshold']}")
        
        return financial_metrics
    
    def combine_metrics(self, 
                       predictions_df: pd.DataFrame,
                       models: List[str],
                       investment_per_signal: float = 100,
                       transaction_cost: float = 0.001,
                       min_trades_threshold: int = 50) -> pd.DataFrame:
        """
        Calculate and combine ML and financial metrics for all models.
        
        Args:
            predictions_df: DataFrame with predictions and true labels
            models: List of model names to evaluate
            investment_per_signal: Investment per signal
            transaction_cost: Transaction cost rate
            min_trades_threshold: Minimum number of trades required for valid statistics
            
        Returns:
            Combined metrics DataFrame
        """
        combined_results = []
        
        for model_name in models:
            y_proba_col = f'y_proba_{model_name}'
            if y_proba_col not in predictions_df.columns:
                self.logger.warning(f"Skipping {model_name}: column {y_proba_col} not found")
                continue
            
            # Calculate ML metrics
            ml_metrics = self.calculate_ml_metrics(
                predictions_df['y_true'].values,
                predictions_df[y_proba_col].values,
                model_name
            )
            
            # Calculate financial metrics
            financial_metrics = self.calculate_financial_metrics(
                predictions_df,
                model_name,
                investment_per_signal,
                transaction_cost
            )
            
            # Combine metrics for each threshold
            for i, threshold in enumerate([0.5, 0.55, 0.6, 0.65, 0.7]):
                ml_data = ml_metrics['threshold_analysis'][i]
                fin_data = financial_metrics['threshold_analysis'][i]
                
                # Mark results with insufficient trades for statistical validity
                is_valid_sample = fin_data['total_trades'] >= min_trades_threshold
                
                combined_row = {
                    'model_name': model_name,
                    'threshold': threshold,
                    # ML metrics
                    'roc_auc': ml_metrics['roc_auc'],
                    'precision': ml_data['precision'],
                    'recall': ml_data['recall'],
                    'f1_score': ml_data['f1_score'],
                    'accuracy': ml_data['accuracy'],
                    'signal_rate': ml_data['signal_rate'],
                    'ml_hit_rate': ml_data['hit_rate'],
                    'n_signals': ml_data['n_signals'],
                    # Financial metrics (marked if insufficient trades)
                    'total_trades': fin_data['total_trades'],
                    'cagr': fin_data['cagr'] if is_valid_sample else np.nan,
                    'sharpe_ratio': fin_data['sharpe_ratio'] if is_valid_sample else np.nan,
                    'max_drawdown': fin_data['max_drawdown'] if is_valid_sample else np.nan,
                    'hit_rate': fin_data['hit_rate'],
                    'profit_factor': fin_data['profit_factor'] if is_valid_sample else np.nan,
                    'total_return': fin_data['total_return'] if is_valid_sample else np.nan,
                    'avg_trade_return': fin_data['avg_trade_return'],
                    'is_valid_sample': is_valid_sample,
                    'min_trades_required': min_trades_threshold
                }
                
                combined_results.append(combined_row)
        
        combined_df = pd.DataFrame(combined_results)
        
        # Save to file
        output_file = self.results_dir / "combined_metrics.csv"
        combined_df.to_csv(output_file, index=False)
        self.logger.info(f"Combined metrics saved to {output_file}")
        
        return combined_df
    
    def generate_summary_report(self, combined_metrics: pd.DataFrame) -> Dict:
        """
        Generate executive summary report.
        
        Args:
            combined_metrics: DataFrame with combined ML and financial metrics
            
        Returns:
            Summary report dictionary
        """
        summary = {
            'run_timestamp': self.run_timestamp,
            'evaluation_date': datetime.now().isoformat(),
            'models_evaluated': combined_metrics['model_name'].unique().tolist(),
            'thresholds_tested': sorted(combined_metrics['threshold'].unique()),
            'total_observations': len(combined_metrics),
        }
        
        # Best performers by different criteria (only from valid samples)
        valid_metrics = combined_metrics[combined_metrics['is_valid_sample'] == True]
        
        if len(valid_metrics) > 0:
            summary['best_performers'] = {
                'best_by_sharpe': self._get_best_config(valid_metrics, 'sharpe_ratio'),
                'best_by_cagr': self._get_best_config(valid_metrics, 'cagr'),
                'best_by_f1': self._get_best_config(combined_metrics, 'f1_score'),  # F1 always valid
                'best_by_roc_auc': self._get_best_config(combined_metrics, 'roc_auc')  # ROC-AUC always valid
            }
        else:
            summary['best_performers'] = {
                'best_by_sharpe': None,
                'best_by_cagr': None,
                'best_by_f1': self._get_best_config(combined_metrics, 'f1_score'),
                'best_by_roc_auc': self._get_best_config(combined_metrics, 'roc_auc')
            }
        
        # Model comparison at default threshold (0.6)
        default_threshold_results = combined_metrics[combined_metrics['threshold'] == 0.6]
        summary['default_threshold_comparison'] = default_threshold_results.to_dict('records')
        
        # Statistics with validity indicators
        summary['statistics'] = {
            'avg_roc_auc': combined_metrics['roc_auc'].mean(),
            'avg_sharpe_ratio': valid_metrics['sharpe_ratio'].mean() if len(valid_metrics) > 0 else np.nan,
            'max_cagr': valid_metrics['cagr'].max() if len(valid_metrics) > 0 else np.nan,
            'min_max_drawdown': valid_metrics['max_drawdown'].min() if len(valid_metrics) > 0 else np.nan,
            'valid_samples_count': len(valid_metrics),
            'total_samples_count': len(combined_metrics),
            'validity_rate': len(valid_metrics) / len(combined_metrics) if len(combined_metrics) > 0 else 0
        }
        
        # Save summary
        summary_file = self.results_dir / "summary_report.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Summary report saved to {summary_file}")
        return summary
    
    def _get_best_config(self, df: pd.DataFrame, metric: str) -> Dict:
        """Get best configuration for a specific metric, handling NaN values."""
        # Filter out NaN values for the metric
        valid_df = df.dropna(subset=[metric])
        
        if len(valid_df) == 0:
            return {
                'model_name': None,
                'threshold': None,
                'value': np.nan,
                'cagr': np.nan,
                'sharpe_ratio': np.nan,
                'total_trades': 0
            }
        
        best_row = valid_df.loc[valid_df[metric].idxmax()]
        return {
            'model_name': best_row['model_name'],
            'threshold': best_row['threshold'],
            'value': best_row[metric],
            'cagr': best_row.get('cagr', np.nan),
            'sharpe_ratio': best_row.get('sharpe_ratio', np.nan),
            'total_trades': best_row['total_trades']
        }
    
    def create_visualization_plots(self, combined_metrics: pd.DataFrame, 
                                 predictions_df: pd.DataFrame = None,
                                 model_trainer = None):
        """
        Create comprehensive visualization plots with validity filtering.
        
        Args:
            combined_metrics: DataFrame with combined metrics
            predictions_df: DataFrame with predictions for equity curves
            model_trainer: Trained models for feature importance
        """
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Filter for valid samples (sufficient trades)
        valid_metrics = combined_metrics[combined_metrics['is_valid_sample'] == True]
        
        # 1. Main Performance Analysis (4 subplots)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Model Performance Analysis - {self.run_timestamp}', fontsize=16)
        
        # Sharpe ratio by threshold (only valid samples)
        for model in combined_metrics['model_name'].unique():
            model_data = combined_metrics[combined_metrics['model_name'] == model]
            valid_model_data = model_data[model_data['is_valid_sample'] == True]
            invalid_model_data = model_data[model_data['is_valid_sample'] == False]
            
            # Plot valid data
            if len(valid_model_data) > 0:
                axes[0, 0].plot(valid_model_data['threshold'], valid_model_data['sharpe_ratio'], 
                               marker='o', label=f'{model} (valid)', linewidth=2, alpha=0.8)
            
            # Mark invalid data points
            if len(invalid_model_data) > 0:
                axes[0, 0].scatter(invalid_model_data['threshold'], [0] * len(invalid_model_data), 
                                  marker='x', s=80, color='red', alpha=0.6, 
                                  label=f'{model} (insufficient trades)' if model == combined_metrics['model_name'].unique()[0] else "")
        
        axes[0, 0].set_title('Sharpe Ratio by Threshold')
        axes[0, 0].set_xlabel('Probability Threshold')
        axes[0, 0].set_ylabel('Sharpe Ratio')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Good Sharpe (1.0)')
        
        # CAGR by threshold (only valid samples)
        for model in combined_metrics['model_name'].unique():
            model_data = combined_metrics[combined_metrics['model_name'] == model]
            valid_model_data = model_data[model_data['is_valid_sample'] == True]
            
            if len(valid_model_data) > 0:
                axes[0, 1].plot(valid_model_data['threshold'], valid_model_data['cagr'], 
                               marker='s', label=model, linewidth=2)
        axes[0, 1].set_title('CAGR by Threshold (Valid Samples Only)')
        axes[0, 1].set_xlabel('Probability Threshold')
        axes[0, 1].set_ylabel('CAGR (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=10.0, color='gray', linestyle='--', alpha=0.5, label='Good CAGR (10%)')
        
        # ROC-AUC vs Sharpe ratio scatter
        for model in combined_metrics['model_name'].unique():
            valid_model_data = valid_metrics[valid_metrics['model_name'] == model]
            if len(valid_model_data) > 0:
                axes[1, 0].scatter(valid_model_data['roc_auc'], valid_model_data['sharpe_ratio'], 
                                  label=model, s=60, alpha=0.7)
        axes[1, 0].set_title('ROC-AUC vs Sharpe Ratio (Valid Samples)')
        axes[1, 0].set_xlabel('ROC-AUC')
        axes[1, 0].set_ylabel('Sharpe Ratio')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Number of trades by threshold (with validity indicators)
        for model in combined_metrics['model_name'].unique():
            model_data = combined_metrics[combined_metrics['model_name'] == model]
            axes[1, 1].plot(model_data['threshold'], model_data['total_trades'], 
                           marker='^', label=model, linewidth=2)
            
            # Add horizontal line for minimum trades threshold
            min_threshold = model_data['min_trades_required'].iloc[0]
            axes[1, 1].axhline(y=min_threshold, color='red', linestyle='--', alpha=0.5)
        
        axes[1, 1].set_title('Number of Trades by Threshold')
        axes[1, 1].set_xlabel('Probability Threshold')
        axes[1, 1].set_ylabel('Total Trades')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].text(0.52, min_threshold + 10, f'Min required: {min_threshold}', color='red', fontsize=9)
        
        plt.tight_layout()
        
        # Save main plot
        plot_file = self.results_dir / "performance_analysis.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Additional Detailed Analysis Plots
        self._create_detailed_analysis_plots(combined_metrics, valid_metrics)
        
        # 3. Equity Curves (if predictions provided)
        if predictions_df is not None:
            self._create_equity_curves_plot(predictions_df, combined_metrics)
        
        # 4. Feature Importance (if model trainer provided)
        if model_trainer is not None:
            self._create_feature_importance_plot(model_trainer)
        
        # Force close all matplotlib figures to prevent tkinter threading issues
        plt.close('all')
        
        self.logger.info(f"Performance plots saved to {self.results_dir}")
    
    def _create_detailed_analysis_plots(self, combined_metrics: pd.DataFrame, valid_metrics: pd.DataFrame):
        """Create additional detailed analysis plots."""
        
        # Plot 2: ML Metrics Analysis
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'ML Metrics Analysis - {self.run_timestamp}', fontsize=16)
        
        # Precision vs Recall
        for model in combined_metrics['model_name'].unique():
            model_data = combined_metrics[combined_metrics['model_name'] == model]
            axes[0, 0].plot(model_data['recall'], model_data['precision'], 
                           marker='o', label=model, linewidth=2)
        axes[0, 0].set_title('Precision vs Recall')
        axes[0, 0].set_xlabel('Recall')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # F1 Score by Threshold
        for model in combined_metrics['model_name'].unique():
            model_data = combined_metrics[combined_metrics['model_name'] == model]
            axes[0, 1].plot(model_data['threshold'], model_data['f1_score'], 
                           marker='s', label=model, linewidth=2)
        axes[0, 1].set_title('F1 Score by Threshold')
        axes[0, 1].set_xlabel('Probability Threshold')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Signal Rate vs Hit Rate
        for model in combined_metrics['model_name'].unique():
            model_data = combined_metrics[combined_metrics['model_name'] == model]
            axes[1, 0].plot(model_data['signal_rate'], model_data['ml_hit_rate'], 
                           marker='^', label=model, linewidth=2)
        axes[1, 0].set_title('Signal Rate vs ML Hit Rate')
        axes[1, 0].set_xlabel('Signal Rate (% of time trading)')
        axes[1, 0].set_ylabel('ML Hit Rate (% correct predictions)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # ROC-AUC comparison (bar chart)
        models = combined_metrics['model_name'].unique()
        roc_aucs = [combined_metrics[combined_metrics['model_name'] == model]['roc_auc'].iloc[0] for model in models]
        bars = axes[1, 1].bar(models, roc_aucs, alpha=0.7)
        axes[1, 1].set_title('ROC-AUC by Model')
        axes[1, 1].set_ylabel('ROC-AUC')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random (0.5)')
        
        # Add value labels on bars
        for bar, value in zip(bars, roc_aucs):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        ml_plot_file = self.results_dir / "ml_metrics_analysis.png"
        plt.savefig(ml_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3: Financial Metrics Deep Dive (only valid samples)
        if len(valid_metrics) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'Financial Performance Deep Dive - {self.run_timestamp}', fontsize=16)
            
            # Max Drawdown vs CAGR
            for model in valid_metrics['model_name'].unique():
                model_data = valid_metrics[valid_metrics['model_name'] == model]
                axes[0, 0].scatter(model_data['max_drawdown'], model_data['cagr'], 
                                  label=model, s=80, alpha=0.7)
            axes[0, 0].set_title('CAGR vs Max Drawdown (Valid Samples)')
            axes[0, 0].set_xlabel('Max Drawdown (%)')
            axes[0, 0].set_ylabel('CAGR (%)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Profit Factor by Threshold
            for model in valid_metrics['model_name'].unique():
                model_data = valid_metrics[valid_metrics['model_name'] == model]
                # Cap profit factor for visualization
                capped_pf = model_data['profit_factor'].clip(upper=10)
                axes[0, 1].plot(model_data['threshold'], capped_pf, 
                               marker='o', label=model, linewidth=2)
            axes[0, 1].set_title('Profit Factor by Threshold (Capped at 10)')
            axes[0, 1].set_xlabel('Probability Threshold')
            axes[0, 1].set_ylabel('Profit Factor')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Break-even')
            
            # Hit Rate Financial vs ML Hit Rate
            for model in valid_metrics['model_name'].unique():
                model_data = valid_metrics[valid_metrics['model_name'] == model]
                axes[1, 0].scatter(model_data['ml_hit_rate'], model_data['hit_rate'], 
                                  label=model, s=80, alpha=0.7)
            axes[1, 0].set_title('Financial Hit Rate vs ML Hit Rate')
            axes[1, 0].set_xlabel('ML Hit Rate (%)')
            axes[1, 0].set_ylabel('Financial Hit Rate (%)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].plot([0, 100], [0, 100], 'k--', alpha=0.5, label='Perfect correlation')
            
            # Average Trade Return Distribution
            all_trade_returns = valid_metrics['avg_trade_return'].dropna()
            axes[1, 1].hist(all_trade_returns, bins=15, alpha=0.7, edgecolor='black')
            axes[1, 1].axvline(all_trade_returns.mean(), color='red', linestyle='--', 
                              label=f'Mean: ${all_trade_returns.mean():.2f}')
            axes[1, 1].set_title('Average Trade Return Distribution')
            axes[1, 1].set_xlabel('Average Trade Return ($)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            financial_plot_file = self.results_dir / "financial_deep_dive.png"
            plt.savefig(financial_plot_file, dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_equity_curves_plot(self, predictions_df: pd.DataFrame, combined_metrics: pd.DataFrame):
        """Create equity curves plot for each model with baseline comparisons."""
        from src.config import TICKER_MARKETS
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))
        fig.suptitle(f'Equity Curves Analysis - {self.run_timestamp}', fontsize=16)
        
        # === PLOT 1: Model Strategies vs Baselines ===
        models = combined_metrics['model_name'].unique()
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        equity_data = {}  # For CSV export
        
        # Calculate baseline equity curves (normalized to $10,000 start)
        baseline_curves = self._calculate_baseline_curves(predictions_df)
        
        # Plot baseline curves (normalized to same scale)
        baseline_plotted = False
        for name, curve in baseline_curves.items():
            if len(curve) > 0:
                # Normalize baseline to start from $10,000 like model strategies
                normalized_curve = 10000 * (curve / curve.iloc[0])
                
                # Calculate metrics for legend
                total_return = (normalized_curve.iloc[-1] / 10000) - 1
                years = (normalized_curve.index[-1] - normalized_curve.index[0]).days / 365.25
                cagr = ((normalized_curve.iloc[-1] / 10000) ** (1/years) - 1) if years > 0 else 0
                
                daily_returns = normalized_curve.pct_change().dropna()
                sharpe = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
                
                label = f'{name} (CAGR: {cagr:.1%}, Sharpe: {sharpe:.2f})'
                ax1.plot(normalized_curve.index, normalized_curve.values, 
                        label=label, linestyle='--', alpha=0.7, linewidth=1.5)
                equity_data[f'baseline_{name}'] = normalized_curve
                baseline_plotted = True
        
        # Plot model equity curves
        models_plotted = 0
        for i, model in enumerate(models):
            model_metrics = combined_metrics[combined_metrics['model_name'] == model]
            valid_model_metrics = model_metrics[model_metrics['is_valid_sample'] == True]
            
            if len(valid_model_metrics) == 0:
                self.logger.warning(f"No valid metrics for model {model}, skipping equity curve")
                continue
                
            best_threshold = valid_model_metrics.loc[valid_model_metrics['sharpe_ratio'].idxmax(), 'threshold']
            
            # Generate signals for best threshold
            y_proba_col = f'y_proba_{model}'
            if y_proba_col not in predictions_df.columns:
                self.logger.warning(f"Column {y_proba_col} not found in predictions_df")
                continue
                
            signals = (predictions_df[y_proba_col] >= best_threshold).astype(int)
            trades_df = predictions_df[signals == 1].copy()
            
            if len(trades_df) == 0:
                self.logger.warning(f"No trades generated for {model} at threshold {best_threshold}")
                continue
            
            # Calculate model equity curve following original notebook approach
            trades_df['gross_pnl'] = 100 * (trades_df['growth_future_30d'] - 1)
            trades_df['transaction_costs'] = -100 * 0.002  # 0.2% total transaction cost
            trades_df['net_pnl'] = trades_df['gross_pnl'] + trades_df['transaction_costs']
            
            # Group by date and sum PnL
            daily_pnl = trades_df.groupby(trades_df.index)['net_pnl'].sum().sort_index()
            
            # Create equity curve starting from initial capital
            equity_curve = 10000 + daily_pnl.cumsum()
            
            if len(equity_curve) > 0:
                # Calculate metrics for legend
                total_return = (equity_curve.iloc[-1] / 10000) - 1
                years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
                cagr = ((equity_curve.iloc[-1] / 10000) ** (1/years) - 1) if years > 0 else 0
                
                daily_returns = equity_curve.pct_change().dropna()
                sharpe = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
                
                label = f'{model} (t={best_threshold:.2f}, CAGR: {cagr:.1%}, Sharpe: {sharpe:.2f})'
                ax1.plot(equity_curve.index, equity_curve.values, 
                       label=label, linewidth=2.5, color=colors[i % len(colors)])
                equity_data[f'model_{model}'] = equity_curve
                models_plotted += 1
                
                self.logger.info(f"Model {model}: CAGR={cagr:.1%}, Sharpe={sharpe:.2f}, Trades={len(trades_df)}")
        
        if models_plotted == 0 and not baseline_plotted:
            # If no data to plot, add a placeholder
            ax1.text(0.5, 0.5, 'No equity data available', transform=ax1.transAxes, 
                    ha='center', va='center', fontsize=16, alpha=0.6)
        
        ax1.axhline(y=10000, color='gray', linestyle=':', alpha=0.5, label='Initial Capital ($10,000)')
        ax1.set_title('Model Strategies vs Market Baselines (Normalized to $10,000)')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # === PLOT 2: Regional Breakdown ===
        regional_curves = self._calculate_regional_curves(predictions_df)
        
        regions_plotted = 0
        for name, curve in regional_curves.items():
            if len(curve) > 0:
                # Normalize regional curves to start from $10,000
                normalized_curve = 10000 * (curve / curve.iloc[0])
                
                # Check if normalized curve has extreme values
                max_normalized = normalized_curve.max()
                if max_normalized > 100000:  # If still extreme after normalization
                    # Apply log scaling or cap extreme values
                    self.logger.warning(f"Region {name} has extreme values (max: ${max_normalized:,.0f}), applying cap")
                    # Cap the curve at reasonable maximum (e.g., 10x growth = $100,000)
                    normalized_curve = normalized_curve.clip(upper=100000)
                
                # Calculate metrics for legend
                total_return = (normalized_curve.iloc[-1] / 10000) - 1
                years = (normalized_curve.index[-1] - normalized_curve.index[0]).days / 365.25
                cagr = ((normalized_curve.iloc[-1] / 10000) ** (1/years) - 1) if years > 0 else 0
                
                daily_returns = normalized_curve.pct_change().dropna()
                sharpe = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
                
                label = f'{name} (CAGR: {cagr:.1%}, Sharpe: {sharpe:.2f})'
                ax2.plot(normalized_curve.index, normalized_curve.values, 
                        label=label, linewidth=2, alpha=0.8)
                equity_data[f'region_{name}'] = normalized_curve
                regions_plotted += 1
        
        if regions_plotted == 0:
            ax2.text(0.5, 0.5, 'No regional data available', transform=ax2.transAxes, 
                    ha='center', va='center', fontsize=16, alpha=0.6)
        
        ax2.axhline(y=10000, color='gray', linestyle=':', alpha=0.5, label='Initial Capital ($10,000)')
        ax2.set_title('Regional Market Performance (Normalized to $10,000)')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Format dates
        import matplotlib.dates as mdates
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        equity_plot_file = self.results_dir / "equity_curves.png"
        plt.savefig(equity_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save equity curves to CSV
        if equity_data:
            self._save_equity_curves_csv(equity_data)
        
        # Calculate and save baseline metrics
        self._save_baseline_metrics(baseline_curves, regional_curves)
    
    def _calculate_baseline_curves(self, predictions_df: pd.DataFrame):
        """
        Calculate baseline buy-and-hold equity curves following original notebook approach.
        
        Market Average: Daily rebalanced equal-weight portfolio (more volatile, higher turnover)
        Buy & Hold: Monthly rebalanced equal-weight portfolio (less volatile, lower turnover)
        """
        baseline_curves = {}
        
        # Check if we have the required columns
        if 'growth_future_30d' not in predictions_df.columns:
            self.logger.warning("growth_future_30d column not found for baseline calculation")
            return baseline_curves
        
        # Reset index to make date a column if needed
        if predictions_df.index.name == 'date' or 'Date' in predictions_df.index.names:
            df_with_date = predictions_df.reset_index()
        else:
            df_with_date = predictions_df.copy()
            
        # Ensure we have a Date column
        if 'Date' not in df_with_date.columns and 'date' in df_with_date.columns:
            df_with_date['Date'] = df_with_date['date']
        elif 'Date' not in df_with_date.columns:
            self.logger.warning("No Date column found for baseline calculation")
            return baseline_curves
        
        # Convert Date column to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df_with_date['Date']):
            df_with_date['Date'] = pd.to_datetime(df_with_date['Date'])
        
        # Calculate baseline strategies
        try:
            # Strategy 1: Market Average (Equal-weight daily rebalanced)
            # Use 30-day forward returns but rebalance daily for higher turnover
            daily_data = df_with_date.groupby(['Date', 'symbol'])['growth_future_30d'].first().unstack(fill_value=np.nan)
            
            if len(daily_data) > 1:
                # Convert 30-day growth ratios to daily returns for daily rebalancing
                # growth_future_30d is already a ratio (new_price/old_price), so convert to daily equivalent
                daily_returns = daily_data.apply(lambda x: (x ** (1/21)) - 1, axis=1)  # 21 trading days ≈ 30 calendar days
                
                # Equal-weight portfolio daily returns (mean across all available assets)
                market_daily_returns = daily_returns.mean(axis=1, skipna=True)
                
                # Build equity curve with daily compounding
                market_equity = 10000 * (1 + market_daily_returns).cumprod()
                baseline_curves['Market Average (Daily Rebal.)'] = market_equity
                
                final_value = market_equity.iloc[-1]
                days_total = (market_equity.index[-1] - market_equity.index[0]).days
                years = days_total / 365.25
                cagr = ((final_value / 10000) ** (1/years) - 1) if years > 0 else 0
                
                self.logger.info(f"Market Average baseline: {len(market_equity)} periods, "
                               f"Final value: ${final_value:,.0f}, CAGR: {cagr:.1%}")
            
            # Strategy 2: Buy & Hold (Monthly rebalanced, more realistic)
            # Use monthly aggregation of 30-day forward returns
            monthly_data = df_with_date.copy()
            monthly_data['YearMonth'] = monthly_data['Date'].dt.to_period('M')
            
            # Get last observation per month per symbol
            monthly_last = monthly_data.groupby(['YearMonth', 'symbol'])['growth_future_30d'].last().unstack(fill_value=np.nan)
            
            if len(monthly_last) > 1:
                # Convert to monthly returns: use 30-day forward growth but only rebalance monthly
                monthly_returns = monthly_last.apply(lambda x: x - 1, axis=1)  # Convert ratio to return
                
                # Equal-weight portfolio monthly returns
                portfolio_monthly_returns = monthly_returns.mean(axis=1, skipna=True)
                
                # Build equity curve with monthly compounding
                buyhold_equity = 10000 * (1 + portfolio_monthly_returns).cumprod()
                
                # Convert YearMonth index back to timestamp for plotting
                buyhold_equity.index = buyhold_equity.index.to_timestamp()
                baseline_curves['Buy & Hold (Monthly Rebal.)'] = buyhold_equity
                
                final_value = buyhold_equity.iloc[-1]
                periods = len(buyhold_equity)
                years = periods / 12  # Monthly periods
                cagr = ((final_value / 10000) ** (1/years) - 1) if years > 0 else 0
                
                self.logger.info(f"Buy & Hold baseline: {len(buyhold_equity)} periods, "
                               f"Final value: ${final_value:,.0f}, CAGR: {cagr:.1%}")
                
        except Exception as e:
            self.logger.error(f"Error calculating baseline curves: {e}")
            import traceback
            traceback.print_exc()
        
        return baseline_curves
    
    def _calculate_regional_curves(self, predictions_df: pd.DataFrame):
        """Calculate regional equity curves based on TICKER_MARKETS."""
        try:
            from src.config import TICKER_MARKETS
        except ImportError:
            self.logger.warning("TICKER_MARKETS not available for regional analysis")
            return {}
        
        regional_curves = {}
        
        # Check required columns
        if 'growth_future_30d' not in predictions_df.columns:
            self.logger.warning("growth_future_30d column not found for regional calculation")
            return regional_curves
        
        # Reset index to make date a column if needed
        if predictions_df.index.name == 'date' or 'Date' in predictions_df.index.names:
            df_with_date = predictions_df.reset_index()
        else:
            df_with_date = predictions_df.copy()
            
        # Ensure we have a Date column and symbol column
        if 'Date' not in df_with_date.columns and 'date' in df_with_date.columns:
            df_with_date['Date'] = df_with_date['date']
        elif 'Date' not in df_with_date.columns:
            self.logger.warning("No Date column found for regional calculation")
            return regional_curves
            
        # Check for symbol column
        symbol_col = None
        for col in ['symbol', 'Symbol', 'ticker', 'Ticker']:
            if col in df_with_date.columns:
                symbol_col = col
                break
        
        if symbol_col is None:
            self.logger.warning("No symbol column found for regional calculation")
            return regional_curves
        
        # Convert Date column to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df_with_date['Date']):
            df_with_date['Date'] = pd.to_datetime(df_with_date['Date'])
        
        # Map symbols to regions and calculate regional performance
        for region in ['US', 'EU', 'ASIA', 'COMMODITY', 'SECTOR', 'REGIONAL']:
            region_tickers = [ticker for ticker, market in TICKER_MARKETS.items() if market == region]
            
            if not region_tickers:
                continue
                
            # Filter predictions for this region
            region_data = df_with_date[df_with_date[symbol_col].isin(region_tickers)]
            if len(region_data) == 0:
                self.logger.debug(f"No data found for region {region}")
                continue
                
            # Calculate equal-weight regional performance using periodic returns
            # Convert growth ratios to returns and use safer calculation
            region_data_clean = region_data.copy()
            
            # Limit extreme values to prevent overflow
            region_data_clean['return_30d'] = np.clip(region_data_clean['growth_future_30d'] - 1, -0.9, 10.0)  # Cap at -90% to +1000%
            
            regional_returns = region_data_clean.groupby('Date')['return_30d'].mean()
            
            if len(regional_returns) > 0:
                # Build equity curve with capped returns
                regional_equity = 10000 * (1 + regional_returns).cumprod()
                regional_curves[region] = regional_equity
                
                # Calculate simple metrics for logging
                total_return = (regional_equity.iloc[-1] / 10000) - 1
                years = (regional_equity.index[-1] - regional_equity.index[0]).days / 365.25
                cagr = ((regional_equity.iloc[-1] / 10000) ** (1/years) - 1) if years > 0 else 0
                
                self.logger.info(f"Region {region}: {len(region_tickers)} tickers, "
                               f"{len(regional_equity)} periods, "
                               f"Total return: {total_return:.1%}, CAGR: {cagr:.1%}")
        
        return regional_curves
    
    def _calculate_rebalanced_curve(self, df_with_date: pd.DataFrame):
        """Calculate monthly rebalanced equal-weight curve."""
        # This method is no longer used in the updated implementation
        return pd.Series([], dtype=float)
    
    def _save_equity_curves_csv(self, equity_data: dict):
        """Save all equity curves to CSV files."""
        # Combine all curves into single DataFrame
        equity_df = pd.DataFrame(equity_data)
        equity_df.index.name = 'date'
        
        # Save to CSV
        equity_csv_file = self.results_dir / "equity_curves_data.csv"
        equity_df.to_csv(equity_csv_file)
        
        self.logger.info(f"Equity curves data saved to {equity_csv_file}")
    
    def _save_baseline_metrics(self, baseline_curves: dict, regional_curves: dict):
        """Calculate and save CAGR and Sharpe for baseline curves."""
        all_curves = {**baseline_curves, **regional_curves}
        baseline_metrics = []
        
        for name, curve in all_curves.items():
            if len(curve) < 2:
                continue
                
            # Calculate returns
            returns = curve.pct_change().dropna()
            
            if len(returns) == 0:
                continue
            
            # Calculate metrics
            total_return = (curve.iloc[-1] / curve.iloc[0]) - 1
            years = (curve.index[-1] - curve.index[0]).days / 365.25
            cagr = ((curve.iloc[-1] / curve.iloc[0]) ** (1/years) - 1) if years > 0 else 0
            
            annual_returns = returns.mean() * 252  # Approximate annualization
            annual_volatility = returns.std() * np.sqrt(252)
            sharpe = annual_returns / annual_volatility if annual_volatility > 0 else 0
            
            max_dd = ((curve / curve.expanding().max()) - 1).min()
            
            baseline_metrics.append({
                'strategy': name,
                'total_return': total_return,
                'cagr': cagr,
                'annual_volatility': annual_volatility,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd,
                'start_value': curve.iloc[0],
                'end_value': curve.iloc[-1],
                'num_periods': len(curve)
            })
        
        # Save baseline metrics
        if baseline_metrics:
            baseline_df = pd.DataFrame(baseline_metrics)
            baseline_csv_file = self.results_dir / "baseline_metrics.csv"
            baseline_df.to_csv(baseline_csv_file, index=False)
            
            self.logger.info(f"Baseline metrics saved to {baseline_csv_file}")
            self.logger.info("Baseline performance summary:")
            for _, row in baseline_df.iterrows():
                self.logger.info(f"  {row['strategy']}: CAGR={row['cagr']:.1%}, Sharpe={row['sharpe_ratio']:.2f}")
        else:
            self.logger.warning("No baseline metrics to save")
    
    def _create_feature_importance_plot(self, model_trainer):
        """Create feature importance plots for all models with 3 separate comparison charts."""
        # Check if model trainer has the required attributes
        if not hasattr(model_trainer, 'models') or not model_trainer.models:
            self.logger.warning("Model trainer does not have trained models for importance plot")
            return
            
        # Get feature names
        feature_names = None
        if hasattr(model_trainer, 'feature_names_') and model_trainer.feature_names_ is not None:
            feature_names = model_trainer.feature_names_
        else:
            # Try to get from a trained model
            for model_name, model in model_trainer.models.items():
                if hasattr(model, 'feature_names_in_'):
                    feature_names = model.feature_names_in_
                    break
        
        if feature_names is None:
            self.logger.warning("Feature names not available for importance plot")
            return
        
        # Collect models with feature importance
        model_importances = {}
        for model_name in ['logistic_regression', 'random_forest', 'xgboost']:
            if model_name in model_trainer.models:
                model = model_trainer.models[model_name]
                
                if model_name == 'logistic_regression' and hasattr(model, 'coef_'):
                    # For logistic regression, use absolute coefficients
                    importances = np.abs(model.coef_[0])
                    model_importances[model_name] = importances
                elif model_name == 'xgboost':
                    # For XGBoost, handle different object types
                    if hasattr(model, 'get_score'):
                        # XGBoost Booster object
                        importance_dict = model.get_score(importance_type='weight')
                        # Convert to array matching feature order
                        importances = np.zeros(len(feature_names))
                        for i, feature_name in enumerate(feature_names):
                            importances[i] = importance_dict.get(feature_name, 0.0)
                        model_importances[model_name] = importances
                    elif hasattr(model, 'feature_importances_'):
                        # XGBClassifier object
                        importances = model.feature_importances_
                        model_importances[model_name] = importances
                elif hasattr(model, 'feature_importances_'):
                    # For tree-based models (Random Forest, etc.)
                    importances = model.feature_importances_
                    model_importances[model_name] = importances
        
        if not model_importances:
            self.logger.warning("No models with feature importance available")
            return
        
        # Create 3 side-by-side comparison plots
        n_models = len(model_importances)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 10))
        if n_models == 1:
            axes = [axes]
        
        fig.suptitle(f'Feature Importance Comparison - {self.run_timestamp}', fontsize=16)
        
        colors = {'logistic_regression': 'steelblue', 'random_forest': 'forestgreen', 'xgboost': 'firebrick'}
        
        for i, (model_name, importances) in enumerate(model_importances.items()):
            # Get top 15 features for this model
            indices = np.argsort(importances)[::-1][:15]
            top_importances = importances[indices]
            top_features = [feature_names[idx] for idx in indices]
            
            # Create horizontal bar plot
            y_pos = np.arange(len(top_features))
            color = colors.get(model_name, 'gray')
            bars = axes[i].barh(y_pos, top_importances, color=color, alpha=0.8)
            
            # Customize plot
            model_display_name = model_name.replace('_', ' ').title()
            axes[i].set_title(f'{model_display_name}', fontsize=14, fontweight='bold')
            axes[i].set_xlabel('Feature Importance', fontsize=12)
            axes[i].set_yticks(y_pos)
            axes[i].set_yticklabels(top_features, fontsize=10)
            axes[i].invert_yaxis()  # Highest importance at top
            axes[i].grid(axis='x', alpha=0.3)
            
            # Add value labels on bars
            for j, (bar, value) in enumerate(zip(bars, top_importances)):
                axes[i].text(value + max(top_importances) * 0.01, j, f'{value:.3f}', 
                           va='center', ha='left', fontsize=9)
        
        plt.tight_layout()
        
        # Save plot
        importance_plot_file = self.results_dir / "feature_importance.png"
        plt.savefig(importance_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save feature importance data to CSV
        self._save_feature_importance_csv(model_importances, feature_names)
        
        self.logger.info(f"Feature importance comparison plot saved to {importance_plot_file}")
        
        # Log feature overlap analysis
        if len(model_importances) >= 2:
            self._analyze_feature_overlap(model_importances, feature_names)
    
    def _analyze_feature_overlap(self, model_importances: dict, feature_names):
        """Analyze overlap between top features across models."""
        top_n = 10
        top_features_by_model = {}
        
        # Get top N features for each model
        for model_name, importances in model_importances.items():
            indices = np.argsort(importances)[::-1][:top_n]
            top_features = [feature_names[idx] for idx in indices]
            top_features_by_model[model_name] = set(top_features)
        
        # Calculate pairwise overlaps
        models = list(top_features_by_model.keys())
        self.logger.info(f"Feature overlap analysis (top {top_n} features):")
        
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                model1, model2 = models[i], models[j]
                overlap = top_features_by_model[model1] & top_features_by_model[model2]
                overlap_rate = len(overlap) / top_n
                
                self.logger.info(f"  {model1} ∩ {model2}: {len(overlap)}/{top_n} = {overlap_rate:.1%}")
                if overlap:
                    self.logger.info(f"    Common features: {sorted(overlap)}")
        
        # Find features in top N of all models
        if len(models) >= 2:
            common_features = top_features_by_model[models[0]]
            for model in models[1:]:
                common_features &= top_features_by_model[model]
            
            self.logger.info(f"  Features in top {top_n} of ALL models: {sorted(common_features)}")
            self.logger.info(f"  Universal agreement rate: {len(common_features)}/{top_n} = {len(common_features)/top_n:.1%}")
    
    
    def export_detailed_logs(self, log_content: str):
        """
        Export detailed execution logs to timestamped file.
        
        Args:
            log_content: String content of logs to save
        """
        log_file = self.results_dir / f"execution_log_{self.run_timestamp}.txt"
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"Execution Log - {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            f.write(log_content)
        
        self.logger.info(f"Execution logs saved to {log_file}")
    
    def _save_feature_importance_csv(self, model_importances: Dict[str, np.ndarray], feature_names: List[str]):
        """
        Save feature importance data to CSV file.
        
        Args:
            model_importances: Dictionary mapping model names to importance arrays
            feature_names: List of feature names
        """
        import pandas as pd
        
        # Create DataFrame with feature importance data
        importance_data = {'feature': feature_names}
        for model_name, importances in model_importances.items():
            importance_data[f'{model_name}_importance'] = importances
            
            # Add ranking for each model
            ranking = np.argsort(-importances) + 1  # +1 to make it 1-based ranking
            importance_data[f'{model_name}_rank'] = ranking
        
        df = pd.DataFrame(importance_data)
        
        # Sort by mean importance across all models
        importance_cols = [col for col in df.columns if col.endswith('_importance')]
        df['mean_importance'] = df[importance_cols].mean(axis=1)
        df = df.sort_values('mean_importance', ascending=False)
        df = df.drop('mean_importance', axis=1)
        
        # Save to CSV
        csv_file = self.results_dir / f"feature_importance_comparison_{self.run_timestamp}.csv"
        df.to_csv(csv_file, index=False)
        
        self.logger.info(f"Feature importance data saved to {csv_file}")