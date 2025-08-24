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
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            y_pred_binary: Predicted binary labels (computed if None)
            threshold: Threshold for binary predictions
            
        Returns:
            Dictionary with ML metrics
        """
        if y_pred_binary is None:
            y_pred_binary = (y_pred_proba > threshold).astype(int)
        
        metrics = {
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
            'f1': f1_score(y_true, y_pred_binary),
            'precision': precision_score(y_true, y_pred_binary),
            'recall': recall_score(y_true, y_pred_binary),
            'accuracy': accuracy_score(y_true, y_pred_binary),
            'threshold': threshold
        }
        
        # Class distribution
        pos_rate = y_true.mean()
        pred_pos_rate = y_pred_binary.mean()
        
        metrics['true_positive_rate'] = pos_rate
        metrics['predicted_positive_rate'] = pred_pos_rate
        
        # Confusion matrix elements
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        
        metrics.update({
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        })
        
        return metrics
    
    def calculate_financial_metrics(self, 
                                   equity_curve: pd.Series,
                                   trades_df: pd.DataFrame = None,
                                   risk_free_rate: float = 0.02) -> Dict[str, float]:
        """
        Calculate financial performance metrics.
        
        Args:
            equity_curve: Portfolio equity curve
            trades_df: Individual trades dataframe
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Dictionary with financial metrics
        """
        if len(equity_curve) < 2:
            return self._empty_financial_metrics()
        
        # Calculate returns
        returns = equity_curve.pct_change().dropna()
        
        # Time period
        start_date = equity_curve.index[0]
        end_date = equity_curve.index[-1]
        total_days = (end_date - start_date).days
        total_years = total_days / 365.25
        
        # Total and annualized returns
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        cagr = ((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1 / total_years)) - 1 if total_years > 0 else 0
        
        # Volatility
        annual_volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
        
        # Risk-adjusted returns
        excess_return = cagr - risk_free_rate
        sharpe_ratio = excess_return / annual_volatility if annual_volatility > 0 else 0
        
        # Downside metrics
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = excess_return / downside_deviation if downside_deviation > 0 else 0
        
        # Drawdown analysis
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Drawdown duration
        in_drawdown = drawdown < 0
        if in_drawdown.any():
            drawdown_periods = []
            current_period = 0
            for is_dd in in_drawdown:
                if is_dd:
                    current_period += 1
                else:
                    if current_period > 0:
                        drawdown_periods.append(current_period)
                        current_period = 0
            if current_period > 0:
                drawdown_periods.append(current_period)
            
            avg_drawdown_duration = np.mean(drawdown_periods) if drawdown_periods else 0
            max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0
        else:
            avg_drawdown_duration = 0
            max_drawdown_duration = 0
        
        # Calmar ratio (CAGR / |Max Drawdown|)
        calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trade-based metrics (if trades provided)
        trade_metrics = {}
        if trades_df is not None and len(trades_df) > 0:
            winning_trades = (trades_df['trade_return'] > 0).sum()
            total_trades = len(trades_df)
            
            trade_metrics.update({
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'hit_rate': winning_trades / total_trades,
                'avg_trade_return': trades_df['trade_return'].mean(),
                'avg_winning_trade': trades_df[trades_df['trade_return'] > 0]['trade_return'].mean() if winning_trades > 0 else 0,
                'avg_losing_trade': trades_df[trades_df['trade_return'] <= 0]['trade_return'].mean() if (total_trades - winning_trades) > 0 else 0,
                'trades_per_year': total_trades / total_years if total_years > 0 else 0,
                
                # Profit factor
                'profit_factor': (
                    trades_df[trades_df['trade_pnl_gross'] > 0]['trade_pnl_gross'].sum() /
                    abs(trades_df[trades_df['trade_pnl_gross'] <= 0]['trade_pnl_gross'].sum())
                    if trades_df[trades_df['trade_pnl_gross'] <= 0]['trade_pnl_gross'].sum() != 0 else np.inf
                )
            })
        
        metrics = {
            # Returns
            'total_return': total_return,
            'cagr': cagr,
            'annual_volatility': annual_volatility,
            
            # Risk-adjusted
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            
            # Drawdown
            'max_drawdown': max_drawdown,
            'avg_drawdown_duration': avg_drawdown_duration,
            'max_drawdown_duration': max_drawdown_duration,
            
            # Time
            'total_days': total_days,
            'total_years': total_years,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d')
        }
        
        metrics.update(trade_metrics)
        return metrics
    
    def _empty_financial_metrics(self) -> Dict[str, float]:
        """Return empty financial metrics for edge cases."""
        return {
            'total_return': 0, 'cagr': 0, 'annual_volatility': 0,
            'sharpe_ratio': 0, 'sortino_ratio': 0, 'calmar_ratio': 0,
            'max_drawdown': 0, 'avg_drawdown_duration': 0, 'max_drawdown_duration': 0,
            'total_days': 0, 'total_years': 0, 'start_date': '', 'end_date': ''
        }
    
    def evaluate_model_performance(self, 
                                  y_true: np.ndarray,
                                  y_pred_proba: np.ndarray,
                                  equity_curve: pd.Series,
                                  trades_df: pd.DataFrame = None,
                                  model_name: str = "model",
                                  threshold: float = 0.5) -> Dict[str, Any]:
        """
        Comprehensive model evaluation combining ML and financial metrics.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            equity_curve: Portfolio equity curve
            trades_df: Individual trades dataframe
            model_name: Name of the model
            threshold: Probability threshold used
            
        Returns:
            Dictionary with combined evaluation results
        """
        self.logger.info(f"Evaluating performance for {model_name}")
        
        # ML metrics
        ml_metrics = self.calculate_ml_metrics(y_true, y_pred_proba, threshold=threshold)
        
        # Financial metrics
        financial_metrics = self.calculate_financial_metrics(equity_curve, trades_df)
        
        # Combined evaluation
        evaluation = {
            'model_name': model_name,
            'evaluation_date': datetime.now().isoformat(),
            'ml_metrics': ml_metrics,
            'financial_metrics': financial_metrics,
            
            # Summary scores
            'overall_score': self._calculate_overall_score(ml_metrics, financial_metrics),
            'ml_score': self._calculate_ml_score(ml_metrics),
            'financial_score': self._calculate_financial_score(financial_metrics)
        }
        
        return evaluation
    
    def _calculate_overall_score(self, 
                                ml_metrics: Dict[str, float],
                                financial_metrics: Dict[str, float]) -> float:
        """Calculate weighted overall performance score."""
        # Weighted combination of ML and financial performance
        ml_score = self._calculate_ml_score(ml_metrics)
        financial_score = self._calculate_financial_score(financial_metrics)
        
        # 40% ML performance, 60% financial performance
        overall_score = 0.4 * ml_score + 0.6 * financial_score
        return overall_score
    
    def _calculate_ml_score(self, ml_metrics: Dict[str, float]) -> float:
        """Calculate ML performance score (0-100)."""
        # Weighted combination of key ML metrics
        auc_score = ml_metrics.get('roc_auc', 0.5) * 100  # 50-100 range
        f1_score_val = ml_metrics.get('f1', 0) * 100       # 0-100 range
        
        # Weight AUC more heavily as it's threshold-independent
        ml_score = 0.7 * auc_score + 0.3 * f1_score_val
        return max(0, min(100, ml_score))
    
    def _calculate_financial_score(self, financial_metrics: Dict[str, float]) -> float:
        """Calculate financial performance score (0-100)."""
        # Normalize key financial metrics
        sharpe = financial_metrics.get('sharpe_ratio', 0)
        cagr = financial_metrics.get('cagr', 0)
        max_dd = abs(financial_metrics.get('max_drawdown', 0))
        
        # Score components (normalized to 0-100)
        sharpe_score = min(100, max(0, (sharpe + 1) * 25))  # -1 to 3 Sharpe -> 0 to 100
        return_score = min(100, max(0, cagr * 500))          # 0% to 20% CAGR -> 0 to 100
        drawdown_score = max(0, 100 - max_dd * 200)         # 0% to 50% DD -> 100 to 0
        
        # Weighted combination
        financial_score = 0.4 * sharpe_score + 0.3 * return_score + 0.3 * drawdown_score
        return max(0, min(100, financial_score))
    
    def compare_models(self, 
                      evaluations: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Compare multiple model evaluations.
        
        Args:
            evaluations: List of evaluation dictionaries
            
        Returns:
            DataFrame with model comparison
        """
        self.logger.info(f"Comparing {len(evaluations)} models")
        
        comparison_data = []
        
        for eval_result in evaluations:
            model_name = eval_result['model_name']
            ml_metrics = eval_result['ml_metrics']
            fin_metrics = eval_result['financial_metrics']
            
            row = {
                'model': model_name,
                'overall_score': eval_result['overall_score'],
                'ml_score': eval_result['ml_score'],
                'financial_score': eval_result['financial_score'],
                
                # Key ML metrics
                'roc_auc': ml_metrics['roc_auc'],
                'f1': ml_metrics['f1'],
                'precision': ml_metrics['precision'],
                'recall': ml_metrics['recall'],
                
                # Key financial metrics
                'cagr': fin_metrics['cagr'],
                'sharpe_ratio': fin_metrics['sharpe_ratio'],
                'max_drawdown': fin_metrics['max_drawdown'],
                'calmar_ratio': fin_metrics['calmar_ratio'],
                
                # Trade metrics (if available)
                'total_trades': fin_metrics.get('total_trades', 0),
                'hit_rate': fin_metrics.get('hit_rate', 0),
                'profit_factor': fin_metrics.get('profit_factor', 0)
            }
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by overall score
        comparison_df = comparison_df.sort_values('overall_score', ascending=False)
        
        # Add ranking
        comparison_df['rank'] = range(1, len(comparison_df) + 1)
        
        return comparison_df
    
    def analyze_by_region(self, 
                         predictions_df: pd.DataFrame,
                         model_name: str = 'xgboost') -> pd.DataFrame:
        """
        Performance analysis by market region.
        
        Args:
            predictions_df: DataFrame with predictions and symbols
            model_name: Model name for analysis
            
        Returns:
            DataFrame with regional performance breakdown
        """
        self.logger.info(f"Analyzing {model_name} performance by region")
        
        # Add market classification
        predictions_df = predictions_df.copy()
        predictions_df['market'] = predictions_df['symbol'].map(
            lambda x: TICKER_MARKETS.get(x, 'UNKNOWN')
        )
        
        proba_col = f'y_proba_{model_name}'
        
        if proba_col not in predictions_df.columns:
            raise ValueError(f"Column {proba_col} not found")
        
        regional_analysis = []
        
        for market in predictions_df['market'].unique():
            market_data = predictions_df[predictions_df['market'] == market]
            
            if len(market_data) == 0:
                continue
            
            # ML metrics
            y_true = market_data['y_true']
            y_pred_proba = market_data[proba_col]
            y_pred_binary = (y_pred_proba > 0.6).astype(int)
            
            ml_metrics = self.calculate_ml_metrics(y_true, y_pred_proba, y_pred_binary)
            
            # Basic statistics
            analysis = {
                'market': market,
                'num_symbols': market_data['symbol'].nunique(),
                'num_observations': len(market_data),
                'avg_return': market_data['ret_30d'].mean(),
                'return_volatility': market_data['ret_30d'].std(),
                'positive_rate': y_true.mean(),
                
                # ML performance
                'roc_auc': ml_metrics['roc_auc'],
                'f1': ml_metrics['f1'],
                'precision': ml_metrics['precision'],
                'recall': ml_metrics['recall'],
                
                # Signal statistics
                'signal_rate': y_pred_binary.mean(),
                'avg_predicted_prob': y_pred_proba.mean()
            }
            
            regional_analysis.append(analysis)
        
        regional_df = pd.DataFrame(regional_analysis)
        regional_df = regional_df.sort_values('roc_auc', ascending=False)
        
        return regional_df
    
    def create_performance_plots(self, 
                                evaluation_results: Dict[str, Any],
                                save_prefix: str = "performance") -> Dict[str, plt.Figure]:
        """
        Create comprehensive performance visualization plots.
        
        Args:
            evaluation_results: Dictionary with evaluation results
            save_prefix: Prefix for saved plot files
            
        Returns:
            Dictionary with matplotlib figures
        """
        self.logger.info("Creating performance plots")
        
        figures = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. ROC Curve
        if 'roc_data' in evaluation_results:
            fig_roc = self._plot_roc_curve(evaluation_results['roc_data'])
            figures['roc_curve'] = fig_roc
            
            save_path = self.output_dir / f"{save_prefix}_roc_{timestamp}.png"
            fig_roc.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # 2. Precision-Recall Curve
        if 'pr_data' in evaluation_results:
            fig_pr = self._plot_precision_recall_curve(evaluation_results['pr_data'])
            figures['precision_recall'] = fig_pr
            
            save_path = self.output_dir / f"{save_prefix}_pr_{timestamp}.png"
            fig_pr.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # 3. Model Comparison
        if 'comparison_df' in evaluation_results:
            fig_comp = self._plot_model_comparison(evaluation_results['comparison_df'])
            figures['model_comparison'] = fig_comp
            
            save_path = self.output_dir / f"{save_prefix}_comparison_{timestamp}.png"
            fig_comp.savefig(save_path, dpi=300, bbox_inches='tight')
        
        # 4. Regional Analysis
        if 'regional_df' in evaluation_results:
            fig_regional = self._plot_regional_analysis(evaluation_results['regional_df'])
            figures['regional_analysis'] = fig_regional
            
            save_path = self.output_dir / f"{save_prefix}_regional_{timestamp}.png"
            fig_regional.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return figures
    
    def _plot_roc_curve(self, roc_data: Dict[str, Any]) -> plt.Figure:
        """Plot ROC curves for multiple models."""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        for model_name, (fpr, tpr, auc_score) in roc_data.items():
            ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})', linewidth=2)
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves - Model Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def _plot_precision_recall_curve(self, pr_data: Dict[str, Any]) -> plt.Figure:
        """Plot Precision-Recall curves for multiple models."""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        for model_name, (precision, recall, ap_score) in pr_data.items():
            ax.plot(recall, precision, label=f'{model_name} (AP = {ap_score:.3f})', linewidth=2)
        
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves - Model Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def _plot_model_comparison(self, comparison_df: pd.DataFrame) -> plt.Figure:
        """Plot model comparison metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        metrics = ['roc_auc', 'f1', 'sharpe_ratio', 'cagr']
        titles = ['ROC AUC', 'F1 Score', 'Sharpe Ratio', 'CAGR']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            if metric in comparison_df.columns:
                bars = ax.bar(comparison_df['model'], comparison_df[metric])
                ax.set_title(title)
                ax.set_ylabel(metric.replace('_', ' ').title())
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    if not np.isnan(height):
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def _plot_regional_analysis(self, regional_df: pd.DataFrame) -> plt.Figure:
        """Plot regional performance analysis."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Performance by Market Region', fontsize=16, fontweight='bold')
        
        # ROC AUC by region
        ax1 = axes[0, 0]
        bars1 = ax1.bar(regional_df['market'], regional_df['roc_auc'])
        ax1.set_title('ROC AUC by Region')
        ax1.set_ylabel('ROC AUC')
        plt.setp(ax1.get_xticklabels(), rotation=45)
        
        # F1 Score by region
        ax2 = axes[0, 1]
        bars2 = ax2.bar(regional_df['market'], regional_df['f1'])
        ax2.set_title('F1 Score by Region')
        ax2.set_ylabel('F1 Score')
        plt.setp(ax2.get_xticklabels(), rotation=45)
        
        # Average return by region
        ax3 = axes[1, 0]
        bars3 = ax3.bar(regional_df['market'], regional_df['avg_return'] * 100)
        ax3.set_title('Average 30D Return by Region')
        ax3.set_ylabel('Return (%)')
        plt.setp(ax3.get_xticklabels(), rotation=45)
        
        # Number of observations by region
        ax4 = axes[1, 1]
        bars4 = ax4.bar(regional_df['market'], regional_df['num_observations'])
        ax4.set_title('Number of Observations by Region')
        ax4.set_ylabel('Count')
        plt.setp(ax4.get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        return fig
    
    def export_evaluation_report(self, 
                                evaluation_results: Dict[str, Any],
                                filename: str = None) -> str:
        """
        Export comprehensive evaluation report.
        
        Args:
            evaluation_results: Dictionary with all evaluation results
            filename: Output filename (auto-generated if None)
            
        Returns:
            Path to exported report file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_report_{timestamp}.json"
        
        report_path = self.output_dir / filename
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        # Prepare report data
        report_data = convert_for_json(evaluation_results)
        
        # Add metadata
        report_data['report_metadata'] = {
            'generation_date': datetime.now().isoformat(),
            'report_version': '1.0',
            'evaluator_type': 'PerformanceEvaluator'
        }
        
        # Save report
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        self.logger.info(f"Exported evaluation report to {report_path}")
        return str(report_path)

def create_roc_pr_data(predictions_df: pd.DataFrame, 
                      model_names: List[str] = None) -> Tuple[Dict, Dict]:
    """
    Create ROC and Precision-Recall curve data for plotting.
    
    Args:
        predictions_df: DataFrame with predictions
        model_names: List of model names (auto-detect if None)
        
    Returns:
        Tuple of (roc_data, pr_data) dictionaries
    """
    if model_names is None:
        model_names = [col.replace('y_proba_', '') for col in predictions_df.columns 
                      if col.startswith('y_proba_')]
    
    roc_data = {}
    pr_data = {}
    
    y_true = predictions_df['y_true']
    
    for model_name in model_names:
        proba_col = f'y_proba_{model_name}'
        
        if proba_col in predictions_df.columns:
            y_proba = predictions_df[proba_col]
            
            # ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            auc_score = roc_auc_score(y_true, y_proba)
            roc_data[model_name] = (fpr, tpr, auc_score)
            
            # Precision-Recall curve
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            # Average precision
            ap_score = np.trapz(precision, recall)
            pr_data[model_name] = (precision, recall, ap_score)
    
    return roc_data, pr_data