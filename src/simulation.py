"""
Trading Simulation module for Cross-Market 30D Directional Forecasting project.

Implements vectorized trading simulation with:
- Threshold-based entry signals
- Fixed stake per trade
- Transaction costs
- Portfolio equity tracking
- Financial metrics calculation
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# Set matplotlib backend to avoid tkinter threading issues
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    INVESTMENT_PER_SIGNAL, TRANSACTION_COST, THRESHOLD_SWEEP,
    PREDICTION_HORIZON_DAYS, REPORTS_DIR
)

class TradingSimulator:
    """
    Vectorized trading simulation for directional forecasting models.
    
    Features:
    - Signal generation based on prediction probabilities
    - Portfolio equity tracking
    - Transaction cost modeling
    - Performance metrics calculation
    - Threshold sensitivity analysis
    """
    
    def __init__(self, 
                 investment_per_signal: float = INVESTMENT_PER_SIGNAL,
                 transaction_cost: float = TRANSACTION_COST,
                 initial_capital: float = 10000):
        """
        Initialize TradingSimulator.
        
        Args:
            investment_per_signal: Fixed amount invested per signal
            transaction_cost: Transaction cost as fraction (e.g., 0.001 = 0.1%)
            initial_capital: Starting portfolio value
        """
        self.investment_per_signal = investment_per_signal
        self.transaction_cost = transaction_cost
        self.initial_capital = initial_capital
        
        self.logger = logging.getLogger(__name__)
        
    def generate_signals(self, 
                        predictions_df: pd.DataFrame,
                        probability_column: str,
                        threshold: float = 0.6) -> pd.DataFrame:
        """
        Generate trading signals based on prediction probabilities.
        
        Args:
            predictions_df: DataFrame with predictions
            probability_column: Name of probability column
            threshold: Probability threshold for signal generation
            
        Returns:
            DataFrame with signals added
        """
        signals_df = predictions_df.copy()
        
        # Generate binary signals
        signals_df['signal'] = (signals_df[probability_column] >= threshold).astype(int)
        signals_df['threshold'] = threshold
        
        return signals_df
    
    def simulate_trades(self, 
                       signals_df: pd.DataFrame,
                       return_column: str = 'growth_future_30d') -> pd.DataFrame:
        """
        Simulate individual trades based on signals using original notebook approach.
        
        Args:
            signals_df: DataFrame with signals
            return_column: Name of growth ratio column (growth_future_30d from original)
            
        Returns:
            DataFrame with trade results
        """
        # Filter to trades only (signal = 1)
        trades_df = signals_df[signals_df['signal'] == 1].copy()
        
        if len(trades_df) == 0:
            self.logger.warning("No trades generated")
            return pd.DataFrame()
        
        # Calculate trade PnL using original notebook approach
        # Gross revenue = investment * (growth_ratio - 1) where growth_ratio = price[t+30]/price[t]
        trades_df['growth_ratio'] = trades_df[return_column]
        trades_df['trade_pnl_gross'] = self.investment_per_signal * (trades_df['growth_ratio'] - 1)
        
        # Apply transaction costs: 0.2% total (0.1% buy + 0.1% sell) 
        # Following original notebook: -investment * 0.002
        trades_df['transaction_costs'] = -self.investment_per_signal * (self.transaction_cost * 2)
        trades_df['trade_pnl_net'] = trades_df['trade_pnl_gross'] + trades_df['transaction_costs']
        
        # Trade metadata
        trades_df['investment'] = self.investment_per_signal
        trades_df['trade_success'] = (trades_df['growth_ratio'] > 1.0).astype(int)  # growth > 1 means positive return
        
        return trades_df
    
    def calculate_equity_curve(self, trades_df: pd.DataFrame) -> pd.Series:
        """
        Calculate cumulative equity curve from trades following original notebook approach.
        
        Args:
            trades_df: DataFrame with trade results
            
        Returns:
            Series with cumulative equity
        """
        if len(trades_df) == 0:
            return pd.Series([self.initial_capital], 
                           index=[trades_df.index.min() if not trades_df.empty else pd.Timestamp.now()])
        
        # Aggregate trades by date to get daily PnL (following original notebook)
        # Multiple trades can happen on the same day across different tickers
        # Date is in the index, so group by index
        daily_pnl = trades_df.groupby(trades_df.index)['trade_pnl_net'].sum().sort_index()
        
        # Calculate cumulative equity starting from initial capital
        cumulative_pnl = daily_pnl.cumsum()
        equity_curve = self.initial_capital + cumulative_pnl
        
        return equity_curve
    
    def calculate_performance_metrics(self, 
                                    equity_curve: pd.Series,
                                    trades_df: pd.DataFrame,
                                    risk_free_rate: float = 0.02) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics following original notebook approach.
        
        Args:
            equity_curve: Series with cumulative equity
            trades_df: DataFrame with trade results
            risk_free_rate: Annual risk-free rate
            
        Returns:
            Dictionary with performance metrics
        """
        if len(equity_curve) < 2 or len(trades_df) == 0:
            return self._empty_metrics()
        
        # Basic trade statistics - using original notebook approach
        total_trades = len(trades_df)
        winning_trades = (trades_df['trade_success'] == 1).sum()  # Using trade_success from original
        losing_trades = total_trades - winning_trades
        
        # Returns and PnL - using net PnL from trades
        total_pnl = trades_df['trade_pnl_net'].sum()  # Sum of all net trade PnL
        total_return = total_pnl / self.initial_capital
        
        # Time period
        start_date = equity_curve.index[0]
        end_date = equity_curve.index[-1]
        days_total = (end_date - start_date).days
        years_total = days_total / 365.25
        
        # Annualized metrics - final equity vs initial capital
        final_equity = equity_curve.iloc[-1]
        cagr = ((final_equity / self.initial_capital) ** (1 / years_total)) - 1 if years_total > 0 else 0
        
        # Daily returns for volatility/Sharpe calculation
        daily_returns = equity_curve.pct_change().dropna()
        
        if len(daily_returns) > 1:
            annual_vol = daily_returns.std() * np.sqrt(252)
            sharpe_ratio = (cagr - risk_free_rate) / annual_vol if annual_vol > 0 else 0
            
            # Sortino ratio (downside deviation)
            downside_returns = daily_returns[daily_returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = (cagr - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        else:
            annual_vol = 0
            sharpe_ratio = 0
            sortino_ratio = 0
        
        # Maximum drawdown
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Trade-level metrics using new trade structure
        avg_trade_pnl = trades_df['trade_pnl_net'].mean()
        avg_winning_trade = trades_df[trades_df['trade_success'] == 1]['trade_pnl_net'].mean() if winning_trades > 0 else 0
        avg_losing_trade = trades_df[trades_df['trade_success'] == 0]['trade_pnl_net'].mean() if losing_trades > 0 else 0
        
        hit_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Profit factor - using gross PnL
        gross_profit = trades_df[trades_df['trade_pnl_gross'] > 0]['trade_pnl_gross'].sum()
        gross_loss = -trades_df[trades_df['trade_pnl_gross'] <= 0]['trade_pnl_gross'].sum()
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        
        # Trading frequency
        trades_per_year = total_trades / years_total if years_total > 0 else 0
        
        metrics = {
            # Returns
            'total_return': total_return,
            'cagr': cagr,
            'total_pnl': total_pnl,
            
            # Risk
            'annual_volatility': annual_vol,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            
            # Trade statistics
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'hit_rate': hit_rate,
            'avg_trade_return': avg_trade_pnl / self.investment_per_signal if self.investment_per_signal > 0 else 0,  # Convert to return
            'avg_winning_trade': avg_winning_trade,
            'avg_losing_trade': avg_losing_trade,
            'profit_factor': profit_factor,
            
            # Other
            'trades_per_year': trades_per_year,
            'days_total': days_total,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d')
        }
        
        return metrics
    
    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics dictionary for edge cases."""
        return {
            'total_return': 0, 'cagr': 0, 'total_pnl': 0,
            'annual_volatility': 0, 'max_drawdown': 0,
            'sharpe_ratio': 0, 'sortino_ratio': 0,
            'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
            'hit_rate': 0, 'avg_trade_return': 0,
            'avg_winning_trade': 0, 'avg_losing_trade': 0,
            'profit_factor': 0, 'trades_per_year': 0,
            'days_total': 0, 'start_date': '', 'end_date': ''
        }
    
    def run_single_simulation(self, 
                             predictions_df: pd.DataFrame,
                             model_name: str,
                             threshold: float = 0.6,
                             return_column: str = 'growth_future_30d') -> Dict[str, any]:
        """
        Run simulation for a single model and threshold.
        
        Args:
            predictions_df: DataFrame with predictions
            model_name: Name of the model
            threshold: Probability threshold
            return_column: Name of growth ratio column (growth_future_30d from original)
            
        Returns:
            Dictionary with simulation results
        """
        probability_column = f'y_proba_{model_name}'
        
        if probability_column not in predictions_df.columns:
            raise ValueError(f"Column {probability_column} not found in predictions")
        
        # Generate signals
        signals_df = self.generate_signals(predictions_df, probability_column, threshold)
        
        # Simulate trades with original notebook approach
        trades_df = self.simulate_trades(signals_df, return_column)
        
        # Calculate equity curve
        equity_curve = self.calculate_equity_curve(trades_df)
        
        # Calculate metrics
        metrics = self.calculate_performance_metrics(equity_curve, trades_df)
        
        # Add model and threshold info
        metrics['model_name'] = model_name
        metrics['threshold'] = threshold
        
        return {
            'metrics': metrics,
            'trades': trades_df,
            'equity_curve': equity_curve,
            'signals': signals_df
        }
    
    def run_threshold_sweep(self, 
                           predictions_df: pd.DataFrame,
                           model_name: str,
                           thresholds: List[float] = None) -> pd.DataFrame:
        """
        Run simulation across multiple thresholds.
        
        Args:
            predictions_df: DataFrame with predictions
            model_name: Name of the model
            thresholds: List of thresholds to test
            
        Returns:
            DataFrame with metrics for each threshold
        """
        if thresholds is None:
            thresholds = THRESHOLD_SWEEP
        
        self.logger.info(f"Running threshold sweep for {model_name} with {len(thresholds)} thresholds")
        
        results = []
        
        for threshold in thresholds:
            try:
                result = self.run_single_simulation(predictions_df, model_name, threshold)
                results.append(result['metrics'])
            except Exception as e:
                self.logger.warning(f"Simulation failed for threshold {threshold}: {e}")
                # Add empty result
                empty_metrics = self._empty_metrics()
                empty_metrics['model_name'] = model_name
                empty_metrics['threshold'] = threshold
                results.append(empty_metrics)
        
        return pd.DataFrame(results)
    
    def run_model_comparison(self, 
                            predictions_df: pd.DataFrame,
                            model_names: List[str] = None,
                            threshold: float = 0.6) -> pd.DataFrame:
        """
        Compare multiple models at a fixed threshold.
        
        Args:
            predictions_df: DataFrame with predictions
            model_names: List of model names (auto-detect if None)
            threshold: Probability threshold
            
        Returns:
            DataFrame with metrics for each model
        """
        if model_names is None:
            # Auto-detect model names from columns
            model_names = [col.replace('y_proba_', '') for col in predictions_df.columns 
                          if col.startswith('y_proba_')]
        
        self.logger.info(f"Comparing {len(model_names)} models at threshold {threshold}")
        
        results = []
        
        for model_name in model_names:
            try:
                result = self.run_single_simulation(predictions_df, model_name, threshold)
                results.append(result['metrics'])
            except Exception as e:
                self.logger.warning(f"Simulation failed for model {model_name}: {e}")
                # Add empty result
                empty_metrics = self._empty_metrics()
                empty_metrics['model_name'] = model_name
                empty_metrics['threshold'] = threshold
                results.append(empty_metrics)
        
        return pd.DataFrame(results)
    
    def create_benchmark_comparison(self, 
                                   equity_curve: pd.Series,
                                   benchmark_symbol: str = 'SPY',
                                   benchmark_data: pd.DataFrame = None) -> Dict[str, pd.Series]:
        """
        Compare strategy equity curve with benchmark.
        
        Args:
            equity_curve: Strategy equity curve
            benchmark_symbol: Benchmark symbol (e.g., 'SPY')
            benchmark_data: Optional benchmark price data
            
        Returns:
            Dictionary with strategy and benchmark curves
        """
        if benchmark_data is None:
            # If no benchmark data provided, create dummy benchmark
            self.logger.warning("No benchmark data provided, creating synthetic benchmark")
            # Simple 7% annual return benchmark
            start_value = self.initial_capital
            end_value = start_value * (1.07 ** ((equity_curve.index[-1] - equity_curve.index[0]).days / 365.25))
            benchmark_curve = pd.Series(
                np.linspace(start_value, end_value, len(equity_curve)),
                index=equity_curve.index
            )
        else:
            # Calculate benchmark returns aligned with strategy dates
            benchmark_aligned = benchmark_data.reindex(equity_curve.index, method='ffill')
            benchmark_returns = benchmark_aligned['Close'].pct_change().fillna(0)
            benchmark_curve = self.initial_capital * (1 + benchmark_returns).cumprod()
        
        return {
            'strategy': equity_curve,
            'benchmark': benchmark_curve
        }

class SimulationReporter:
    """
    Generate reports and visualizations for simulation results.
    """
    
    def __init__(self, output_dir: str = None):
        """Initialize SimulationReporter."""
        self.output_dir = Path(output_dir) if output_dir else REPORTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
    
    def create_metrics_summary(self, 
                              results_df: pd.DataFrame,
                              title: str = "Simulation Results") -> pd.DataFrame:
        """
        Create formatted summary of metrics.
        
        Args:
            results_df: DataFrame with simulation results
            title: Title for the summary
            
        Returns:
            Formatted DataFrame
        """
        # Select key metrics for display
        key_metrics = [
            'model_name', 'threshold', 'cagr', 'sharpe_ratio', 'max_drawdown',
            'total_trades', 'hit_rate', 'profit_factor'
        ]
        
        available_metrics = [col for col in key_metrics if col in results_df.columns]
        summary_df = results_df[available_metrics].copy()
        
        # Format percentages
        percentage_cols = ['cagr', 'max_drawdown', 'hit_rate']
        for col in percentage_cols:
            if col in summary_df.columns:
                summary_df[col] = summary_df[col] * 100
        
        # Round numeric columns
        numeric_cols = summary_df.select_dtypes(include=[np.number]).columns
        summary_df[numeric_cols] = summary_df[numeric_cols].round(2)
        
        self.logger.info(f"{title}:")
        self.logger.info(f"\n{summary_df.to_string(index=False)}")
        
        return summary_df
    
    def plot_equity_curves(self, 
                          equity_curves: Dict[str, pd.Series],
                          title: str = "Equity Curves",
                          save_path: str = None) -> plt.Figure:
        """
        Plot multiple equity curves.
        
        Args:
            equity_curves: Dictionary of {name: equity_series}
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for name, curve in equity_curves.items():
            ax.plot(curve.index, curve.values, label=name, linewidth=2)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved equity curve plot to {save_path}")
        
        return fig
    
    def plot_threshold_analysis(self, 
                               results_df: pd.DataFrame,
                               metric: str = 'sharpe_ratio',
                               title: str = None,
                               save_path: str = None) -> plt.Figure:
        """
        Plot metric vs threshold for threshold analysis.
        
        Args:
            results_df: DataFrame with threshold sweep results
            metric: Metric to plot
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        if title is None:
            title = f"{metric.replace('_', ' ').title()} vs Threshold"
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Group by model if multiple models
        if 'model_name' in results_df.columns:
            for model_name in results_df['model_name'].unique():
                model_data = results_df[results_df['model_name'] == model_name]
                ax.plot(model_data['threshold'], model_data[metric], 
                       marker='o', label=model_name, linewidth=2)
        else:
            ax.plot(results_df['threshold'], results_df[metric], 
                   marker='o', linewidth=2)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Probability Threshold', fontsize=12)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
        
        if 'model_name' in results_df.columns:
            ax.legend(fontsize=10)
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved threshold analysis plot to {save_path}")
        
        return fig
    
    def export_results(self, 
                      results: Dict[str, any],
                      filename_prefix: str = "simulation_results") -> Dict[str, str]:
        """
        Export simulation results to files.
        
        Args:
            results: Dictionary with simulation results
            filename_prefix: Prefix for output filenames
            
        Returns:
            Dictionary with exported file paths
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exported_files = {}
        
        # Export metrics DataFrame
        if 'metrics_df' in results:
            metrics_file = self.output_dir / f"{filename_prefix}_metrics_{timestamp}.csv"
            results['metrics_df'].to_csv(metrics_file, index=False)
            exported_files['metrics'] = str(metrics_file)
            self.logger.info(f"Exported metrics to {metrics_file}")
        
        # Export trades DataFrame
        if 'trades_df' in results:
            trades_file = self.output_dir / f"{filename_prefix}_trades_{timestamp}.csv"
            results['trades_df'].to_csv(trades_file, index=True)
            exported_files['trades'] = str(trades_file)
            self.logger.info(f"Exported trades to {trades_file}")
        
        # Export equity curve
        if 'equity_curves' in results:
            equity_file = self.output_dir / f"{filename_prefix}_equity_{timestamp}.csv"
            equity_df = pd.DataFrame(results['equity_curves'])
            equity_df.to_csv(equity_file, index=True)
            exported_files['equity'] = str(equity_file)
            self.logger.info(f"Exported equity curves to {equity_file}")
        
        return exported_files

def run_complete_simulation(predictions_df: pd.DataFrame,
                           model_names: List[str] = None,
                           output_dir: str = None) -> Dict[str, any]:
    """
    Run complete simulation analysis with all models and thresholds.
    
    Args:
        predictions_df: DataFrame with model predictions
        model_names: List of model names (auto-detect if None)
        output_dir: Output directory for results
        
    Returns:
        Dictionary with complete simulation results
    """
    logger = logging.getLogger(__name__)
    logger.info("Running complete simulation analysis...")
    
    # Initialize simulator and reporter
    simulator = TradingSimulator()
    reporter = SimulationReporter(output_dir)
    
    # Auto-detect model names if not provided
    if model_names is None:
        model_names = [col.replace('y_proba_', '') for col in predictions_df.columns 
                      if col.startswith('y_proba_')]
    
    logger.info(f"Running simulation for models: {model_names}")
    
    results = {
        'threshold_analysis': {},
        'model_comparison': None,
        'best_configs': {},
        'equity_curves': {},
        'summary_metrics': {}
    }
    
    # Run threshold analysis for each model
    for model_name in model_names:
        logger.info(f"Running threshold sweep for {model_name}")
        threshold_results = simulator.run_threshold_sweep(predictions_df, model_name)
        results['threshold_analysis'][model_name] = threshold_results
        
        # Find best threshold by Sharpe ratio
        if len(threshold_results) > 0:
            best_idx = threshold_results['sharpe_ratio'].idxmax()
            best_config = threshold_results.iloc[best_idx]
            results['best_configs'][model_name] = best_config
    
    # Model comparison at default threshold
    logger.info("Running model comparison at default threshold")
    model_comparison = simulator.run_model_comparison(predictions_df, model_names, 0.6)
    results['model_comparison'] = model_comparison
    
    # Generate equity curves for best configurations
    for model_name, best_config in results['best_configs'].items():
        sim_result = simulator.run_single_simulation(
            predictions_df, model_name, best_config['threshold']
        )
        results['equity_curves'][f"{model_name}_best"] = sim_result['equity_curve']
    
    # Create summary
    all_metrics = []
    for model_name, threshold_df in results['threshold_analysis'].items():
        all_metrics.append(threshold_df)
    
    if all_metrics:
        results['summary_metrics'] = pd.concat(all_metrics, ignore_index=True)
    
    logger.info("Complete simulation analysis finished")
    return results