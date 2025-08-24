"""
Enhanced Trading Simulation module for Cross-Market 30D Directional Forecasting project.

This version corrects deviations from Module 04 lecture patterns:
- Fixed PnL calculation: investment * (growth_future_30d - 1) 
- Proper naming conventions following Module 04
- Transaction costs: 0.2% (0.1% buy + 0.1% sell)
- Vectorized simulation with threshold analysis
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

try:
    from .config import (
        INVESTMENT_PER_SIGNAL, TRANSACTION_COST, THRESHOLD_SWEEP,
        PREDICTION_HORIZON_DAYS, REPORTS_DIR
    )
except ImportError:
    from config import (
        INVESTMENT_PER_SIGNAL, TRANSACTION_COST, THRESHOLD_SWEEP,
        PREDICTION_HORIZON_DAYS, REPORTS_DIR
    )

class EnhancedTradingSimulator:
    """
    Enhanced vectorized trading simulation following Module 04 patterns exactly.
    
    Key corrections from original:
    - PnL calculation: investment * (growth_future_30d - 1) [Module 04 line 2203]
    - Proper variable naming: growth_future_30d, is_positive_growth_30d_future
    - Transaction costs: 0.2% total (0.1% buy + 0.1% sell) [Module 04 line 2208]
    - Simulation patterns following Module 04 section 3.1
    """
    
    def __init__(self, 
                 investment_per_signal: float = 100,  # Module 04 default: $100
                 transaction_cost: float = 0.002,     # Module 04 default: 0.2% total
                 initial_capital: float = 10000):
        """
        Initialize Enhanced Trading Simulator with Module 04 defaults.
        
        Args:
            investment_per_signal: Fixed amount invested per signal ($100 in Module 04)
            transaction_cost: Total transaction cost (0.2% = 0.1% buy + 0.1% sell)
            initial_capital: Starting portfolio value
        """
        self.investment_per_signal = investment_per_signal
        self.transaction_cost = transaction_cost
        self.initial_capital = initial_capital
        
        self.logger = logging.getLogger(__name__)
        
    def calculate_growth_future_30d(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate growth_future_30d following Module 04 patterns.
        
        This is the key variable from Module 04 simulation:
        growth_future_30d = future_price / current_price
        
        Args:
            predictions_df: DataFrame with predictions and prices
            
        Returns:
            DataFrame with growth_future_30d added
        """
        df = predictions_df.copy()
        
        # Calculate growth_future_30d if not present
        if 'growth_future_30d' not in df.columns:
            # Use ret_30d to reconstruct growth_future_30d
            # ret_30d = (future_price / current_price) - 1
            # growth_future_30d = ret_30d + 1 = future_price / current_price
            if 'ret_30d' in df.columns:
                df['growth_future_30d'] = df['ret_30d'] + 1
                self.logger.info("Calculated growth_future_30d from ret_30d")
            else:
                # If no future returns available, cannot calculate
                raise ValueError("Cannot calculate growth_future_30d: no ret_30d column found")
        
        # Also create binary target following Module 04
        if 'is_positive_growth_30d_future' not in df.columns:
            df['is_positive_growth_30d_future'] = (df['growth_future_30d'] > 1).astype(int)
            self.logger.info("Created is_positive_growth_30d_future from growth_future_30d")
        
        return df
    
    def simulate_prediction_strategy(self, 
                                   predictions_df: pd.DataFrame,
                                   prediction_column: str,
                                   threshold: float = 0.6) -> Dict:
        """
        Simulate trading strategy for one prediction following Module 04 section 3.1.1.
        
        Exactly replicates the calculation from Module 04:
        - sim1_gross_rev = prediction * 100 * (growth_future_30d - 1)
        - sim1_fees = -prediction * 100 * 0.002  
        - sim1_net_rev = sim1_gross_rev + sim1_fees
        
        Args:
            predictions_df: DataFrame with predictions and future growth
            prediction_column: Name of prediction probability column
            threshold: Probability threshold for signal generation
            
        Returns:
            Dictionary with simulation results
        """
        # Ensure we have growth_future_30d
        df = self.calculate_growth_future_30d(predictions_df)
        
        # Generate binary signals based on threshold
        signal_column = f'pred_{prediction_column}_threshold_{threshold:.2f}'
        df[signal_column] = (df[prediction_column] >= threshold).astype(int)
        
        # Calculate financial results following Module 04 exactly
        # Line 2203: new_df['sim1_gross_rev_pred6'] = new_df[pred] * 100 * (new_df['growth_future_30d']-1)
        gross_revenue_col = f'sim1_gross_rev_{signal_column}'
        df[gross_revenue_col] = df[signal_column] * self.investment_per_signal * (df['growth_future_30d'] - 1)
        
        # Line 2208: new_df['sim1_fees_pred6'] = -new_df[pred] * 100 * 0.002
        fees_col = f'sim1_fees_{signal_column}'
        df[fees_col] = -df[signal_column] * self.investment_per_signal * self.transaction_cost
        
        # Line 2211: new_df['sim1_net_rev_pred6'] = new_df['sim1_gross_rev_pred6'] + new_df['sim1_fees_pred6']
        net_revenue_col = f'sim1_net_rev_{signal_column}'
        df[net_revenue_col] = df[gross_revenue_col] + df[fees_col]
        
        # Filter to positive predictions (trades executed)
        trades_filter = df[signal_column] == 1
        trades_df = df[trades_filter].copy()
        
        if len(trades_df) == 0:
            return {
                'threshold': threshold,
                'num_trades': 0,
                'gross_revenue': 0,
                'fees': 0,
                'net_revenue': 0,
                'avg_net_revenue_per_trade': 0,
                'hit_rate': 0,
                'sharpe_ratio': 0,
                'cagr': 0,
                'trades_df': trades_df
            }
        
        # Aggregate results following Module 04 patterns
        num_trades = len(trades_df)
        gross_revenue = trades_df[gross_revenue_col].sum()
        fees = trades_df[fees_col].sum()  # Already negative
        net_revenue = trades_df[net_revenue_col].sum()
        
        # Performance metrics
        avg_net_revenue_per_trade = net_revenue / num_trades if num_trades > 0 else 0
        hit_rate = (trades_df['growth_future_30d'] > 1).mean() if num_trades > 0 else 0
        
        # Risk metrics (simplified)
        trade_returns = trades_df[net_revenue_col] / self.investment_per_signal
        sharpe_ratio = (trade_returns.mean() / trade_returns.std() * np.sqrt(252)) if trade_returns.std() > 0 else 0
        
        # CAGR calculation following Module 04 approximation
        # Module 04 uses: sim1_capital = 100 * 30 * q75_investments_per_day
        # Then: sim1_CAGR = ((capital + net_revenue) / capital) ** (1/4)
        
        # Estimate required capital (simplified)
        daily_trades = trades_df.groupby(trades_df.index.date).size()
        avg_daily_trades = daily_trades.mean() if len(daily_trades) > 0 else 1
        required_capital = self.investment_per_signal * 30 * avg_daily_trades  # 30 days buffer
        
        # Time period for CAGR (approximate)
        date_range = (df.index.max() - df.index.min()).days / 365.25
        if date_range > 0 and required_capital > 0:
            total_value = required_capital + net_revenue
            cagr = (total_value / required_capital) ** (1 / date_range) - 1
        else:
            cagr = 0
        
        return {
            'threshold': threshold,
            'num_trades': num_trades,
            'gross_revenue': gross_revenue,
            'fees': fees,
            'net_revenue': net_revenue,
            'avg_net_revenue_per_trade': avg_net_revenue_per_trade,
            'hit_rate': hit_rate,
            'sharpe_ratio': sharpe_ratio,
            'cagr': cagr,
            'required_capital': required_capital,
            'trades_df': trades_df
        }
    
    def run_threshold_sweep(self, 
                           predictions_df: pd.DataFrame,
                           prediction_columns: List[str],
                           thresholds: List[float] = None) -> Dict[str, pd.DataFrame]:
        """
        Run threshold sweep analysis for multiple models following Module 04 patterns.
        
        Args:
            predictions_df: DataFrame with predictions
            prediction_columns: List of prediction column names
            thresholds: List of thresholds to test
            
        Returns:
            Dictionary with results DataFrames for each model
        """
        if thresholds is None:
            thresholds = THRESHOLD_SWEEP
        
        results = {}
        
        for pred_col in prediction_columns:
            self.logger.info(f"Running threshold sweep for {pred_col}...")
            
            model_results = []
            for threshold in thresholds:
                try:
                    result = self.simulate_prediction_strategy(
                        predictions_df, pred_col, threshold
                    )
                    model_results.append(result)
                except Exception as e:
                    self.logger.warning(f"Failed simulation for {pred_col} at threshold {threshold}: {e}")
                    continue
            
            if model_results:
                # Convert to DataFrame
                results_df = pd.DataFrame(model_results)
                results_df['model'] = pred_col.replace('y_proba_', '')  # Clean model name
                results[pred_col] = results_df
            
        return results
    
    def create_module04_style_summary(self, threshold_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Create summary table following Module 04 style reporting.
        
        Args:
            threshold_results: Results from threshold sweep
            
        Returns:
            Summary DataFrame in Module 04 style
        """
        summary_rows = []
        
        for model_name, results_df in threshold_results.items():
            if len(results_df) == 0:
                continue
                
            # Find best configuration by net revenue (Module 04 pattern)
            best_config = results_df.loc[results_df['net_revenue'].idxmax()]
            
            summary_rows.append({
                'prediction': model_name,
                'best_threshold': best_config['threshold'],
                'sim1_count_investments': best_config['num_trades'],
                'sim1_gross_rev': best_config['gross_revenue'],
                'sim1_fees': best_config['fees'],
                'sim1_net_rev': best_config['net_revenue'],
                'sim1_average_net_revenue': best_config['avg_net_revenue_per_trade'],
                'sim1_hit_rate': best_config['hit_rate'],
                'sim1_sharpe_ratio': best_config['sharpe_ratio'],
                'sim1_CAGR': best_config['cagr']
            })
        
        summary_df = pd.DataFrame(summary_rows)
        
        # Add growth metrics like Module 04
        if len(summary_df) > 0:
            summary_df['sim1_growth_capital_4y'] = (
                (summary_df['sim1_net_rev'] + summary_df['sim1_count_investments'] * self.investment_per_signal) 
                / (summary_df['sim1_count_investments'] * self.investment_per_signal)
            )
        
        return summary_df.sort_values('sim1_CAGR', ascending=False)


def run_enhanced_simulation(predictions_df: pd.DataFrame) -> Dict:
    """
    Run complete enhanced simulation following Module 04 patterns.
    
    Args:
        predictions_df: DataFrame with model predictions
        
    Returns:
        Dictionary with simulation results
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting enhanced simulation with Module 04 corrections...")
    
    # Initialize enhanced simulator
    simulator = EnhancedTradingSimulator()
    
    # Extract prediction columns
    pred_columns = [col for col in predictions_df.columns if col.startswith('y_proba_')]
    
    if not pred_columns:
        logger.error("No prediction columns found (expecting 'y_proba_*')")
        return {}
    
    logger.info(f"Found prediction columns: {pred_columns}")
    
    # Run threshold sweep
    threshold_results = simulator.run_threshold_sweep(predictions_df, pred_columns)
    
    # Create Module 04 style summary
    summary_df = simulator.create_module04_style_summary(threshold_results)
    
    # Export results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = REPORTS_DIR / f"enhanced_simulation_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary
    summary_file = results_dir / "module04_style_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    logger.info(f"Saved Module 04 style summary to {summary_file}")
    
    # Save detailed results
    for model_name, results_df in threshold_results.items():
        model_file = results_dir / f"threshold_analysis_{model_name.replace('y_proba_', '')}.csv"
        results_df.to_csv(model_file, index=False)
    
    # Create visualization following Module 04 bubble chart pattern
    if len(summary_df) > 0:
        create_module04_bubble_chart(summary_df, results_dir)
    
    logger.info(f"Enhanced simulation completed. Results in {results_dir}")
    
    return {
        'summary': summary_df,
        'threshold_results': threshold_results,
        'results_dir': results_dir,
        'corrections_applied': [
            'Fixed PnL formula: investment * (growth_future_30d - 1)',
            'Proper transaction costs: 0.2% total',
            'Module 04 naming conventions',
            'Vectorized simulation patterns'
        ]
    }


def create_module04_bubble_chart(summary_df: pd.DataFrame, output_dir: Path):
    """
    Create bubble chart following Module 04 visualization patterns.
    
    Replicates Module 04 plotly bubble chart:
    - X-axis: Average investments per day (proxy for time spent)
    - Y-axis: CAGR
    - Size: Capital growth
    - Text: Strategy name
    """
    try:
        import plotly.express as px
        
        # Estimate avg investments per day (simplified)
        summary_df_viz = summary_df.copy()
        summary_df_viz['avg_investments_per_day'] = summary_df_viz['sim1_count_investments'] / 1000  # Approximate
        
        fig = px.scatter(
            summary_df_viz,
            x='avg_investments_per_day',
            y='sim1_CAGR',
            size='sim1_growth_capital_4y',
            text='prediction',
            title='Compound Annual Growth vs. Time spent (Module 04 Style)',
            labels={
                'avg_investments_per_day': 'Average investments per day (approx)',
                'sim1_CAGR': 'CAGR',
                'sim1_growth_capital_4y': 'Capital Growth'
            },
            height=600
        )
        
        fig.update_traces(textposition='top center')
        
        # Save as HTML
        chart_file = output_dir / "module04_bubble_chart.html"
        fig.write_html(chart_file)
        
        # Also save as static image if possible
        try:
            static_file = output_dir / "module04_bubble_chart.png"
            fig.write_image(static_file)
        except:
            pass  # kaleido not available
            
    except ImportError:
        # Fallback to matplotlib
        plt.figure(figsize=(12, 8))
        plt.scatter(
            summary_df['sim1_count_investments'], 
            summary_df['sim1_CAGR'],
            s=summary_df['sim1_growth_capital_4y'] * 100,
            alpha=0.6
        )
        
        for i, row in summary_df.iterrows():
            plt.annotate(
                row['prediction'], 
                (row['sim1_count_investments'], row['sim1_CAGR']),
                xytext=(5, 5), textcoords='offset points'
            )
        
        plt.xlabel('Number of Trades')
        plt.ylabel('CAGR')
        plt.title('Enhanced Simulation Results (Module 04 Style)')
        plt.grid(True, alpha=0.3)
        
        chart_file = output_dir / "enhanced_simulation_chart.png"
        plt.savefig(chart_file, dpi=150, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    # Test with sample data
    print("Enhanced Trading Simulator with Module 04 corrections")
    print("Key fixes:")
    print("  1. PnL = investment * (growth_future_30d - 1)")
    print("  2. Transaction costs = 0.2% total")
    print("  3. Module 04 naming conventions")
    print("  4. Proper vectorized simulation")