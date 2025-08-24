"""
Simplified feature engineering module for Cross-Market 30D Directional Forecasting project.
Creates basic technical indicators with robust error handling.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
import logging
from pathlib import Path
import warnings

from config import PREDICTION_HORIZON_DAYS, PROCESSED_DATA_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add basic technical features using simple pandas operations.
    
    Args:
        df: DataFrame with OHLCV data including 'symbol' column
        
    Returns:
        DataFrame with basic technical features
    """
    logger.info("Adding basic technical features...")
    
    # Create a copy to avoid modifying original
    result_df = df.copy()
    
    # Process each symbol separately to avoid index issues
    all_results = []
    
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].copy().sort_index()
        
        logger.debug(f"Processing {symbol}...")
        
        # 1. Basic Returns
        symbol_df['ret_1d'] = symbol_df['Close'].pct_change()
        symbol_df['ret_5d'] = symbol_df['Close'].pct_change(periods=5)
        symbol_df['ret_10d'] = symbol_df['Close'].pct_change(periods=10)
        symbol_df['ret_20d'] = symbol_df['Close'].pct_change(periods=20)
        symbol_df['ret_30d'] = symbol_df['Close'].pct_change(periods=30)
        
        # 2. Moving Averages (from lectures)
        symbol_df['sma_5'] = symbol_df['Close'].rolling(5).mean()
        symbol_df['sma_10'] = symbol_df['Close'].rolling(10).mean()
        symbol_df['sma_20'] = symbol_df['Close'].rolling(20).mean()
        symbol_df['sma_50'] = symbol_df['Close'].rolling(50).mean()
        symbol_df['sma_200'] = symbol_df['Close'].rolling(200).mean()
        
        # 3. Exponential Moving Averages
        symbol_df['ema_12'] = symbol_df['Close'].ewm(span=12).mean()
        symbol_df['ema_26'] = symbol_df['Close'].ewm(span=26).mean()
        
        # 4. MACD
        symbol_df['macd'] = symbol_df['ema_12'] - symbol_df['ema_26']
        symbol_df['macd_signal'] = symbol_df['macd'].ewm(span=9).mean()
        symbol_df['macd_hist'] = symbol_df['macd'] - symbol_df['macd_signal']
        
        # 5. RSI (simplified version)
        delta = symbol_df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        symbol_df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # 6. Volatility (from lectures)
        symbol_df['vol_5d'] = symbol_df['ret_1d'].rolling(5).std() * np.sqrt(252)
        symbol_df['vol_10d'] = symbol_df['ret_1d'].rolling(10).std() * np.sqrt(252)
        symbol_df['vol_20d'] = symbol_df['ret_1d'].rolling(20).std() * np.sqrt(252)
        symbol_df['vol_30d'] = symbol_df['ret_1d'].rolling(30).std() * np.sqrt(252)
        
        # 7. Price ratios
        symbol_df['price_sma20'] = symbol_df['Close'] / symbol_df['sma_20']
        symbol_df['price_sma50'] = symbol_df['Close'] / symbol_df['sma_50']
        symbol_df['price_sma200'] = symbol_df['Close'] / symbol_df['sma_200']
        
        # 8. Distance to highs/lows
        symbol_df['dist_high_20d'] = (symbol_df['Close'] - symbol_df['High'].rolling(20).max()) / symbol_df['High'].rolling(20).max()
        symbol_df['dist_low_20d'] = (symbol_df['Close'] - symbol_df['Low'].rolling(20).min()) / symbol_df['Low'].rolling(20).min()
        symbol_df['dist_high_252d'] = (symbol_df['Close'] - symbol_df['High'].rolling(252).max()) / symbol_df['High'].rolling(252).max()
        symbol_df['dist_low_252d'] = (symbol_df['Close'] - symbol_df['Low'].rolling(252).min()) / symbol_df['Low'].rolling(252).min()
        
        # 9. Volume features (from lectures)
        symbol_df['ln_volume'] = np.log(symbol_df['Volume'] + 1e-6)  # Avoid log(0)
        symbol_df['volume_sma_20'] = symbol_df['Volume'].rolling(20).mean()
        
        # 10. Lag features
        symbol_df['close_lag_1'] = symbol_df['Close'].shift(1)
        symbol_df['close_lag_2'] = symbol_df['Close'].shift(2)
        symbol_df['close_lag_3'] = symbol_df['Close'].shift(3)
        symbol_df['close_lag_5'] = symbol_df['Close'].shift(5)
        symbol_df['close_lag_10'] = symbol_df['Close'].shift(10)
        
        # 11. Target variables (must be last) - following original notebook approach
        horizon = PREDICTION_HORIZON_DAYS
        
        # Forward returns using consistent calculation approach
        # growth_future_30d = Close[t+30] / Close[t] (price ratio)
        symbol_df[f'growth_future_{horizon}d'] = symbol_df['Close'].shift(-horizon) / symbol_df['Close']
        
        # ret_30d = (Close[t+30] - Close[t]) / Close[t] = growth_future_30d - 1
        symbol_df[f'ret_{horizon}d'] = symbol_df[f'growth_future_{horizon}d'] - 1
        
        # Binary target: positive return
        symbol_df[f'y_{horizon}d'] = (symbol_df[f'ret_{horizon}d'] > 0).astype(int)
        
        # Target variables to match original notebooks exactly
        symbol_df[f'is_positive_growth_{horizon}d_future'] = symbol_df[f'y_{horizon}d']
        
        all_results.append(symbol_df)
    
    # Combine all symbols
    result_df = pd.concat(all_results, ignore_index=False).sort_index()
    
    logger.info(f"Basic features added. Shape: {result_df.shape}")
    
    return result_df

def add_cross_sectional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cross-sectional features: z-scores and ranks across tickers by date.
    """
    logger.info("Adding cross-sectional features...")
    
    base_features = ['ret_20d', 'vol_20d', 'rsi_14']
    
    # Filter existing features
    base_features = [f for f in base_features if f in df.columns]
    
    for feature in base_features:
        # Z-score across tickers by date
        df[f'{feature}_zscore'] = df.groupby(df.index)[feature].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )
        
        # Rank percentile across tickers by date
        df[f'{feature}_rank'] = df.groupby(df.index)[feature].rank(pct=True)
    
    return df

def merge_with_macro(ticker_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge ticker data with macro indicators by date.
    """
    logger.info("Merging ticker data with macro indicators...")
    
    # Forward fill macro data to handle missing values
    macro_df_filled = macro_df.fillna(method='ffill').fillna(method='bfill')
    
    # Merge on date index
    merged_df = ticker_df.join(macro_df_filled, how='left')
    
    # Forward fill any remaining macro NaNs
    macro_cols = macro_df.columns.tolist()
    merged_df[macro_cols] = merged_df[macro_cols].fillna(method='ffill')
    
    logger.info(f"Merged data shape: {merged_df.shape}")
    
    return merged_df

def create_unified_dataset_simple(ticker_data: pd.DataFrame, macro_data: pd.DataFrame, 
                                 save_path: str = None) -> pd.DataFrame:
    """
    Create unified dataset with simplified feature engineering.
    """
    logger.info("Creating unified dataset (simplified approach)...")
    
    # Add basic technical features
    featured_data = add_basic_features(ticker_data)
    
    # Add cross-sectional features
    featured_data = add_cross_sectional_features(featured_data)
    
    # Merge with macro data
    unified_data = merge_with_macro(featured_data, macro_data)
    
    # Remove rows where target is undefined (near end of series)
    horizon = PREDICTION_HORIZON_DAYS
    unified_data = unified_data.dropna(subset=[f'ret_{horizon}d'])
    
    # Save if path provided
    if save_path:
        unified_data.to_parquet(save_path)
        logger.info(f"Unified dataset saved to {save_path}")
    
    logger.info("Unified dataset creation completed!")
    logger.info(f"Final dataset shape: {unified_data.shape}")
    
    # Feature summary
    all_cols = set(unified_data.columns)
    base_cols = {'symbol', 'market', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits'}
    feature_cols = all_cols - base_cols
    logger.info(f"Features created: {len(feature_cols)}")
    
    return unified_data

if __name__ == "__main__":
    # Test the simplified feature engineer
    print("Testing simplified feature engineering...")
    
    # Create sample data
    dates = pd.date_range('2020-01-01', '2021-12-31', freq='D')
    symbols = ['AAPL', 'MSFT']
    
    data = []
    for symbol in symbols:
        for date in dates:
            data.append({
                'symbol': symbol,
                'Open': 100 + np.random.randn(),
                'High': 102 + np.random.randn(),
                'Low': 98 + np.random.randn(),
                'Close': 100 + np.random.randn(),
                'Volume': 1000000 + np.random.randint(0, 500000),
                'market': 'US'
            })
    
    sample_df = pd.DataFrame(data)
    sample_df['Date'] = pd.to_datetime([d['Date'] for d in data] if 'Date' in data[0] else dates.tolist() * len(symbols))
    sample_df = sample_df.set_index('Date')
    
    # Test feature engineering
    result = add_basic_features(sample_df)
    
    print(f"Sample result shape: {result.shape}")
    print(f"Technical features: {len([c for c in result.columns if c not in ['symbol', 'market', 'Open', 'High', 'Low', 'Close', 'Volume']])}")