"""
Feature engineering module for Cross-Market 30D Directional Forecasting project.
Creates technical indicators, cross-sectional features, and target variables.

This module maximally reuses patterns from 2025 Stock Markets Analytics Zoomcamp:
- Module 02: rolling calculations and shift operations
- Module 04: technical indicators (SMA, RSI, MACD, volatility)  
- Module 05: advanced feature engineering and data transformations
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path
import warnings

from config import (
    SMA_PERIODS, EMA_PERIODS, RSI_PERIOD, STOCH_PERIODS, WILLIAMS_R_PERIOD,
    CCI_PERIOD, ATR_PERIOD, BOLLINGER_PERIOD, BOLLINGER_STD,
    RETURN_PERIODS, VOLATILITY_PERIODS, DISTANCE_PERIODS,
    PREDICTION_HORIZON_DAYS, PROCESSED_DATA_DIR, TRAIN_SPLIT, VALIDATION_SPLIT, TEST_SPLIT
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    Feature engineering class for creating technical indicators and target variables.
    
    Features:
    - Technical indicators: SMA, EMA, RSI, MACD, Bollinger Bands, ATR
    - Price features: returns, volatility, price lags, ratios  
    - Relative features: ratios to moving averages, momentum indicators
    - Cross-sectional features: z-scores and ranks across tickers by date
    - Target variable: binary classification for 30-day growth/decline
    """
    
    def __init__(self):
        """Initialize FeatureEngineer."""
        logger.info("FeatureEngineer initialized")
    
    def add_basic_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic return features using patterns from Module 02.
        
        Args:
            df: DataFrame with OHLCV data, grouped by symbol
            
        Returns:
            DataFrame with added return features
        """
        logger.debug("Adding basic return features...")
        
        for period in RETURN_PERIODS:
            # Historical returns (look back)
            df[f'ret_{period}d'] = df.groupby(level='symbol')['Close'].pct_change(periods=period)
            
            # Future returns for target creation (look forward) 
            df[f'ret_future_{period}d'] = df.groupby(level='symbol')['Close'].pct_change(periods=-period)
        
        # Log returns (more stable for ML)
        df['log_ret_1d'] = (
            df.groupby(level='symbol')['Close']
            .apply(lambda x: np.log(x / x.shift(1)))
            .droplevel(0)  # Remove the groupby level
        )
        
        return df
    
    def add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility features using rolling standard deviation.
        
        Reuses pattern from Module 05: volatility calculation.
        """
        logger.debug("Adding volatility features...")
        
        for period in VOLATILITY_PERIODS:
            # Rolling volatility of daily returns (annualized)
            df[f'vol_{period}d'] = (
                df.groupby(level='symbol')['ret_1d']
                .rolling(period, min_periods=period//2)
                .std() * np.sqrt(252)  # Annualized
            ).droplevel(0)  # Remove the groupby level
        
        return df
    
    def add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Simple and Exponential Moving Averages.
        
        Reuses patterns from Module 05: SMA10, SMA20 calculations.
        """
        logger.debug("Adding moving average features...")
        
        # Simple Moving Averages
        for period in SMA_PERIODS:
            df[f'sma_{period}'] = (
                df.groupby(level='symbol')['Close']
                .rolling(period, min_periods=period//2)
                .mean()
            ).droplevel(0)
        
        # Exponential Moving Averages
        for period in EMA_PERIODS:
            df[f'ema_{period}'] = (
                df.groupby(level='symbol')['Close']
                .ewm(span=period, adjust=False)
                .mean()
            ).droplevel(0)
        
        return df
    
    def add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum indicators: RSI, Stochastic, Williams %R, CCI.
        """
        logger.debug("Adding momentum indicators...")
        
        # RSI (Relative Strength Index)
        df = self._add_rsi(df, RSI_PERIOD)
        
        # Stochastic Oscillator
        df = self._add_stochastic(df, STOCH_PERIODS)
        
        # Williams %R
        df = self._add_williams_r(df, WILLIAMS_R_PERIOD)
        
        # Commodity Channel Index (CCI)
        df = self._add_cci(df, CCI_PERIOD)
        
        return df
    
    def _add_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate RSI using standard formula."""
        def calculate_rsi(close_prices):
            delta = close_prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        df[f'rsi_{period}'] = (
            df.groupby('symbol')['Close']
            .apply(calculate_rsi)
            .reset_index(level=0, drop=True)
        )
        
        return df
    
    def _add_stochastic(self, df: pd.DataFrame, periods: Tuple[int, int, int]) -> pd.DataFrame:
        """Calculate Stochastic Oscillator (%K and %D)."""
        k_period, d_period, smooth = periods
        
        def calculate_stoch(group):
            high_roll = group['High'].rolling(k_period).max()
            low_roll = group['Low'].rolling(k_period).min()
            
            # %K calculation
            stoch_k = 100 * (group['Close'] - low_roll) / (high_roll - low_roll)
            
            # %D calculation (moving average of %K)
            stoch_d = stoch_k.rolling(d_period).mean()
            
            return pd.DataFrame({
                'stoch_k': stoch_k,
                'stoch_d': stoch_d
            })
        
        stoch_result = df.groupby('symbol').apply(calculate_stoch).reset_index(level=0, drop=True)
        df['stoch_k'] = stoch_result['stoch_k']
        df['stoch_d'] = stoch_result['stoch_d']
        
        return df
    
    def _add_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Williams %R."""
        def calculate_williams_r(group):
            high_roll = group['High'].rolling(period).max()
            low_roll = group['Low'].rolling(period).min()
            williams_r = -100 * (high_roll - group['Close']) / (high_roll - low_roll)
            return williams_r
        
        df['williams_r'] = (
            df.groupby('symbol')
            .apply(calculate_williams_r)
            .reset_index(level=0, drop=True)
        )
        
        return df
    
    def _add_cci(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Calculate Commodity Channel Index."""
        def calculate_cci(group):
            typical_price = (group['High'] + group['Low'] + group['Close']) / 3
            sma_tp = typical_price.rolling(period).mean()
            mad = typical_price.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())))
            cci = (typical_price - sma_tp) / (0.015 * mad)
            return cci
        
        df['cci'] = (
            df.groupby('symbol')
            .apply(calculate_cci)
            .reset_index(level=0, drop=True)
        )
        
        return df
    
    def add_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        Add MACD indicator.
        """
        logger.debug("Adding MACD features...")
        
        def calculate_macd(close_prices):
            ema_fast = close_prices.ewm(span=fast).mean()
            ema_slow = close_prices.ewm(span=slow).mean()
            
            macd_line = ema_fast - ema_slow
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            
            return pd.DataFrame({
                'macd': macd_line,
                'macd_signal': signal_line,
                'macd_hist': histogram
            })
        
        macd_result = df.groupby('symbol')['Close'].apply(calculate_macd).reset_index(level=0, drop=True)
        df['macd'] = macd_result['macd']
        df['macd_signal'] = macd_result['macd_signal']
        df['macd_hist'] = macd_result['macd_hist']
        
        return df
    
    def add_bollinger_bands(self, df: pd.DataFrame, period: int = BOLLINGER_PERIOD, 
                           std_dev: float = BOLLINGER_STD) -> pd.DataFrame:
        """Add Bollinger Bands."""
        logger.debug("Adding Bollinger Bands...")
        
        def calculate_bollinger(close_prices):
            sma = close_prices.rolling(period).mean()
            std = close_prices.rolling(period).std()
            
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            return pd.DataFrame({
                'bb_upper': upper_band,
                'bb_lower': lower_band,
                'bb_width': (upper_band - lower_band) / sma  # Normalized width
            })
        
        bb_result = df.groupby('symbol')['Close'].apply(calculate_bollinger).reset_index(level=0, drop=True)
        df['bb_upper'] = bb_result['bb_upper']
        df['bb_lower'] = bb_result['bb_lower']  
        df['bb_width'] = bb_result['bb_width']
        
        return df
    
    def add_atr(self, df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.DataFrame:
        """Add Average True Range."""
        logger.debug("Adding ATR features...")
        
        def calculate_atr(group):
            high_low = group['High'] - group['Low']
            high_close_prev = np.abs(group['High'] - group['Close'].shift(1))
            low_close_prev = np.abs(group['Low'] - group['Close'].shift(1))
            
            true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
            atr = true_range.rolling(period).mean()
            
            return atr
        
        df['atr_14'] = (
            df.groupby('symbol')
            .apply(calculate_atr)
            .reset_index(level=0, drop=True)
        )
        
        return df
    
    def add_price_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add price ratios to moving averages and distance measures.
        """
        logger.debug("Adding price ratio features...")
        
        # Price to SMA ratios
        for period in [20, 50, 200]:
            if f'sma_{period}' in df.columns:
                df[f'price_sma_{period}'] = df['Close'] / df[f'sma_{period}']
        
        # Distance to highs/lows
        for period in DISTANCE_PERIODS:
            # Distance to high/low (as percentage)
            high_roll = df.groupby('symbol')['High'].rolling(period).max().reset_index(level=0, drop=True)
            low_roll = df.groupby('symbol')['Low'].rolling(period).min().reset_index(level=0, drop=True)
            
            df[f'dist_high_{period}d'] = (df['Close'] - high_roll) / high_roll
            df[f'dist_low_{period}d'] = (df['Close'] - low_roll) / low_roll
        
        return df
    
    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume-based features.
        
        Reuses pattern from Module 04: ln_volume calculation.
        """
        logger.debug("Adding volume features...")
        
        # Log volume (to handle extreme values)
        df['ln_volume'] = np.log(df['Volume'] + 1e-6)  # Add small constant to avoid log(0)
        
        # Volume moving average
        df['volume_sma_20'] = (
            df.groupby('symbol')['Volume']
            .rolling(20, min_periods=10)
            .mean()
        ).reset_index(level=0, drop=True)
        
        # Volume ratio to MA
        df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
        
        return df
    
    def add_cross_sectional_features(self, df: pd.DataFrame, 
                                   base_features: List[str] = None) -> pd.DataFrame:
        """
        Add cross-sectional features: z-scores and ranks across tickers by date.
        """
        logger.debug("Adding cross-sectional features...")
        
        if base_features is None:
            base_features = ['ret_20d', 'vol_20d', 'rsi_14', 'Close']
        
        # Filter existing features
        base_features = [f for f in base_features if f in df.columns]
        
        for feature in base_features:
            # Z-score across tickers by date
            df[f'{feature}_zscore'] = (
                df.groupby(df.index)[feature]
                .transform(lambda x: (x - x.mean()) / x.std())
            )
            
            # Rank percentile across tickers by date
            df[f'{feature}_rank'] = (
                df.groupby(df.index)[feature]
                .rank(pct=True)
            )
        
        return df
    
    def add_target_variables(self, df: pd.DataFrame, horizon: int = PREDICTION_HORIZON_DAYS) -> pd.DataFrame:
        """
        Add target variables for prediction.
        
        Reuses pattern from Module 02: future growth calculations.
        """
        logger.debug(f"Adding target variables with {horizon}-day horizon...")
        
        # Future return (continuous target)
        df[f'ret_{horizon}d'] = df.groupby('symbol')['Close'].pct_change(periods=-horizon)
        
        # Binary classification target (up/down)
        df[f'y_{horizon}d'] = (df[f'ret_{horizon}d'] > 0).astype(int)
        
        # Remove rows where target is undefined (near end of series)
        df = df.dropna(subset=[f'ret_{horizon}d'])
        
        return df
    
    def add_lag_features(self, df: pd.DataFrame, periods: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
        """
        Add lagged price features.
        """
        logger.debug("Adding lag features...")
        
        for period in periods:
            df[f'close_lag_{period}'] = df.groupby('symbol')['Close'].shift(period)
            df[f'volume_lag_{period}'] = df.groupby('symbol')['Volume'].shift(period)
        
        return df
    
    def add_all_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical features in proper order.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all technical features added
        """
        logger.info("Adding all technical features...")
        
        # Ensure proper grouping structure
        if 'symbol' not in df.columns:
            raise ValueError("DataFrame must have 'symbol' column")
        
        # Create MultiIndex if not present to avoid duplicate index issues
        if not isinstance(df.index, pd.MultiIndex):
            df = df.reset_index().set_index(['Date', 'symbol']).sort_index()
        
        # 1. Basic returns and volatility
        df = self.add_basic_returns(df)
        df = self.add_volatility_features(df)
        
        # 2. Moving averages
        df = self.add_moving_averages(df)
        
        # 3. Technical indicators
        df = self.add_momentum_indicators(df)
        df = self.add_macd(df)
        df = self.add_bollinger_bands(df)
        df = self.add_atr(df)
        
        # 4. Price ratios and distances
        df = self.add_price_ratios(df)
        
        # 5. Volume features
        df = self.add_volume_features(df)
        
        # 6. Lag features
        df = self.add_lag_features(df)
        
        # 7. Cross-sectional features (need date-level grouping)
        df = self.add_cross_sectional_features(df)
        
        # 8. Target variables (must be last)
        df = self.add_target_variables(df)
        
        logger.info(f"Technical features added. Final shape: {df.shape}")
        
        return df
    
    def merge_with_macro(self, ticker_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge ticker data with macro indicators by date.
        
        Args:
            ticker_df: DataFrame with ticker data and technical features
            macro_df: DataFrame with macro indicators
            
        Returns:
            Merged DataFrame
        """
        logger.info("Merging ticker data with macro indicators...")
        
        # Ensure both have date index
        if not isinstance(ticker_df.index, pd.DatetimeIndex):
            raise ValueError("ticker_df must have DatetimeIndex")
            
        if not isinstance(macro_df.index, pd.DatetimeIndex):
            raise ValueError("macro_df must have DatetimeIndex")
        
        # Forward fill macro data to handle missing values
        macro_df_filled = macro_df.fillna(method='ffill').fillna(method='bfill')
        
        # Merge on date index
        merged_df = ticker_df.join(macro_df_filled, how='left')
        
        # Forward fill any remaining macro NaNs
        macro_cols = macro_df.columns.tolist()
        merged_df[macro_cols] = merged_df[macro_cols].fillna(method='ffill')
        
        logger.info(f"Merged data shape: {merged_df.shape}")
        logger.info(f"Macro columns added: {macro_cols}")
        
        return merged_df

def create_unified_dataset(ticker_data: pd.DataFrame, macro_data: pd.DataFrame, 
                          save_path: str = None) -> pd.DataFrame:
    """
    Convenience function to create unified dataset with all features.
    
    Args:
        ticker_data: Raw ticker data from data_loader
        macro_data: Raw macro data from data_loader
        save_path: Optional path to save the dataset
        
    Returns:
        Unified dataset with all features
    """
    logger.info("Creating unified dataset...")
    
    # Initialize feature engineer
    fe = FeatureEngineer()
    
    # Add all technical features
    featured_data = fe.add_all_technical_features(ticker_data)
    
    # Merge with macro data
    unified_data = fe.merge_with_macro(featured_data, macro_data)
    
    # Save if path provided
    if save_path:
        unified_data.to_parquet(save_path)
        logger.info(f"Unified dataset saved to {save_path}")
    
    logger.info("Unified dataset creation completed!")
    logger.info(f"Final dataset shape: {unified_data.shape}")
    logger.info(f"Feature columns: {len([c for c in unified_data.columns if c not in ['symbol', 'market']])}")
    
    return unified_data

def add_temporal_split(df: pd.DataFrame, 
                      train_split: float = TRAIN_SPLIT,
                      validation_split: float = VALIDATION_SPLIT,
                      test_split: float = TEST_SPLIT) -> pd.DataFrame:
    """
    Add temporal split column to dataset following lecture patterns.
    Splits data chronologically into train/validation/test sets.
    
    Args:
        df: DataFrame with Date index or date column
        train_split: Proportion for training (default 0.7)
        validation_split: Proportion for validation (default 0.2) 
        test_split: Proportion for test (default 0.1)
        
    Returns:
        DataFrame with added 'split' column
    """
    logger.info("Adding temporal split...")
    
    # Ensure we have a copy
    df_split = df.copy()
    
    # Get date column (handle both Date index and date column)
    if 'date' in df_split.columns:
        date_col = 'date'
    elif 'Date' in df_split.columns:
        date_col = 'Date'
    elif df_split.index.name == 'Date' or isinstance(df_split.index, pd.DatetimeIndex):
        # Reset index to work with dates as column
        df_split = df_split.reset_index()
        date_col = 'Date' if 'Date' in df_split.columns else 'date'
    else:
        raise ValueError("No date column or DatetimeIndex found")
    
    # Sort by date to ensure chronological order
    df_split = df_split.sort_values(date_col)
    
    # Get unique dates
    unique_dates = sorted(df_split[date_col].unique())
    n_dates = len(unique_dates)
    
    # Calculate split indices
    train_end_idx = int(n_dates * train_split)
    validation_end_idx = int(n_dates * (train_split + validation_split))
    
    # Split dates
    train_dates = unique_dates[:train_end_idx]
    validation_dates = unique_dates[train_end_idx:validation_end_idx]
    test_dates = unique_dates[validation_end_idx:]
    
    # Add split column
    df_split['split'] = 'test'  # Default to test
    df_split.loc[df_split[date_col].isin(train_dates), 'split'] = 'train'
    df_split.loc[df_split[date_col].isin(validation_dates), 'split'] = 'validation'
    
    # Log split information
    train_count = (df_split['split'] == 'train').sum()
    val_count = (df_split['split'] == 'validation').sum()  
    test_count = (df_split['split'] == 'test').sum()
    
    logger.info(f"Temporal split created:")
    logger.info(f"  Train: {train_count:,} rows ({train_count/len(df_split):.1%}) | Dates: {min(train_dates)} to {max(train_dates)}")
    logger.info(f"  Validation: {val_count:,} rows ({val_count/len(df_split):.1%}) | Dates: {min(validation_dates)} to {max(validation_dates)}")
    logger.info(f"  Test: {test_count:,} rows ({test_count/len(df_split):.1%}) | Dates: {min(test_dates)} to {max(test_dates)}")
    
    return df_split

if __name__ == "__main__":
    # Test the feature engineer
    print("Testing FeatureEngineer...")
    
    # Create sample data
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    symbols = ['AAPL', 'MSFT']
    
    # Create sample ticker data
    data = []
    for symbol in symbols:
        for date in dates:
            data.append({
                'Date': date,
                'symbol': symbol,
                'Open': 100 + np.random.randn(),
                'High': 102 + np.random.randn(),
                'Low': 98 + np.random.randn(),
                'Close': 100 + np.random.randn(),
                'Volume': 1000000 + np.random.randint(0, 500000),
                'market': 'US'
            })
    
    sample_df = pd.DataFrame(data).set_index('Date')
    
    # Test feature engineering
    fe = FeatureEngineer()
    result = fe.add_all_technical_features(sample_df)
    
    print(f"Sample result shape: {result.shape}")
    print(f"Technical features: {[c for c in result.columns if c not in ['symbol', 'market', 'Open', 'High', 'Low', 'Close', 'Volume']]}")