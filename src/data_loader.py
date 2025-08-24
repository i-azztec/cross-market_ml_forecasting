"""
Data loading module for Cross-Market 30D Directional Forecasting project.
Handles downloading and caching of stock/ETF/macro data from Yahoo Finance.

This module maximally reuses patterns from 2025 Stock Markets Analytics Zoomcamp:
- Module 01: yfinance download patterns with error handling
- Module 02: data caching and incremental updates
"""

import pandas as pd
import yfinance as yf
import pandas_datareader as pdr
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime, date
from tqdm import tqdm
import warnings
import pickle
import time

from config import (
    ALL_TICKERS, MACRO_INDICATORS, RAW_DATA_DIR, DEFAULT_START_DATE,
    DEFAULT_END_DATE, RAW_DATA_PATTERN, CACHE_ENABLED, CACHE_EXPIRY_DAYS,
    get_ticker_market
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class DataLoader:
    """
    Data loader class for stocks, ETFs, and macro indicators.
    
    Features:
    - Automatic caching in parquet files for faster re-runs
    - Error handling for individual ticker failures with logging
    - Support for different ticker formats (US, EU, Asia with suffixes)
    - Automatic forward-fill for macro data gaps
    - Validation for anomalous values (negative prices/volumes)
    """
    
    def __init__(self, cache_dir: str = None, cache_enabled: bool = True):
        """Initialize DataLoader with caching options."""
        self.cache_dir = Path(cache_dir) if cache_dir else RAW_DATA_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_enabled = cache_enabled and CACHE_ENABLED
        
        # Track failed downloads for reporting
        self.failed_tickers = []
        self.failed_macros = []
        
        logger.info(f"DataLoader initialized with cache_dir: {self.cache_dir}")
        logger.info(f"Cache enabled: {self.cache_enabled}")
    
    def _get_cache_path(self, symbol: str) -> Path:
        """Get cache file path for a symbol."""
        filename = RAW_DATA_PATTERN.format(symbol=symbol.replace('.', '_').replace('^', ''))
        return self.cache_dir / filename
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cache file exists and is recent enough."""
        if not cache_path.exists():
            return False
            
        if not self.cache_enabled:
            return False
            
        # Check cache age
        file_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return file_age.days < CACHE_EXPIRY_DAYS
    
    def _download_ticker_data(self, symbol: str, start_date: str, end_date: str = None) -> Optional[pd.DataFrame]:
        """
        Download OHLCV data for a single ticker using yfinance.
        
        Reuses pattern from Module 01: ticker_obj.history() method with error handling.
        """
        try:
            logger.debug(f"Downloading {symbol}...")
            
            # Create ticker object (preferred method from lectures)
            ticker_obj = yf.Ticker(symbol)
            
            # Download historical data
            data = ticker_obj.history(
                start=start_date,
                end=end_date,
                interval="1d",
                auto_adjust=True,  # Use adjusted prices
                prepost=False      # No pre/post market data
            )
            
            if data.empty:
                logger.warning(f"No data returned for {symbol}")
                return None
            
            # Convert timezone-aware index to naive (remove timezone info)
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            
            # Add symbol column for identification
            data['symbol'] = symbol
            data['market'] = get_ticker_market(symbol)
            
            # Basic validation
            if len(data) < 100:  # Require at least 100 days of data
                logger.warning(f"Insufficient data for {symbol}: {len(data)} days")
                return None
            
            # Check for anomalous values
            if (data['Close'] <= 0).any() or (data['Volume'] < 0).any():
                logger.warning(f"Anomalous values detected in {symbol}")
                # Don't return None, just log warning
            
            logger.debug(f"Successfully downloaded {symbol}: {len(data)} rows")
            return data
            
        except Exception as e:
            logger.error(f"Failed to download {symbol}: {str(e)}")
            self.failed_tickers.append(symbol)
            return None
    
    def _download_macro_data(self, indicator: str, symbol: str, start_date: str, end_date: str = None) -> Optional[pd.DataFrame]:
        """
        Download macro indicator data from Yahoo Finance or FRED.
        
        Reuses pattern from Module 01: both yfinance and pandas_datareader methods.
        """
        try:
            logger.debug(f"Downloading macro {indicator} ({symbol})...")
            
            # First try Yahoo Finance (for indicators like ^VIX, ^TNX)
            if symbol.startswith('^') or symbol in ['UUP', 'DXY']:
                ticker_obj = yf.Ticker(symbol)
                data = ticker_obj.history(
                    start=start_date,
                    end=end_date,
                    interval="1d"
                )
                
                if not data.empty:
                    # Convert timezone-aware index to naive
                    if data.index.tz is not None:
                        data.index = data.index.tz_localize(None)
                    
                    # Use Close price for macro indicators
                    macro_data = pd.DataFrame({
                        indicator: data['Close']
                    }, index=data.index)
                    
                    logger.debug(f"Successfully downloaded {indicator} from Yahoo: {len(macro_data)} rows")
                    return macro_data
            
            # Fallback to FRED if available (from lectures)
            # Note: This would require FRED API key and pandas_datareader
            # For now, we'll focus on Yahoo Finance indicators
            
            logger.warning(f"Could not download macro indicator {indicator} ({symbol})")
            self.failed_macros.append(indicator)
            return None
            
        except Exception as e:
            logger.error(f"Failed to download macro {indicator}: {str(e)}")
            self.failed_macros.append(indicator)
            return None
    
    def load_ticker_data(self, symbol: str, start_date: str = DEFAULT_START_DATE, 
                        end_date: str = DEFAULT_END_DATE, force_download: bool = False) -> Optional[pd.DataFrame]:
        """
        Load data for a single ticker with caching support.
        
        Args:
            symbol: Ticker symbol (e.g., 'AAPL', 'ASML.AS')
            start_date: Start date for data download
            end_date: End date for data download (None = current date)
            force_download: If True, ignore cache and re-download
            
        Returns:
            DataFrame with OHLCV data or None if failed
        """
        cache_path = self._get_cache_path(symbol)
        
        # Check cache first
        if not force_download and self._is_cache_valid(cache_path):
            try:
                data = pd.read_parquet(cache_path)
                logger.debug(f"Loaded {symbol} from cache: {len(data)} rows")
                return data
            except Exception as e:
                logger.warning(f"Failed to load cache for {symbol}: {e}")
        
        # Download fresh data
        data = self._download_ticker_data(symbol, start_date, end_date)
        
        if data is not None and self.cache_enabled:
            try:
                # Save to cache
                data.to_parquet(cache_path)
                logger.debug(f"Cached data for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to cache data for {symbol}: {e}")
        
        return data
    
    def load_all_tickers(self, tickers: List[str] = None, start_date: str = DEFAULT_START_DATE,
                        end_date: str = DEFAULT_END_DATE, force_download: bool = False) -> pd.DataFrame:
        """
        Load data for multiple tickers and combine into single DataFrame.
        
        Args:
            tickers: List of ticker symbols (uses ALL_TICKERS if None)
            start_date: Start date for data download
            end_date: End date for data download
            force_download: If True, ignore cache and re-download
            
        Returns:
            Combined DataFrame with MultiIndex (date, symbol)
        """
        if tickers is None:
            tickers = ALL_TICKERS
        
        logger.info(f"Loading data for {len(tickers)} tickers...")
        
        # Reset failed tracking
        self.failed_tickers = []
        
        all_data = []
        
        # Use tqdm for progress tracking (pattern from lectures)
        for symbol in tqdm(tickers, desc="Downloading tickers"):
            data = self.load_ticker_data(symbol, start_date, end_date, force_download)
            
            if data is not None:
                all_data.append(data)
            
            # Small delay to avoid overwhelming Yahoo Finance
            time.sleep(0.1)
        
        if not all_data:
            logger.error("No ticker data was successfully downloaded!")
            return pd.DataFrame()
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=False)
        
        # Reset index to make date a column for sorting
        combined_data = combined_data.reset_index()
        
        # Sort by date and symbol for consistency
        combined_data = combined_data.sort_values(['Date', 'symbol'])
        
        # Set index back to Date for time series operations
        combined_data = combined_data.set_index('Date')
        
        logger.info(f"Successfully loaded {len(all_data)}/{len(tickers)} tickers")
        logger.info(f"Combined dataset shape: {combined_data.shape}")
        
        if self.failed_tickers:
            logger.warning(f"Failed to download: {', '.join(self.failed_tickers)}")
        
        return combined_data
    
    def load_macro_data(self, indicators: Dict[str, str] = None, start_date: str = DEFAULT_START_DATE,
                       end_date: str = DEFAULT_END_DATE, force_download: bool = False) -> pd.DataFrame:
        """
        Load macro indicator data.
        
        Args:
            indicators: Dict mapping indicator names to Yahoo symbols
            start_date: Start date for data download  
            end_date: End date for data download
            force_download: If True, ignore cache and re-download
            
        Returns:
            DataFrame with macro indicators by date
        """
        if indicators is None:
            indicators = MACRO_INDICATORS
        
        logger.info(f"Loading {len(indicators)} macro indicators...")
        
        # Reset failed tracking
        self.failed_macros = []
        
        macro_data = []
        
        for indicator, symbol in tqdm(indicators.items(), desc="Downloading macro"):
            cache_path = self._get_cache_path(f"macro_{indicator}")
            
            # Check cache
            if not force_download and self._is_cache_valid(cache_path):
                try:
                    data = pd.read_parquet(cache_path)
                    macro_data.append(data)
                    logger.debug(f"Loaded {indicator} from cache")
                    continue
                except Exception as e:
                    logger.warning(f"Failed to load cache for {indicator}: {e}")
            
            # Download fresh data
            data = self._download_macro_data(indicator, symbol, start_date, end_date)
            
            if data is not None:
                macro_data.append(data)
                
                # Cache the data
                if self.cache_enabled:
                    try:
                        data.to_parquet(cache_path)
                        logger.debug(f"Cached macro data for {indicator}")
                    except Exception as e:
                        logger.warning(f"Failed to cache macro data for {indicator}: {e}")
            
            time.sleep(0.1)  # Rate limiting
        
        if not macro_data:
            logger.warning("No macro data was successfully downloaded!")
            return pd.DataFrame()
        
        # Combine all macro data
        combined_macro = pd.concat(macro_data, axis=1)
        
        # Forward fill missing values for macro data (common pattern)
        combined_macro = combined_macro.fillna(method='ffill').fillna(method='bfill')
        
        logger.info(f"Successfully loaded {len(macro_data)}/{len(indicators)} macro indicators")
        logger.info(f"Macro dataset shape: {combined_macro.shape}")
        
        if self.failed_macros:
            logger.warning(f"Failed to download macro: {', '.join(self.failed_macros)}")
        
        return combined_macro
    
    def load_all_data(self, tickers: List[str] = None, indicators: Dict[str, str] = None,
                     start_date: str = DEFAULT_START_DATE, end_date: str = DEFAULT_END_DATE,
                     force_download: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load both ticker and macro data.
        
        Returns:
            Tuple of (ticker_data, macro_data) DataFrames
        """
        logger.info("Starting complete data download...")
        
        # Load ticker data
        ticker_data = self.load_all_tickers(tickers, start_date, end_date, force_download)
        
        # Load macro data
        macro_data = self.load_macro_data(indicators, start_date, end_date, force_download)
        
        # Summary statistics
        if not ticker_data.empty:
            date_range = f"{ticker_data.index.min().date()} to {ticker_data.index.max().date()}"
            logger.info(f"Ticker data date range: {date_range}")
        
        if not macro_data.empty:
            date_range = f"{macro_data.index.min().date()} to {macro_data.index.max().date()}"
            logger.info(f"Macro data date range: {date_range}")
        
        return ticker_data, macro_data
    
    def get_download_summary(self) -> Dict:
        """Get summary of download results."""
        return {
            'failed_tickers': self.failed_tickers,
            'failed_macros': self.failed_macros,
            'cache_dir': str(self.cache_dir),
            'cache_enabled': self.cache_enabled
        }

# Convenience functions for CLI usage
def download_tickers(tickers: List[str] = None, start_date: str = DEFAULT_START_DATE,
                    end_date: str = DEFAULT_END_DATE, force_download: bool = False) -> pd.DataFrame:
    """Convenience function to download ticker data."""
    loader = DataLoader()
    return loader.load_all_tickers(tickers, start_date, end_date, force_download)

def download_macro(indicators: Dict[str, str] = None, start_date: str = DEFAULT_START_DATE,
                  end_date: str = DEFAULT_END_DATE, force_download: bool = False) -> pd.DataFrame:
    """Convenience function to download macro data."""
    loader = DataLoader()
    return loader.load_macro_data(indicators, start_date, end_date, force_download)

def download_all_data(tickers: List[str] = None, indicators: Dict[str, str] = None,
                     start_date: str = DEFAULT_START_DATE, end_date: str = DEFAULT_END_DATE,
                     force_download: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience function to download all data."""
    loader = DataLoader()
    return loader.load_all_data(tickers, indicators, start_date, end_date, force_download)

if __name__ == "__main__":
    # Test the data loader
    print("Testing DataLoader...")
    
    # Test with a small subset
    test_tickers = ['AAPL', 'MSFT', 'GOOGL']
    test_macros = {'vix': '^VIX', 'tnx10y': '^TNX'}
    
    loader = DataLoader()
    ticker_data, macro_data = loader.load_all_data(
        tickers=test_tickers,
        indicators=test_macros,
        start_date="2020-01-01"
    )
    
    print(f"Ticker data shape: {ticker_data.shape}")
    print(f"Macro data shape: {macro_data.shape}")
    print(f"Download summary: {loader.get_download_summary()}")