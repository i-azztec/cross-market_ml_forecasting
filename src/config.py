"""
Configuration module for Cross-Market 30D Directional Forecasting project.
Contains all constants, parameters, and settings.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import os

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models" 
REPORTS_DIR = DATA_DIR / "reports"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# Ensure directories exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# TEMPORAL PARAMETERS
# =============================================================================

PREDICTION_HORIZON_DAYS = 30  # forecast horizon in days
TRAIN_SPLIT = 0.7             # training set proportion
VALIDATION_SPLIT = 0.2        # validation set proportion  
TEST_SPLIT = 0.1              # test set proportion

# Data range
DEFAULT_START_DATE = "2000-01-01"
DEFAULT_END_DATE = None  # None = current date

# =============================================================================
# TRADING PARAMETERS
# =============================================================================

INVESTMENT_PER_SIGNAL = 100     # investment amount per signal in USD
PROBABILITY_THRESHOLD = 0.6     # default probability threshold for entry
TRANSACTION_COST = 0.001        # broker commission (0.1%)
THRESHOLD_SWEEP = [0.5, 0.55, 0.6, 0.65, 0.7]  # thresholds for analysis

# =============================================================================
# ASSET UNIVERSE
# =============================================================================

# US stocks (~25 tickers): large-cap, liquid companies
US_STOCKS = [
    # Technology
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META',
    'NFLX', 'ORCL', 'CSCO', 'INTC',
    # Finance
    'JPM', 'V', 'MA', 'BAC', 'WFC',
    # Healthcare
    'UNH', 'JNJ', 'PFE', 'ABBV',
    # Consumer
    'PG', 'KO', 'HD', 'WMT', 'DIS',
    # Energy/Industrial
    'XOM', 'CVX', 'BA', 'CAT',
    # Telecom
    'T', 'VZ'
]

# European stocks (~10 tickers): major EU companies  
EU_STOCKS = [
    'ASML.AS',    # ASML Holding (Netherlands)
    'SAP',        # SAP SE (Germany) - ADR
    'LVMH.PA',    # LVMH (France)
    'NESN.SW',    # Nestle (Switzerland)
    'NVO',        # Novo Nordisk ADR
    'ROG.SW',     # Roche (Switzerland)
    'UL',         # Unilever ADR
    'ADYEN.AS',   # Adyen (Netherlands)
    'MC.PA',      # LVMH (France)
    'OR.PA'       # L'Oreal (France)
]

# Asian stocks (~12 tickers): Japan, India, China, Korea, Taiwan
ASIA_STOCKS = [
    # Japan
    '7203.T',     # Toyota Motor
    '6758.T',     # Sony Group
    '9984.T',     # SoftBank Group
    # India
    'TCS.BO',     # Tata Consultancy Services
    'RELIANCE.BO', # Reliance Industries
    'INFY',       # Infosys ADR
    # China/Hong Kong
    '700.HK',     # Tencent Holdings
    '9988.HK',    # Alibaba Group (HK)
    'BABA',       # Alibaba ADR (backup)
    # South Korea
    '005930.KS',  # Samsung Electronics
    # Taiwan
    '2330.TW',    # Taiwan Semiconductor
    'TSM'         # Taiwan Semi ADR (backup)
]

# Commodities ETFs (~5 tickers)
COMMODITIES = [
    'GLD',        # SPDR Gold Trust
    'USO',        # United States Oil Fund
    'SLV',        # iShares Silver Trust
    'DBA',        # Invesco DB Agriculture Fund
    'DBC'         # Invesco DB Commodity Index
]

# Sector ETFs (~5 tickers)
SECTOR_ETFS = [
    'XLK',        # Technology Select Sector SPDR
    'XLF',        # Financial Select Sector SPDR
    'XLE',        # Energy Select Sector SPDR
    'XLV',        # Health Care Select Sector SPDR
    'XLI'         # Industrial Select Sector SPDR
]

# Regional ETFs (~5 tickers)
REGIONAL_ETFS = [
    'SPY',        # SPDR S&P 500 ETF
    'EFA',        # iShares MSCI EAFE ETF
    'EEM',        # iShares MSCI Emerging Markets ETF
    'IWM',        # iShares Russell 2000 ETF
    'QQQ'         # Invesco QQQ Trust ETF
]

# Combine all tickers
ALL_STOCKS = US_STOCKS + EU_STOCKS + ASIA_STOCKS
ALL_ETFS = COMMODITIES + SECTOR_ETFS + REGIONAL_ETFS
ALL_TICKERS = ALL_STOCKS + ALL_ETFS

# =============================================================================
# MACRO INDICATORS (Yahoo/FRED symbols)
# =============================================================================

MACRO_INDICATORS = {
    'vix': '^VIX',           # VIX volatility index
    'tnx10y': '^TNX',        # 10-Year Treasury yield
    'dxy': 'UUP',            # US Dollar Index (via ETF)
    'move': '^MOVE',         # MOVE bond volatility (if available)
    'irx': '^IRX',           # 13-Week Treasury yield
    'gspc': '^GSPC',         # S&P 500 index (for context)
    'ndx': '^NDX',           # NASDAQ 100 index
    'rut': '^RUT'            # Russell 2000 index
}

# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

# Moving average periods
SMA_PERIODS = [5, 10, 20, 50, 100, 200]
EMA_PERIODS = [12, 26]

# Momentum indicators
RSI_PERIOD = 14
STOCH_PERIODS = (14, 3, 3)  # %K, %D, smooth
WILLIAMS_R_PERIOD = 14
CCI_PERIOD = 20

# Volatility
ATR_PERIOD = 14
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2

# Return/volatility windows
RETURN_PERIODS = [1, 5, 10, 20, 30]
VOLATILITY_PERIODS = [5, 10, 20, 30]

# Distance/ratio periods
DISTANCE_PERIODS = [20, 252]  # 1 month, 1 year

TECHNICAL_INDICATORS = [
    # Moving averages
    'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_100', 'sma_200',
    'ema_12', 'ema_26',
    # MACD
    'macd', 'macd_signal', 'macd_hist',
    # Momentum
    'rsi_14', 'stoch_k', 'stoch_d', 'williams_r', 'cci',
    # Volatility
    'atr_14', 'bb_upper', 'bb_lower', 'bb_width',
    # Returns
    'ret_1d', 'ret_5d', 'ret_10d', 'ret_20d', 'ret_30d',
    # Volatility measures
    'vol_5d', 'vol_10d', 'vol_20d', 'vol_30d',
    # Price ratios
    'price_sma20', 'price_sma50', 'price_sma200',
    # Distance measures
    'dist_high_20d', 'dist_low_20d', 'dist_high_252d', 'dist_low_252d',
    # Volume
    'volume_sma_20'
]

# =============================================================================
# FEATURE SETS
# =============================================================================

FEATURE_SETS = {
    'PRICE_BASIC': ['Open', 'High', 'Low', 'Close', 'Volume'],
    'TECHNICAL': TECHNICAL_INDICATORS,
    'MACRO': list(MACRO_INDICATORS.keys()),
    'CROSS_SECTIONAL': [],  # Will be populated dynamically
    'TARGET': ['y_30d', 'ret_30d'],
    'MINIMAL': ['Close', 'Volume', 'sma_20', 'rsi_14', 'ret_5d']  # For quick testing
}

# =============================================================================
# MODEL PARAMETERS
# =============================================================================

MODEL_PARAMS = {
    'xgboost': {
        'n_estimators': 100,  # Reduced from 400
        'learning_rate': 0.01,
        'max_depth': 2,       # Reduced from 4
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,     # L1 regularization
        'reg_lambda': 1.0,    # L2 regularization
        'random_state': 42,
        'eval_metric': 'auc',
        'early_stopping_rounds': 30,  # Reduced from 50
        'verbosity': 1
    },
    'random_forest': {
        'n_estimators': 100,  # Reduced from 500
        'max_depth': 2,       # Added depth limit (was None)
        'min_samples_split': 20,  # Increased from 5
        'min_samples_leaf': 10,   # Increased from 2
        'max_features': 'sqrt',
        'max_samples': 0.8,   # Added bootstrap sampling limit
        'random_state': 42,
        'n_jobs': -1,
        'class_weight': 'balanced_subsample'
    },
    'logistic_regression': {
        'penalty': 'l2',
        'C': 1.0,
        'max_iter': 2000,
        'random_state': 42,
        'class_weight': 'balanced',
        'solver': 'lbfgs'
    }
}

# Hyperparameter search grids (optional)
PARAM_GRIDS = {
    'xgboost': {
        'max_depth': [3, 4, 6],
        'learning_rate': [0.03, 0.05, 0.1],
        'n_estimators': [300, 400, 500]
    },
    'logistic_regression': {
        'C': [0.1, 1.0, 10.0],
        'penalty': ['l1', 'l2']
    }
}

# =============================================================================
# MARKET METADATA
# =============================================================================

# Market classifications for analysis
TICKER_MARKETS = {}

# US stocks
for ticker in US_STOCKS:
    TICKER_MARKETS[ticker] = 'US'

# European stocks  
for ticker in EU_STOCKS:
    TICKER_MARKETS[ticker] = 'EU'

# Asian stocks
for ticker in ASIA_STOCKS:
    TICKER_MARKETS[ticker] = 'ASIA'

# ETFs by category
for ticker in COMMODITIES:
    TICKER_MARKETS[ticker] = 'COMMODITY'
    
for ticker in SECTOR_ETFS:
    TICKER_MARKETS[ticker] = 'SECTOR'
    
for ticker in REGIONAL_ETFS:
    TICKER_MARKETS[ticker] = 'REGIONAL'

# =============================================================================
# VALIDATION & EVALUATION
# =============================================================================

# Cross-validation settings
CV_FOLDS = 3
CV_METHOD = 'time_series'  # time-based splits

# Evaluation metrics
CLASSIFICATION_METRICS = ['roc_auc', 'f1', 'precision', 'recall', 'accuracy']
FINANCIAL_METRICS = ['cagr', 'sharpe', 'max_drawdown', 'sortino', 'profit_factor']

# =============================================================================
# FILE PATTERNS & FORMATS
# =============================================================================

# File naming patterns
RAW_DATA_PATTERN = "{symbol}.parquet"
PROCESSED_DATA_FILE = "dataset.parquet"
MODEL_FILE_PATTERN = "{model_name}_{timestamp}.pkl"
PREDICTIONS_FILE_PATTERN = "predictions_{model}_{timestamp}.parquet"
METRICS_FILE_PATTERN = "metrics_{model}_{timestamp}.csv"

# Date format
DATE_FORMAT = "%Y-%m-%d"
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

# =============================================================================
# LOGGING & DEBUGGING
# =============================================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Progress bar settings
PROGRESS_BAR = True
PROGRESS_UNIT = "ticker"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_ticker_market(ticker: str) -> str:
    """Get market classification for a ticker."""
    return TICKER_MARKETS.get(ticker, 'UNKNOWN')

def get_all_tickers() -> List[str]:
    """Get complete list of all tickers."""
    return ALL_TICKERS.copy()

def get_macro_symbols() -> List[str]:
    """Get list of macro indicator symbols."""
    return list(MACRO_INDICATORS.values())

def get_feature_set(name: str) -> List[str]:
    """Get feature set by name."""
    return FEATURE_SETS.get(name, [])

def get_model_params(model_name: str) -> Dict:
    """Get model parameters by name."""
    return MODEL_PARAMS.get(model_name, {})

# =============================================================================
# ENVIRONMENT VARIABLES (optional)
# =============================================================================

# For FRED API (if needed)
FRED_API_KEY = os.getenv('FRED_API_KEY', None)

# For data caching
CACHE_ENABLED = os.getenv('CACHE_ENABLED', 'true').lower() == 'true'
CACHE_EXPIRY_DAYS = int(os.getenv('CACHE_EXPIRY_DAYS', '7'))

if __name__ == "__main__":
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Total tickers: {len(ALL_TICKERS)}")
    print(f"US stocks: {len(US_STOCKS)}")
    print(f"EU stocks: {len(EU_STOCKS)}")  
    print(f"Asia stocks: {len(ASIA_STOCKS)}")
    print(f"ETFs: {len(ALL_ETFS)}")
    print(f"Macro indicators: {len(MACRO_INDICATORS)}")
    print(f"Technical indicators: {len(TECHNICAL_INDICATORS)}")