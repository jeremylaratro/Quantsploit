"""
Ticker Universe Management for Institutional-Grade Analysis
Provides sector-classified ticker lists for major indices
"""

# S&P 500 Top Holdings by Sector (Representative sample - can be expanded)
SP500_TECHNOLOGY = [
    'AAPL', 'MSFT', 'NVDA', 'AVGO', 'CRM', 'ORCL', 'ADBE', 'CSCO', 'ACN', 'AMD',
    'INTC', 'TXN', 'QCOM', 'AMAT', 'MU', 'LRCX', 'KLAC', 'SNPS', 'CDNS', 'MCHP'
]

SP500_COMMUNICATION = [
    'META', 'GOOGL', 'GOOG', 'NFLX', 'DIS', 'CMCSA', 'VZ', 'T', 'TMUS', 'CHTR',
    'EA', 'TTWO', 'NWSA', 'FOX', 'PARA', 'OMC', 'IPG', 'WBD'
]

SP500_CONSUMER_DISCRETIONARY = [
    'AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'LOW', 'SBUX', 'TJX', 'BKNG', 'CMG',
    'MAR', 'ABNB', 'GM', 'F', 'HLT', 'ORLY', 'AZO', 'YUM', 'RCL', 'CCL'
]

SP500_CONSUMER_STAPLES = [
    'WMT', 'PG', 'COST', 'KO', 'PEP', 'PM', 'MO', 'MDLZ', 'CL', 'KMB',
    'GIS', 'KHC', 'STZ', 'SYY', 'HSY', 'K', 'CAG', 'CPB', 'TSN', 'HRL'
]

SP500_HEALTHCARE = [
    'UNH', 'LLY', 'JNJ', 'ABBV', 'MRK', 'TMO', 'ABT', 'AMGN', 'DHR', 'PFE',
    'BMY', 'CVS', 'GILD', 'CI', 'VRTX', 'REGN', 'ELV', 'MCK', 'HUM', 'ZTS'
]

SP500_FINANCIALS = [
    'BRK.B', 'JPM', 'V', 'MA', 'BAC', 'WFC', 'MS', 'GS', 'SPGI', 'BLK',
    'C', 'AXP', 'PGR', 'CB', 'MMC', 'ICE', 'CME', 'AON', 'PNC', 'USB'
]

SP500_INDUSTRIALS = [
    'GE', 'CAT', 'RTX', 'HON', 'UNP', 'BA', 'LMT', 'DE', 'UPS', 'ADP',
    'GD', 'NOC', 'EMR', 'ETN', 'ITW', 'MMM', 'CSX', 'NSC', 'WM', 'FDX'
]

SP500_ENERGY = [
    'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'PSX', 'VLO', 'OXY', 'WMB',
    'KMI', 'HES', 'DVN', 'HAL', 'BKR', 'FANG', 'MRO', 'APA', 'CTRA', 'OKE'
]

SP500_MATERIALS = [
    'LIN', 'APD', 'SHW', 'ECL', 'FCX', 'NEM', 'CTVA', 'DOW', 'DD', 'NUE',
    'VMC', 'MLM', 'PPG', 'ALB', 'BALL', 'AVY', 'CF', 'MOS', 'FMC', 'CE'
]

SP500_REAL_ESTATE = [
    'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'WELL', 'DLR', 'O', 'SPG', 'VICI',
    'AVB', 'EQR', 'WY', 'SBAC', 'VTR', 'ARE', 'INVH', 'ESS', 'MAA', 'KIM'
]

SP500_UTILITIES = [
    'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'PCG', 'ED',
    'WEC', 'PEG', 'ES', 'AWK', 'DTE', 'ETR', 'FE', 'PPL', 'AEE', 'CMS'
]

# NASDAQ 100 (Tech-heavy)
NASDAQ_100 = [
    # Mega Cap Tech
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'ORCL',
    # Large Cap Tech
    'COST', 'NFLX', 'AMD', 'ADBE', 'PEP', 'CSCO', 'TMUS', 'QCOM', 'INTC', 'TXN',
    'CMCSA', 'AMAT', 'HON', 'INTU', 'AMGN', 'BKNG', 'ISRG', 'ADP', 'SBUX', 'GILD',
    # Mid Cap Tech & Growth
    'ADI', 'VRTX', 'REGN', 'LRCX', 'MU', 'PANW', 'KLAC', 'SNPS', 'CDNS', 'MELI',
    'PYPL', 'ABNB', 'ASML', 'MAR', 'NXPI', 'CRWD', 'MRVL', 'WDAY', 'FTNT', 'DASH',
    # Additional Growth Names
    'DXCM', 'TEAM', 'ADSK', 'MNST', 'CHTR', 'PCAR', 'AEP', 'MCHP', 'ROST', 'PAYX',
    'ODFL', 'CPRT', 'CTAS', 'LULU', 'MRNA', 'CSX', 'EA', 'ORLY', 'IDXX', 'FAST',
    'TTWO', 'VRSK', 'DDOG', 'ANSS', 'KDP', 'ZS', 'BIIB', 'EXC', 'GEHC', 'ON',
    'FANG', 'CSGP', 'XEL', 'BKR', 'CDW', 'ILMN', 'GFS', 'WBD', 'CTSH', 'MDB',
    'ZM', 'ALGN', 'DLTR', 'ENPH', 'SIRI', 'JD', 'WBA', 'LCID', 'RIVN', 'SGEN'
]

# Russell 2000 Representatives (Small Cap)
RUSSELL_2000_SAMPLE = [
    'SAIA', 'ABCB', 'APAM', 'ASPN', 'AWR', 'BANR', 'BCPC', 'BL', 'CADE', 'CBU',
    'CEIX', 'CENTA', 'CHCT', 'CTRE', 'CWT', 'DNLI', 'EXPO', 'FFBC', 'FHB', 'FIBK',
    'FORM', 'GATX', 'GMS', 'HASI', 'HBI', 'HQY', 'ICFI', 'IPAR', 'ITGR', 'KRG'
]

# Major ETFs for Benchmarking
BENCHMARK_ETFS = [
    'SPY',   # S&P 500
    'QQQ',   # NASDAQ 100
    'IWM',   # Russell 2000
    'DIA',   # Dow Jones
    'VTI',   # Total Stock Market
    'VOO',   # Vanguard S&P 500
    'VUG',   # Growth
    'VTV',   # Value
    'VB',    # Small Cap
    'VO',    # Mid Cap
    'VEA',   # International Developed
    'VWO',   # Emerging Markets
    'AGG',   # Bond Aggregate
    'TLT',   # Long-Term Treasury
    'GLD',   # Gold
]

# Sector ETFs
SECTOR_ETFS = {
    'Technology': 'XLK',
    'Healthcare': 'XLV',
    'Financials': 'XLF',
    'Consumer Discretionary': 'XLY',
    'Communication': 'XLC',
    'Industrials': 'XLI',
    'Consumer Staples': 'XLP',
    'Energy': 'XLE',
    'Utilities': 'XLU',
    'Real Estate': 'XLRE',
    'Materials': 'XLB'
}

# Complete S&P 500 by Sector
SP500_BY_SECTOR = {
    'Technology': SP500_TECHNOLOGY,
    'Communication Services': SP500_COMMUNICATION,
    'Consumer Discretionary': SP500_CONSUMER_DISCRETIONARY,
    'Consumer Staples': SP500_CONSUMER_STAPLES,
    'Healthcare': SP500_HEALTHCARE,
    'Financials': SP500_FINANCIALS,
    'Industrials': SP500_INDUSTRIALS,
    'Energy': SP500_ENERGY,
    'Materials': SP500_MATERIALS,
    'Real Estate': SP500_REAL_ESTATE,
    'Utilities': SP500_UTILITIES
}

# Complete ticker lists
SP500_ALL = []
for sector_tickers in SP500_BY_SECTOR.values():
    SP500_ALL.extend(sector_tickers)

# Market Cap Classifications
MEGA_CAP = ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B', 'LLY']
LARGE_CAP = SP500_ALL
SMALL_CAP = RUSSELL_2000_SAMPLE

# Volatility Classifications
HIGH_VOLATILITY = ['TSLA', 'NVDA', 'AMD', 'MARA', 'RIOT', 'COIN', 'ARKK', 'SQQQ', 'TQQQ']
LOW_VOLATILITY = ['JNJ', 'PG', 'KO', 'WMT', 'VZ', 'T', 'DUK', 'SO', 'NEE', 'AEP']

# Universe Definitions
UNIVERSES = {
    'sp500': SP500_ALL,
    'nasdaq100': NASDAQ_100,
    'russell2000': RUSSELL_2000_SAMPLE,
    'mega_cap': MEGA_CAP,
    'benchmarks': BENCHMARK_ETFS,
    'tech': SP500_TECHNOLOGY,
    'healthcare': SP500_HEALTHCARE,
    'financials': SP500_FINANCIALS,
    'energy': SP500_ENERGY,
    'consumer': SP500_CONSUMER_DISCRETIONARY + SP500_CONSUMER_STAPLES,
    'high_vol': HIGH_VOLATILITY,
    'low_vol': LOW_VOLATILITY,
}


def get_universe(name):
    """Get ticker list by universe name"""
    return UNIVERSES.get(name.lower(), [])


def get_sector(ticker):
    """Get sector classification for a ticker"""
    for sector, tickers in SP500_BY_SECTOR.items():
        if ticker in tickers:
            return sector
    if ticker in NASDAQ_100:
        return 'Technology'  # Default for NASDAQ
    return 'Unknown'


def get_all_sectors():
    """Get list of all sectors"""
    return list(SP500_BY_SECTOR.keys())


def get_sector_tickers(sector):
    """Get all tickers in a sector"""
    return SP500_BY_SECTOR.get(sector, [])


def get_market_cap_class(ticker):
    """Classify ticker by market cap"""
    if ticker in MEGA_CAP:
        return 'Mega Cap'
    elif ticker in SP500_ALL:
        return 'Large Cap'
    elif ticker in RUSSELL_2000_SAMPLE:
        return 'Small Cap'
    else:
        return 'Unknown'


def get_volatility_class(ticker):
    """Classify ticker by volatility"""
    if ticker in HIGH_VOLATILITY:
        return 'High'
    elif ticker in LOW_VOLATILITY:
        return 'Low'
    else:
        return 'Medium'


def get_all_universes():
    """Get dictionary of all available universes"""
    return {
        'S&P 500': 'sp500',
        'NASDAQ 100': 'nasdaq100',
        'Russell 2000': 'russell2000',
        'Mega Cap': 'mega_cap',
        'Benchmarks/ETFs': 'benchmarks',
        'Technology': 'tech',
        'Healthcare': 'healthcare',
        'Financials': 'financials',
        'Energy': 'energy',
        'Consumer': 'consumer',
        'High Volatility': 'high_vol',
        'Low Volatility': 'low_vol',
    }
