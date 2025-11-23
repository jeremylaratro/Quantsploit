"""
Comprehensive Ticker Universe Management for Institutional-Grade Analysis
High-level sectors and low-level niches for specialized analysis
"""

# ==================== HIGH-LEVEL SECTORS ====================

SPACE_OVERALL = ['RKLB', 'ASTS', 'SATS', 'PL', 'FLY', 'VOYG', 'LUNR', 'RDW', 'SPIR', 'SPCE']

AI_TECH_OVERALL = ['NVDA', 'MSFT', 'GOOG', 'AMZN', 'META', 'TSM', 'AVGO', 'ORCL', 'CRM', 'PLTR']

SEMICONDUCTORS = ['NVDA', 'TSM', 'ASML', 'AMD', 'AVGO', 'QCOM', 'INTC', 'MU', 'LRCX', 'AMAT']

QUANTUM_COMPUTING = ['IONQ', 'RGTI', 'QBTS', 'QUBT', 'ARQQ', 'IBM', 'GOOG', 'MSFT', 'NVDA', 'AMZN']

CYBERSECURITY = ['PANW', 'CRWD', 'FTNT', 'ZS', 'NET', 'S', 'CYBR', 'OKTA', 'RBRK', 'GEN']

CLOUD_COMPUTING = ['AMZN', 'MSFT', 'GOOG', 'ORCL', 'IBM', 'CRM', 'NOW', 'SNOW', 'DDOG', 'NET']

NUCLEAR_URANIUM = ['CEG', 'CCJ', 'NXE', 'LEU', 'UEC', 'DNN', 'UUUU', 'SMR', 'OKLO', 'GEV']

ELECTRIC_VEHICLES = ['TSLA', 'RIVN', 'LCID', 'NIO', 'XPEV', 'LI', 'GM', 'F', 'BYDDF', 'CHPT']

SOLAR_RENEWABLE = ['FSLR', 'ENPH', 'NEE', 'BEP', 'SEDG', 'RUN', 'CSIQ', 'JKS', 'AES', 'ARRY']

DEFENSE_AEROSPACE = ['LMT', 'RTX', 'NOC', 'GD', 'BA', 'LHX', 'HII', 'TDG', 'HWM', 'AVAV']

ROBOTICS_AUTOMATION = ['NVDA', 'TSLA', 'ISRG', 'ROK', 'ABB', 'TER', 'SYM', 'CGNX', 'PATH', 'IRBT']

BIOTECH = ['AMGN', 'GILD', 'VRTX', 'REGN', 'BIIB', 'MRNA', 'BNTX', 'CRSP', 'EXEL', 'BMRN']

FINTECH = ['V', 'MA', 'PYPL', 'SQ', 'AFRM', 'SOFI', 'NU', 'HOOD', 'INTU', 'FIS']

DATA_CENTERS = ['EQIX', 'DLR', 'AMT', 'MSFT', 'AMZN', 'GOOG', 'ORCL', 'CCI', 'IRM', 'SBAC']

# ==================== LOW-LEVEL NICHES - SPACE ====================

SPACE_LAUNCH_SERVICES = ['RKLB', 'FLY', 'AJRD', 'SPCE', 'LLAP', 'LUNR', 'MAXR', 'VORB', 'RDW', 'MNTS']

SPACE_SATELLITE_COMMS = ['ASTS', 'SATS', 'TSAT', 'GSAT', 'IRDM', 'VSAT', 'GILT', 'DISH', 'OSAT', 'SPIR']

# ==================== LOW-LEVEL NICHES - AI ====================

AI_INFRASTRUCTURE = ['NVDA', 'TSM', 'ASML', 'AMD', 'AVGO', 'MRVL', 'PLTR', 'SMCI', 'ORCL', 'CRM']

AI_SOFTWARE = ['PLTR', 'AI', 'PATH', 'SNOW', 'DDOG', 'MDB', 'NOW', 'ADBE', 'SOUN', 'BBAI']

# ==================== LOW-LEVEL NICHES - SEMICONDUCTORS ====================

SEMIS_GPU_AI_CHIPS = ['NVDA', 'AMD', 'INTC', 'AVGO', 'QCOM', 'MRVL', 'ARM', 'TSM', 'MU', 'LRCX']

SEMIS_EQUIPMENT = ['ASML', 'AMAT', 'LRCX', 'KLAC', 'ENTG', 'ACMR', 'TER', 'ONTO', 'UCTT', 'ICHR']

SEMIS_MEMORY = ['MU', 'WDC', 'STX', 'RMBS', 'NAND', 'MCHP', 'ON', 'SWKS', 'WOLF', 'CRUS']

# ==================== LOW-LEVEL NICHES - NUCLEAR ====================

URANIUM_MINING = ['CCJ', 'NXE', 'UEC', 'DNN', 'UUUU', 'URG', 'EU', 'FCUUF', 'SRUUF', 'URNM']

SMALL_MODULAR_REACTORS = ['SMR', 'OKLO', 'NNE', 'GEV', 'BWX', 'FLR', 'CEG', 'PCG', 'ETR', 'D']

# ==================== LOW-LEVEL NICHES - ELECTRIC VEHICLES ====================

EV_MANUFACTURERS = ['TSLA', 'RIVN', 'LCID', 'NIO', 'XPEV', 'LI', 'GM', 'F', 'BYDDF', 'FSRE']

EV_CHARGING = ['CHPT', 'EVGO', 'BLNK', 'WBX', 'TSLA', 'ABB', 'SHEL', 'BP', 'PCAR', 'CMI']

EV_BATTERIES = ['QS', 'ENVX', 'SLDP', 'ALB', 'LTHM', 'SQM', 'LAC', 'PLL', 'MVST', 'FREEF']

# ==================== LOW-LEVEL NICHES - CLOUD ====================

CLOUD_IAAS = ['AMZN', 'MSFT', 'GOOG', 'ORCL', 'IBM', 'CRM', 'NOW', 'SNOW', 'DDOG', 'NET']

CLOUD_SAAS = ['CRM', 'NOW', 'WDAY', 'ZM', 'TEAM', 'HUBS', 'DOCU', 'OKTA', 'ZS', 'SHOP']

DATA_CENTER_REITS = ['EQIX', 'DLR', 'AMT', 'CCI', 'SBAC', 'IRM', 'QTS', 'CONE', 'COR', 'UNIT']

# ==================== LOW-LEVEL NICHES - DEFENSE ====================

DEFENSE_LARGE_CAP = ['LMT', 'RTX', 'NOC', 'GD', 'BA', 'LHX', 'HII', 'TDG', 'HWM', 'AVAV']

DEFENSE_DRONES = ['AVAV', 'KTOS', 'RCAT', 'JOBY', 'ACHR', 'UAVS', 'UMAC', 'GILT', 'TXT', 'LMT']

# ==================== LOW-LEVEL NICHES - ROBOTICS ====================

ROBOTICS_INDUSTRIAL = ['ABB', 'FANUY', 'ROK', 'EMR', 'HON', 'ISRG', 'TER', 'CGNX', 'GNRC', 'IRBT']

ROBOTICS_SURGICAL = ['ISRG', 'MDT', 'ABT', 'SYK', 'BSX', 'EW', 'GMED', 'NUVA', 'MASI', 'HOLX']

ROBOTICS_WAREHOUSE = ['SYM', 'AMZN', 'GXO', 'XPO', 'EXPD', 'CHRW', 'FWRD', 'JBHT', 'KNX', 'LSTR']

# ==================== LOW-LEVEL NICHES - RENEWABLE ENERGY ====================

SOLAR_ENERGY = ['FSLR', 'ENPH', 'SEDG', 'RUN', 'NOVA', 'CSIQ', 'JKS', 'ARRY', 'MAXN', 'SPWR']

WIND_ENERGY = ['NEE', 'GEV', 'VWDRY', 'BEP', 'CWEN', 'ORSTED', 'IBDRY', 'TKA', 'EDPR', 'ORA']

RENEWABLE_UTILITIES = ['NEE', 'BEP', 'BEPC', 'AES', 'D', 'DUK', 'SO', 'XEL', 'ED', 'AEP']

# ==================== LOW-LEVEL NICHES - BIOTECH ====================

BIOTECH_LARGE_CAP = ['AMGN', 'GILD', 'VRTX', 'REGN', 'BIIB', 'MRNA', 'BNTX', 'ABBV', 'BMY', 'LLY']

BIOTECH_GENE_THERAPY = ['CRSP', 'NTLA', 'EDIT', 'BEAM', 'VERV', 'SGMO', 'BLUE', 'RARE', 'RGNX', 'BMRN']

BIOTECH_ONCOLOGY = ['MRNA', 'BNTX', 'EXEL', 'SGEN', 'IMMU', 'MRUS', 'MRTX', 'RCUS', 'IOVA', 'KYMR']

# ==================== LOW-LEVEL NICHES - FINTECH ====================

FINTECH_PAYMENTS = ['V', 'MA', 'PYPL', 'SQ', 'AFRM', 'UPST', 'SOFI', 'NU', 'INTU', 'FIS']

FINTECH_DIGITAL_BANKING = ['SOFI', 'NU', 'HOOD', 'LC', 'DAVE', 'CHIME', 'ALLY', 'NYCB', 'WAL', 'ZION']

FINTECH_BNPL = ['AFRM', 'SEZL', 'ZIP', 'PYPL', 'AAPL', 'KLRN', 'UPST', 'SOFI', 'NU', 'SQ']

# ==================== CLASSIC INDEX UNIVERSES ====================

# S&P 500 Top Holdings by Sector (from original file)
SP500_TECHNOLOGY = [
    'AAPL', 'MSFT', 'NVDA', 'AVGO', 'CRM', 'ORCL', 'ADBE', 'CSCO', 'ACN', 'AMD',
    'INTC', 'TXN', 'QCOM', 'AMAT', 'MU', 'LRCX', 'KLAC', 'SNPS', 'CDNS', 'MCHP'
]

MEGA_CAP = ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B', 'LLY']

# ==================== UNIVERSE MAPPINGS ====================

HIGH_LEVEL_SECTORS = {
    'Space (Overall)': SPACE_OVERALL,
    'AI/Tech (Overall)': AI_TECH_OVERALL,
    'Semiconductors': SEMICONDUCTORS,
    'Quantum Computing': QUANTUM_COMPUTING,
    'Cybersecurity': CYBERSECURITY,
    'Cloud Computing': CLOUD_COMPUTING,
    'Nuclear/Uranium': NUCLEAR_URANIUM,
    'Electric Vehicles': ELECTRIC_VEHICLES,
    'Solar/Renewable': SOLAR_RENEWABLE,
    'Defense/Aerospace': DEFENSE_AEROSPACE,
    'Robotics/Automation': ROBOTICS_AUTOMATION,
    'Biotech': BIOTECH,
    'Fintech': FINTECH,
    'Data Centers': DATA_CENTERS,
}

NICHE_SECTORS = {
    # Space Niches
    'Space - Launch Services': SPACE_LAUNCH_SERVICES,
    'Space - Satellite Communications': SPACE_SATELLITE_COMMS,

    # AI Niches
    'AI - Infrastructure': AI_INFRASTRUCTURE,
    'AI - Software/Applications': AI_SOFTWARE,

    # Semiconductor Niches
    'Semiconductors - GPUs/AI Chips': SEMIS_GPU_AI_CHIPS,
    'Semiconductors - Equipment': SEMIS_EQUIPMENT,
    'Semiconductors - Memory': SEMIS_MEMORY,

    # Nuclear Niches
    'Nuclear - Uranium Mining': URANIUM_MINING,
    'Nuclear - Small Modular Reactors': SMALL_MODULAR_REACTORS,

    # EV Niches
    'EV - Manufacturers': EV_MANUFACTURERS,
    'EV - Charging Infrastructure': EV_CHARGING,
    'EV - Batteries': EV_BATTERIES,

    # Cloud Niches
    'Cloud - IaaS Infrastructure': CLOUD_IAAS,
    'Cloud - SaaS': CLOUD_SAAS,
    'Cloud - Data Center REITs': DATA_CENTER_REITS,

    # Defense Niches
    'Defense - Large Cap': DEFENSE_LARGE_CAP,
    'Defense - Drones/UAVs': DEFENSE_DRONES,

    # Robotics Niches
    'Robotics - Industrial': ROBOTICS_INDUSTRIAL,
    'Robotics - Surgical/Medical': ROBOTICS_SURGICAL,
    'Robotics - Warehouse/Logistics': ROBOTICS_WAREHOUSE,

    # Renewable Niches
    'Solar Energy': SOLAR_ENERGY,
    'Wind Energy': WIND_ENERGY,
    'Renewable Utilities': RENEWABLE_UTILITIES,

    # Biotech Niches
    'Biotech - Large Cap': BIOTECH_LARGE_CAP,
    'Biotech - Gene Therapy': BIOTECH_GENE_THERAPY,
    'Biotech - Oncology': BIOTECH_ONCOLOGY,

    # Fintech Niches
    'Fintech - Payments': FINTECH_PAYMENTS,
    'Fintech - Digital Banking': FINTECH_DIGITAL_BANKING,
    'Fintech - BNPL': FINTECH_BNPL,
}

# Classic Index Universes
CLASSIC_UNIVERSES = {
    'Mega Cap': MEGA_CAP,
    'S&P 500 Technology': SP500_TECHNOLOGY,
}

# Combine all sectors
ALL_SECTORS = {**HIGH_LEVEL_SECTORS, **NICHE_SECTORS}

# ==================== HELPER FUNCTIONS ====================

def _normalize_name(name):
    """Normalize a display name to a lookup key"""
    return name.lower().replace(' ', '_').replace('/', '_').replace('-', '_').replace('(', '').replace(')', '')

def _create_name_mapping():
    """Create a mapping from normalized names to display names"""
    mapping = {}
    for display_name in HIGH_LEVEL_SECTORS.keys():
        mapping[_normalize_name(display_name)] = display_name
    for display_name in NICHE_SECTORS.keys():
        mapping[_normalize_name(display_name)] = display_name
    for display_name in CLASSIC_UNIVERSES.keys():
        mapping[_normalize_name(display_name)] = display_name
    return mapping

# Create the reverse mapping
_NAME_MAPPING = _create_name_mapping()

def get_universe(name):
    """Get ticker list by universe name (accepts both display names and normalized keys)"""
    # First try direct lookup
    tickers = ALL_SECTORS.get(name, CLASSIC_UNIVERSES.get(name, None))
    if tickers is not None:
        return tickers

    # Try normalized name lookup
    normalized = _normalize_name(name)
    display_name = _NAME_MAPPING.get(normalized)
    if display_name:
        return ALL_SECTORS.get(display_name, CLASSIC_UNIVERSES.get(display_name, []))

    return []

def get_all_universes():
    """Get dictionary of all available universes organized by category"""
    result = {}

    # Add high-level sectors
    for name in HIGH_LEVEL_SECTORS.keys():
        result[name] = _normalize_name(name)

    # Add niche sectors
    for name in NICHE_SECTORS.keys():
        result[name] = _normalize_name(name)

    # Add classic universes
    for name in CLASSIC_UNIVERSES.keys():
        result[name] = _normalize_name(name)

    return result

def get_all_sectors():
    """Get list of all sector names"""
    return list(HIGH_LEVEL_SECTORS.keys()) + list(NICHE_SECTORS.keys())

def get_sector_tickers(sector):
    """Get all tickers in a sector (accepts both display names and normalized keys)"""
    # First try direct lookup
    tickers = ALL_SECTORS.get(sector, None)
    if tickers is not None:
        return tickers

    # Try normalized name lookup
    normalized = _normalize_name(sector)
    display_name = _NAME_MAPPING.get(normalized)
    if display_name:
        return ALL_SECTORS.get(display_name, [])

    return []

def get_sector(ticker):
    """Get sector classification for a ticker"""
    for sector, tickers in ALL_SECTORS.items():
        if ticker in tickers:
            return sector
    return 'Unknown'

def get_market_cap_class(ticker):
    """Classify ticker by market cap"""
    if ticker in MEGA_CAP:
        return 'Mega Cap'
    return 'Unknown'
