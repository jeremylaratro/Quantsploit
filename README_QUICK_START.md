# Quantsploit Quick Start Guide

## Simple Installation & Launch

### 1. Install Dependencies

Run the install script:
```bash
./install_deps.sh
```

Or install manually:
```bash
pip3 install --user yfinance pandas numpy scipy scikit-learn prompt_toolkit rich pyyaml
```

### 2. Launch Quantsploit

```bash
python3 quantsploit.py
```

That's it! No need to deal with package installation or setup.py.

---

## Available Strategies

Once Quantsploit is running, type `show modules` to see all available modules.

### New Advanced Strategies

1. **ml_swing_trading** - Machine Learning swing trading (Random Forest + XGBoost)
2. **pairs_trading** - Statistical arbitrage using cointegration
3. **kalman_adaptive** - Kalman Filter adaptive trend following
4. **hmm_regime_detection** - Market regime detection with HMM
5. **volume_profile_swing** - Volume profile based swing trading
6. **options_volatility** - Options volatility strategies (straddles, strangles)
7. **options_spreads** - Advanced spreads (Iron Condor, Butterfly, Calendar)

### Example Usage

```
quantsploit > use ml_swing_trading
quantsploit (ml_swing_trading) > set SYMBOL AAPL
quantsploit (ml_swing_trading) > run
```

---

## Troubleshooting

### Dependencies not installing?

Try installing them one at a time:
```bash
pip3 install --user pandas
pip3 install --user numpy  
pip3 install --user scipy
pip3 install --user scikit-learn
pip3 install --user yfinance
pip3 install --user prompt_toolkit
pip3 install --user rich
pip3 install --user pyyaml
```

### Module not showing up?

The framework automatically discovers all modules in `quantsploit/modules/`. If a new strategy doesn't appear, make sure:
1. All dependencies are installed
2. The strategy file is in the correct directory
3. Run `python3 quantsploit.py` fresh

---

## Development

To add new strategies, simply add a Python file to:
- `quantsploit/modules/strategies/` for trading strategies
- `quantsploit/modules/analysis/` for analysis tools
- `quantsploit/modules/scanners/` for stock scanners

The framework will automatically discover and load them!
