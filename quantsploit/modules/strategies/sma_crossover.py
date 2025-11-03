"""
Simple Moving Average Crossover Strategy
"""

import pandas as pd
from quantsploit.utils.ta_compat import ta
from typing import Dict, Any
from quantsploit.core.module import BaseModule
from quantsploit.utils.data_fetcher import DataFetcher


class SMACrossover(BaseModule):
    """
    Simple Moving Average Crossover backtesting strategy
    """

    @property
    def name(self) -> str:
        return "SMA Crossover Strategy"

    @property
    def description(self) -> str:
        return "Backtest a simple moving average crossover strategy"

    @property
    def author(self) -> str:
        return "Quantsploit Team"

    @property
    def category(self) -> str:
        return "strategy"

    def _init_options(self):
        super()._init_options()
        self.options.update({
            "FAST_PERIOD": {
                "value": 10,
                "required": False,
                "description": "Fast SMA period"
            },
            "SLOW_PERIOD": {
                "value": 30,
                "required": False,
                "description": "Slow SMA period"
            },
            "INITIAL_CAPITAL": {
                "value": 10000,
                "required": False,
                "description": "Initial capital for backtest"
            },
        })

    def run(self) -> Dict[str, Any]:
        """Execute strategy backtest"""
        symbol = self.get_option("SYMBOL")
        period = self.get_option("PERIOD")
        interval = self.get_option("INTERVAL")
        fast_period = int(self.get_option("FAST_PERIOD"))
        slow_period = int(self.get_option("SLOW_PERIOD"))
        initial_capital = float(self.get_option("INITIAL_CAPITAL"))

        # Fetch data
        fetcher = DataFetcher(self.framework.database)
        df = fetcher.get_stock_data(symbol, period, interval)

        if df is None or df.empty:
            return {"success": False, "error": f"Failed to fetch data for {symbol}"}

        # Calculate SMAs
        df['SMA_fast'] = ta.sma(df['Close'], length=fast_period)
        df['SMA_slow'] = ta.sma(df['Close'], length=slow_period)

        # Generate signals
        df['signal'] = 0
        df.loc[df['SMA_fast'] > df['SMA_slow'], 'signal'] = 1  # Buy signal
        df.loc[df['SMA_fast'] < df['SMA_slow'], 'signal'] = -1  # Sell signal

        # Detect crossovers
        df['position'] = df['signal'].diff()

        # Backtest
        capital = initial_capital
        position = 0
        shares = 0
        trades = []

        for idx, row in df.iterrows():
            if pd.isna(row['position']):
                continue

            # Buy signal
            if row['position'] == 2 and position == 0:  # Changed from -1 to 1 (crossover up)
                shares = capital / row['Close']
                position = 1
                trades.append({
                    'date': idx,
                    'action': 'BUY',
                    'price': row['Close'],
                    'shares': shares,
                    'value': capital
                })

            # Sell signal
            elif row['position'] == -2 and position == 1:  # Changed from 1 to -1 (crossover down)
                capital = shares * row['Close']
                position = 0
                trades.append({
                    'date': idx,
                    'action': 'SELL',
                    'price': row['Close'],
                    'shares': shares,
                    'value': capital
                })
                shares = 0

        # Calculate final value
        if position == 1:
            final_value = shares * df['Close'].iloc[-1]
        else:
            final_value = capital

        # Calculate metrics
        total_return = final_value - initial_capital
        total_return_pct = (total_return / initial_capital) * 100

        # Buy and hold comparison
        buy_hold_shares = initial_capital / df['Close'].iloc[0]
        buy_hold_final = buy_hold_shares * df['Close'].iloc[-1]
        buy_hold_return = buy_hold_final - initial_capital
        buy_hold_return_pct = (buy_hold_return / initial_capital) * 100

        results = {
            "symbol": symbol,
            "period": period,
            "fast_sma": fast_period,
            "slow_sma": slow_period,
            "initial_capital": initial_capital,
            "final_value": final_value,
            "total_return": total_return,
            "total_return_pct": total_return_pct,
            "buy_hold_return": buy_hold_return,
            "buy_hold_return_pct": buy_hold_return_pct,
            "strategy_vs_buy_hold": total_return_pct - buy_hold_return_pct,
            "total_trades": len(trades),
            "trades": pd.DataFrame(trades) if trades else pd.DataFrame(),
            "price_chart_data": df[['Close', 'SMA_fast', 'SMA_slow']].tail(100)
        }

        return results
