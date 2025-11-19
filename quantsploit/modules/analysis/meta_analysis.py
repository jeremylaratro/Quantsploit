"""
Meta-Analysis Module

This module runs multiple trading strategies on one or more symbols and correlates
their signals to identify stocks with the most consistent buy/sell signals across
different strategies.

Author: Claude AI
"""

from typing import Dict, Any, List, Tuple
import importlib
import inspect
import os
from datetime import datetime
from collections import defaultdict

from quantsploit.core.module import BaseModule


class MetaAnalysis(BaseModule):
    """
    Meta-Analysis: Runs all available strategies and correlates their signals
    to find stocks with the most consistent buy/sell signals.
    """

    @property
    def name(self) -> str:
        return "Meta-Analysis Strategy Correlation"

    @property
    def description(self) -> str:
        return "Runs all strategies and correlates signals to find stocks with consistent signals"

    @property
    def author(self) -> str:
        return "Claude AI"

    @property
    def category(self) -> str:
        return "analysis"

    def _init_options(self):
        """Initialize module options."""
        super()._init_options()
        self.options.update({
            "SYMBOLS": {
                "value": "",
                "required": True,
                "description": "Comma-separated list of stock symbols to analyze"
            },
            "PERIOD": {
                "value": "1y",
                "required": False,
                "description": "Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)"
            },
            "INTERVAL": {
                "value": "1d",
                "required": False,
                "description": "Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)"
            },
            "MIN_CONSENSUS": {
                "value": 60,
                "required": False,
                "description": "Minimum consensus percentage to highlight (0-100)"
            },
            "STRATEGIES": {
                "value": "all",
                "required": False,
                "description": "Comma-separated list of strategies to run, or 'all' for all strategies"
            }
        })

    def _load_available_strategies(self) -> Dict[str, Any]:
        """
        Load all available strategy modules from the strategies directory.

        Returns:
            Dictionary mapping strategy name to strategy class
        """
        strategies = {}
        strategies_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'strategies'
        )

        # List of strategies to load
        strategy_files = [
            'sma_crossover',
            'mean_reversion',
            'momentum_signals',
            'ml_swing_trading',
            'multifactor_scoring',
            'volume_profile_swing',
            'kalman_adaptive',
            'hmm_regime_detection'
        ]

        for strategy_file in strategy_files:
            try:
                module_path = f'quantsploit.modules.strategies.{strategy_file}'
                module = importlib.import_module(module_path)

                # Find the strategy class (subclass of BaseModule)
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(obj, BaseModule) and
                        obj != BaseModule and
                        obj.__module__ == module_path):
                        strategies[strategy_file] = obj
                        break
            except Exception as e:
                print(f"[!] Could not load strategy {strategy_file}: {str(e)}")
                continue

        return strategies

    def _extract_signal_from_results(self, results: Dict[str, Any], strategy_name: str) -> Tuple[float, str, float]:
        """
        Extract a normalized signal (-100 to +100) from strategy results.

        Args:
            results: Strategy results dictionary
            strategy_name: Name of the strategy

        Returns:
            Tuple of (signal_strength, signal_type, confidence)
            signal_strength: -100 (strong sell) to +100 (strong buy)
            signal_type: 'BUY', 'SELL', 'HOLD', or 'NEUTRAL'
            confidence: 0-100 confidence level
        """
        if not results or not results.get('success', False):
            return 0.0, 'NEUTRAL', 0.0

        signal_strength = 0.0
        signal_type = 'NEUTRAL'
        confidence = 50.0

        # Extract signals based on strategy type

        # Check for direct signal field
        if 'signal' in results:
            signal = results['signal']
            if isinstance(signal, str):
                if 'BUY' in signal.upper():
                    signal_strength = 75.0
                    signal_type = 'BUY'
                elif 'SELL' in signal.upper():
                    signal_strength = -75.0
                    signal_type = 'SELL'
                else:
                    signal_type = 'HOLD'
            elif isinstance(signal, (int, float)):
                signal_strength = float(signal)
                if signal_strength > 50:
                    signal_type = 'BUY'
                elif signal_strength < -50:
                    signal_type = 'SELL'
                else:
                    signal_type = 'HOLD'

        # Check for signal_strength field
        if 'signal_strength' in results:
            signal_strength = float(results.get('signal_strength', 0))
            if signal_strength > 50:
                signal_type = 'BUY'
            elif signal_strength < -50:
                signal_type = 'SELL'
            else:
                signal_type = 'HOLD'

        # Check for recommendation
        if 'recommendation' in results:
            rec = results['recommendation'].upper()
            if 'BUY' in rec or 'LONG' in rec:
                if signal_strength == 0:
                    signal_strength = 70.0
                signal_type = 'BUY'
            elif 'SELL' in rec or 'SHORT' in rec:
                if signal_strength == 0:
                    signal_strength = -70.0
                signal_type = 'SELL'

        # Check for trades (backtesting results)
        if 'trades' in results and results['trades']:
            # Last trade signal
            last_trade = results['trades'][-1] if results['trades'] else None
            if last_trade:
                if last_trade.get('type') == 'BUY' or last_trade.get('side') == 'BUY':
                    signal_strength = 60.0
                    signal_type = 'BUY'
                elif last_trade.get('type') == 'SELL' or last_trade.get('side') == 'SELL':
                    signal_strength = -60.0
                    signal_type = 'SELL'

        # Check for metrics that indicate performance
        if 'metrics' in results:
            metrics = results['metrics']

            # Use win rate as confidence
            if 'win_rate' in metrics:
                confidence = float(metrics['win_rate'])

            # Use Sharpe ratio to adjust signal
            if 'sharpe_ratio' in metrics:
                sharpe = float(metrics.get('sharpe_ratio', 0))
                if sharpe > 1.5:
                    signal_strength *= 1.2  # Boost signal for good Sharpe
                elif sharpe < 0:
                    signal_strength *= 0.5  # Reduce signal for negative Sharpe

        # Check for confidence field
        if 'confidence' in results:
            confidence = float(results['confidence'])

        # Check for score (multifactor scoring)
        if 'composite_score' in results:
            score = float(results['composite_score'])
            signal_strength = (score - 50) * 2  # Convert 0-100 to -100 to +100
            if score > 60:
                signal_type = 'BUY'
            elif score < 40:
                signal_type = 'SELL'
            else:
                signal_type = 'HOLD'
            confidence = abs(score - 50) * 2

        # Clamp values
        signal_strength = max(-100, min(100, signal_strength))
        confidence = max(0, min(100, confidence))

        return signal_strength, signal_type, confidence

    def _calculate_consensus(self, signals: List[Tuple[str, float, str, float]]) -> Dict[str, Any]:
        """
        Calculate consensus metrics from multiple strategy signals.

        Args:
            signals: List of (strategy_name, signal_strength, signal_type, confidence)

        Returns:
            Dictionary with consensus metrics
        """
        if not signals:
            return {
                'consensus_signal': 'NEUTRAL',
                'consensus_strength': 0.0,
                'consensus_confidence': 0.0,
                'agreement_pct': 0.0,
                'buy_count': 0,
                'sell_count': 0,
                'hold_count': 0,
                'total_strategies': 0
            }

        buy_count = sum(1 for s in signals if s[2] == 'BUY')
        sell_count = sum(1 for s in signals if s[2] == 'SELL')
        hold_count = sum(1 for s in signals if s[2] in ['HOLD', 'NEUTRAL'])
        total = len(signals)

        # Weighted average signal strength (weighted by confidence)
        total_weighted_signal = sum(s[1] * s[3] for s in signals)
        total_confidence = sum(s[3] for s in signals)

        if total_confidence > 0:
            consensus_strength = total_weighted_signal / total_confidence
        else:
            consensus_strength = sum(s[1] for s in signals) / total if total > 0 else 0

        # Determine consensus signal
        if buy_count > sell_count and buy_count > hold_count:
            consensus_signal = 'BUY'
            agreement_pct = (buy_count / total) * 100
        elif sell_count > buy_count and sell_count > hold_count:
            consensus_signal = 'SELL'
            agreement_pct = (sell_count / total) * 100
        else:
            consensus_signal = 'HOLD'
            agreement_pct = (hold_count / total) * 100 if hold_count > 0 else 0

        # Average confidence
        consensus_confidence = sum(s[3] for s in signals) / total if total > 0 else 0

        return {
            'consensus_signal': consensus_signal,
            'consensus_strength': consensus_strength,
            'consensus_confidence': consensus_confidence,
            'agreement_pct': agreement_pct,
            'buy_count': buy_count,
            'sell_count': sell_count,
            'hold_count': hold_count,
            'total_strategies': total
        }

    def run(self) -> Dict[str, Any]:
        """
        Execute meta-analysis across multiple strategies and symbols.

        Returns:
            Dictionary containing analysis results
        """
        print("\n[*] Starting Meta-Analysis...")

        # Get options
        symbols_input = self.get_option("SYMBOLS")
        period = self.get_option("PERIOD")
        interval = self.get_option("INTERVAL")
        min_consensus = float(self.get_option("MIN_CONSENSUS"))
        strategies_filter = self.get_option("STRATEGIES")

        # Parse symbols
        symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]

        if not symbols:
            return {
                "success": False,
                "error": "No symbols provided"
            }

        print(f"[*] Analyzing {len(symbols)} symbol(s): {', '.join(symbols)}")

        # Load available strategies
        available_strategies = self._load_available_strategies()

        if not available_strategies:
            return {
                "success": False,
                "error": "No strategies could be loaded"
            }

        # Filter strategies if requested
        if strategies_filter.lower() != 'all':
            requested = [s.strip() for s in strategies_filter.split(',')]
            available_strategies = {
                k: v for k, v in available_strategies.items()
                if k in requested
            }

        print(f"[*] Running {len(available_strategies)} strategies: {', '.join(available_strategies.keys())}")

        # Results storage
        all_results = {}

        # Run analysis for each symbol
        for symbol in symbols:
            print(f"\n[*] Analyzing {symbol}...")

            symbol_results = {
                'symbol': symbol,
                'strategy_signals': [],
                'strategy_details': {}
            }

            # Run each strategy
            for strategy_name, strategy_class in available_strategies.items():
                try:
                    # Instantiate strategy
                    strategy = strategy_class(self.framework)

                    # Set options
                    strategy.set_option('SYMBOL', symbol)
                    strategy.set_option('PERIOD', period)
                    strategy.set_option('INTERVAL', interval)

                    # Run strategy
                    print(f"  [*] Running {strategy_name}...")
                    results = strategy.run()

                    # Extract signal
                    signal_strength, signal_type, confidence = self._extract_signal_from_results(
                        results, strategy_name
                    )

                    # Store signal
                    symbol_results['strategy_signals'].append(
                        (strategy_name, signal_strength, signal_type, confidence)
                    )

                    # Store details
                    symbol_results['strategy_details'][strategy_name] = {
                        'signal_strength': signal_strength,
                        'signal_type': signal_type,
                        'confidence': confidence,
                        'raw_results': results
                    }

                except Exception as e:
                    print(f"  [!] Strategy {strategy_name} failed: {str(e)}")
                    continue

            # Calculate consensus
            consensus = self._calculate_consensus(symbol_results['strategy_signals'])
            symbol_results['consensus'] = consensus

            # Store results
            all_results[symbol] = symbol_results

        # Rank symbols by consensus strength
        ranked_symbols = sorted(
            all_results.items(),
            key=lambda x: abs(x[1]['consensus']['consensus_strength']),
            reverse=True
        )

        # Display results
        self._display_results(ranked_symbols, min_consensus)

        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "symbols_analyzed": symbols,
            "strategies_used": list(available_strategies.keys()),
            "results": all_results,
            "ranked_symbols": [(s[0], s[1]['consensus']) for s in ranked_symbols]
        }

    def _display_results(self, ranked_symbols: List[Tuple[str, Dict]], min_consensus: float):
        """
        Display meta-analysis results in a formatted table.

        Args:
            ranked_symbols: List of (symbol, results) tuples sorted by consensus strength
            min_consensus: Minimum consensus threshold to highlight
        """
        print("\n" + "=" * 100)
        print("META-ANALYSIS RESULTS".center(100))
        print("=" * 100 + "\n")

        # Header
        print(f"{'Rank':<6} {'Symbol':<10} {'Consensus':<12} {'Strength':<12} {'Confidence':<12} {'Agreement':<12} {'B/S/H':<15}")
        print("-" * 100)

        # Results
        for rank, (symbol, results) in enumerate(ranked_symbols, 1):
            consensus = results['consensus']

            signal = consensus['consensus_signal']
            strength = consensus['consensus_strength']
            confidence = consensus['consensus_confidence']
            agreement = consensus['agreement_pct']
            buy_count = consensus['buy_count']
            sell_count = consensus['sell_count']
            hold_count = consensus['hold_count']

            # Color coding based on signal
            if signal == 'BUY' and agreement >= min_consensus:
                signal_display = f"🟢 {signal}"
            elif signal == 'SELL' and agreement >= min_consensus:
                signal_display = f"🔴 {signal}"
            else:
                signal_display = f"⚪ {signal}"

            bsh_display = f"{buy_count}/{sell_count}/{hold_count}"

            print(f"{rank:<6} {symbol:<10} {signal_display:<12} {strength:>+7.1f} {'':>4} {confidence:>6.1f}% {'':>5} {agreement:>6.1f}% {'':>5} {bsh_display:<15}")

        print("\n" + "=" * 100)
        print("\nLEGEND:")
        print("  🟢 Strong BUY consensus (agreement >= threshold)")
        print("  🔴 Strong SELL consensus (agreement >= threshold)")
        print("  ⚪ Mixed or HOLD signals")
        print("  B/S/H = Buy signals / Sell signals / Hold signals")
        print("\n" + "=" * 100)

        # Detailed breakdown for top symbols
        print("\n\nDETAILED BREAKDOWN (Top 3 Symbols):\n")

        for rank, (symbol, results) in enumerate(ranked_symbols[:3], 1):
            consensus = results['consensus']
            print(f"\n#{rank} {symbol} - {consensus['consensus_signal']} "
                  f"(Strength: {consensus['consensus_strength']:+.1f}, "
                  f"Agreement: {consensus['agreement_pct']:.1f}%)")
            print("-" * 80)

            # Sort strategies by signal strength
            strategy_signals = sorted(
                results['strategy_signals'],
                key=lambda x: abs(x[1]),
                reverse=True
            )

            for strategy_name, signal_strength, signal_type, confidence in strategy_signals:
                signal_icon = "🟢" if signal_type == "BUY" else "🔴" if signal_type == "SELL" else "⚪"
                print(f"  {signal_icon} {strategy_name:<30} {signal_type:<8} "
                      f"Strength: {signal_strength:>+7.1f}  Confidence: {confidence:>6.1f}%")

        print("\n" + "=" * 100 + "\n")
