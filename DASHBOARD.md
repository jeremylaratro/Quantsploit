# Backtesting Analytics Dashboard

## Overview

The Quantsploit Backtesting Dashboard is a comprehensive web-based visualization and analysis tool designed specifically for the Quantsploit backtesting framework. It provides interactive charts, detailed metrics, and comparative analysis across strategies, time periods, and symbols.

## Quick Start

### 1. Run a Backtest

First, generate some backtest data:

```bash
# Basic backtest with default symbols
python run_comprehensive_backtest.py --symbols AAPL,MSFT,GOOGL

# Quarterly analysis
python run_comprehensive_backtest.py --symbols SPY,QQQ --quarter 2 --period 4

# Custom time periods
python run_comprehensive_backtest.py --symbols AAPL,TSLA --tspan 2y --bspan 6m --period 4
```

### 2. Launch the Dashboard

**On Linux/Mac:**
```bash
./start_dashboard.sh
```

**On Windows:**
```cmd
start_dashboard.bat
```

**Manual start:**
```bash
cd dashboard
pip install -r requirements.txt
python app.py
```

### 3. Open in Browser

Navigate to: `http://localhost:5000`

## Dashboard Features

### 🏠 Main Dashboard

The main dashboard provides a comprehensive overview of your latest backtest run:

#### Key Metrics
- **Total Backtests**: Number of strategy/symbol/period combinations tested
- **Average Return**: Mean return across all backtests
- **Average Sharpe Ratio**: Mean risk-adjusted return metric
- **Average Win Rate**: Percentage of winning trades

#### Visualizations
1. **Top 10 Strategies** - Ranked by total return
2. **Return Distribution** - Histogram showing return frequency
3. **Sharpe Ratio Distribution** - Categorized pie chart
4. **Period Comparison** - Performance across time periods
5. **Strategy Heatmap** - Interactive heatmap (strategy × period)
6. **Symbol Performance** - Returns by ticker
7. **Risk vs Return** - Volatility vs return scatter plot
8. **Strategy Table** - Comprehensive sortable table

### 📊 Data Insights

The dashboard automatically calculates and displays:

- **Performance Metrics**: Returns, Sharpe, Sortino, Calmar ratios
- **Risk Metrics**: Max drawdown, volatility, downside deviation
- **Trade Statistics**: Win rate, profit factor, trade counts
- **Comparative Analysis**: Beating buy-and-hold percentage
- **Consistency Metrics**: Standard deviation of returns

### 🔍 Strategy Deep Dive

Click on any strategy to see:
- Performance across all periods
- Box plots by symbol
- Detailed trade statistics
- Risk metrics breakdown

### 📅 Period Analysis

View performance for specific time periods:
- Strategy rankings for that period
- Symbol-level breakdown
- Win rate distribution
- Comprehensive metrics table

### 🔄 Multi-Run Comparison

Compare up to 5 different backtest runs to:
- Track performance improvements
- Identify trends over time
- Compare different configurations
- Analyze metric evolution

## Understanding the Charts

### Return Distribution
Shows how returns are distributed across all backtests. A right-skewed distribution indicates more positive returns.

### Sharpe Ratio Distribution
Categories:
- **< -1**: Poor performance with high risk
- **-1 to 0**: Negative risk-adjusted returns
- **0 to 1**: Positive but suboptimal
- **1 to 2**: Good risk-adjusted returns
- **2 to 3**: Very good performance
- **> 3**: Exceptional risk-adjusted returns

### Strategy Heatmap
- **Green cells**: High positive returns
- **Yellow cells**: Moderate returns
- **Red cells**: Negative returns
- Hover to see exact values

### Risk vs Return Scatter
- **Upper left quadrant**: Best (high return, low volatility)
- **Upper right quadrant**: High return but high risk
- **Lower left quadrant**: Low return, low risk
- **Lower right quadrant**: Worst (low return, high risk)

## Use Cases

### 1. Strategy Selection
Use the dashboard to identify which strategies work best:
- Check the Top 10 chart
- Filter by Sharpe ratio > 1.5
- Look for consistency (low standard deviation)

### 2. Period Analysis
Understand how strategies perform in different market conditions:
- Compare quarterly performance
- Identify seasonal patterns
- Analyze regime-specific behavior

### 3. Symbol Selection
Determine which symbols are most profitable:
- View symbol performance chart
- Check best strategy per symbol
- Analyze risk-adjusted returns

### 4. Risk Management
Assess risk characteristics:
- Review max drawdown metrics
- Check volatility levels
- Compare Sharpe and Sortino ratios

### 5. Strategy Optimization
Iterate and improve:
- Compare different backtest runs
- Track metric improvements
- A/B test strategy variations

## Advanced Features

### API Access
The dashboard exposes RESTful APIs for programmatic access:

```bash
# Get all runs
curl http://localhost:5000/api/runs

# Get summary for specific run
curl http://localhost:5000/api/summary/20251122_203908

# Get detailed data
curl http://localhost:5000/api/detailed/20251122_203908

# Get heatmap data
curl http://localhost:5000/api/heatmap/20251122_203908
```

### Custom Metrics
You can extend the dashboard to calculate custom metrics:

1. Edit `dashboard/app.py`
2. Add calculation logic to `DashboardDataLoader`
3. Create new API endpoint
4. Update templates to display

### Export Data
All charts support export:
- Click the camera icon on Plotly charts to save as PNG
- Right-click Chart.js charts to save image
- Use browser print to PDF for full dashboard

## Performance Tips

### For Large Datasets
- The dashboard loads all data into browser memory
- For > 1000 data points, charts may be slow
- Consider filtering by date range or symbols

### Refresh Strategy
- Use the "Refresh Data" button after new backtests
- Or reload the page to pick up new runs
- Dashboard auto-selects most recent run

### Browser Recommendations
- Chrome/Edge: Best performance
- Firefox: Good performance
- Safari: May have chart rendering delays

## Customization

### Change Port
Edit `dashboard/app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=8080)  # Change to 8080
```

### Modify Colors
Edit color schemes in `templates/base.html`:
```css
:root {
    --primary-color: #2c3e50;
    --success-color: #27ae60;
    /* Add your colors */
}
```

### Add Custom Charts
1. Create new chart function in template
2. Add data processing in `app.py`
3. Create API endpoint if needed

## Troubleshooting

### Dashboard Won't Start
```bash
# Check Python version (need 3.8+)
python --version

# Install dependencies
pip install Flask pandas numpy

# Check if port 5000 is free
lsof -i :5000  # On Linux/Mac
netstat -ano | findstr :5000  # On Windows
```

### No Data Showing
- Verify `backtest_results/` directory exists
- Check for `summary_*.json` files
- Run a backtest first
- Check browser console for errors

### Charts Not Rendering
- Ensure internet connection (CDN resources)
- Check browser compatibility
- Clear browser cache
- Try different browser

### Slow Performance
- Close other browser tabs
- Reduce number of strategies/symbols
- Use shorter time periods
- Consider data pagination

## Best Practices

1. **Regular Backtesting**: Run backtests weekly to build historical data
2. **Consistent Parameters**: Use same capital/commission for comparability
3. **Document Changes**: Note strategy modifications in git commits
4. **Export Results**: Save key dashboards as PDFs for reports
5. **Version Control**: Track dashboard customizations in git

## Integration Examples

### Automated Reporting
```bash
#!/bin/bash
# Run backtest and capture results
python run_comprehensive_backtest.py --symbols AAPL,MSFT > backtest.log

# Start dashboard in background
cd dashboard
python app.py &
DASHBOARD_PID=$!

# Wait for startup
sleep 5

# Screenshot with headless browser (requires playwright)
python -c "
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto('http://localhost:5000')
    page.screenshot(path='dashboard.png', full_page=True)
    browser.close()
"

# Stop dashboard
kill $DASHBOARD_PID
```

### Webhook Integration
Add to your backtest completion:
```python
import requests
import json

# After backtest completes
results = {...}  # Your results

# Send to webhook
requests.post(
    'https://your-webhook.com/backtest-complete',
    json={'results': results, 'dashboard_url': 'http://localhost:5000'}
)
```

## Updates and Maintenance

The dashboard is actively maintained alongside Quantsploit. To update:

```bash
cd dashboard
git pull origin main
pip install -r requirements.txt --upgrade
```

## Support

For dashboard-specific issues:
- Check the `dashboard/README.md` for detailed documentation
- Review browser console for JavaScript errors
- Verify Flask logs for server errors
- Open an issue on GitHub with error details

For backtesting framework issues:
- See main `README.md`
- Review strategy documentation
- Check `ADVANCED_STRATEGIES.md`

## Contributing

To contribute dashboard improvements:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

Focus areas for contributions:
- New chart types
- Additional metrics
- Performance optimizations
- Mobile responsiveness
- Export features
- Documentation

## Roadmap

Planned features:
- [ ] Real-time backtest progress tracking
- [ ] Machine learning strategy suggestions
- [ ] Automated report generation
- [ ] Mobile app version
- [ ] Cloud deployment option
- [ ] Multi-user support
- [ ] Strategy backtesting scheduler
- [ ] Alert/notification system

---

**Happy backtesting and may your Sharpe ratios be ever high!** 📈
