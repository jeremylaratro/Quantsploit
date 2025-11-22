# Quantsploit Backtesting Dashboard

A comprehensive web-based dashboard for visualizing and analyzing backtesting results from the Quantsploit trading framework.

## Features

### 📊 Main Dashboard
- **Key Performance Metrics**: Total backtests, average return, Sharpe ratio, win rate
- **Top Performers**: Visual ranking of best-performing strategies
- **Distribution Charts**: Return and Sharpe ratio distributions
- **Period Analysis**: Performance comparison across different time periods
- **Strategy Heatmap**: Interactive heatmap showing strategy × period performance
- **Symbol Performance**: Per-symbol analysis and comparison
- **Risk vs Return**: Scatter plot analysis

### 🔍 Strategy Detail View
- Deep dive into individual strategy performance
- Performance across periods and symbols
- Detailed metrics table
- Box plots for return distribution

### 📅 Period Detail View
- Analysis of specific time periods
- Strategy rankings for the period
- Symbol-level performance
- Win rate distribution

### 🔄 Multi-Run Comparison
- Compare up to 5 different backtest runs
- Track performance trends over time
- Side-by-side metrics comparison

## Installation

### Prerequisites
- Python 3.8+
- Quantsploit backtesting framework installed

### Setup

1. Navigate to the dashboard directory:
```bash
cd dashboard
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Starting the Dashboard

Run the Flask application:
```bash
python app.py
```

The dashboard will be available at: `http://localhost:5000`

### Generating Backtest Data

Before using the dashboard, you need to run backtests to generate data:

```bash
# From the project root directory
python run_comprehensive_backtest.py --symbols AAPL,MSFT,GOOGL --capital 100000
```

This will create JSON and CSV files in the `backtest_results/` directory that the dashboard will read.

### Running Custom Backtests

#### Time-based periods:
```bash
# 4 periods of 6 months each over 2 years
python run_comprehensive_backtest.py --symbols AAPL,MSFT --tspan 2y --bspan 6m --period 4
```

#### Quarterly analysis:
```bash
# Test the most recent Q2 (2nd fiscal quarter)
python run_comprehensive_backtest.py --symbols AAPL,MSFT --quarter 2 --period 1

# Test the past 4 Q2s
python run_comprehensive_backtest.py --symbols AAPL,MSFT --quarter 2 --period 4

# Test Q1 through Q3 of the most recent year
python run_comprehensive_backtest.py --symbols SPY,QQQ --quarter 1,2,3
```

## Dashboard Features Guide

### 1. Run Selection
Use the dropdown at the top to select which backtest run to analyze. The most recent run is selected by default.

### 2. Key Metrics Cards
Four cards at the top show:
- Total number of backtests executed
- Average return across all strategies
- Average Sharpe ratio
- Average win rate

### 3. Top 10 Strategies Chart
A horizontal bar chart showing the best-performing strategies by total return. Color-coded:
- Green: Positive returns
- Red: Negative returns

### 4. Distribution Charts
- **Return Distribution**: Histogram showing the frequency of different return ranges
- **Sharpe Ratio Distribution**: Pie chart breaking down Sharpe ratios into categories

### 5. Period Comparison
Bar chart comparing average and best returns across different time periods (quarters or custom periods).

### 6. Strategy Heatmap
Interactive heatmap showing performance of each strategy across different periods:
- Green: High returns
- Yellow: Medium returns
- Red: Low/negative returns

### 7. Strategy Performance Table
Comprehensive table with sortable columns:
- Strategy name
- Average return
- Average Sharpe ratio
- Win rate
- Total trades
- Maximum return
- Consistency (lower is better)

### 8. Symbol Performance
Bar chart showing average returns for each symbol tested.

### 9. Risk vs Return Scatter
Plots strategies on a volatility (risk) vs return chart to identify optimal risk-adjusted performers.

### 10. Comparison View
Navigate to the Comparison page to:
- Select up to 5 different backtest runs
- Compare overall performance metrics
- Track trends in returns and Sharpe ratios over time
- View detailed comparison tables

## File Structure

```
dashboard/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── templates/
│   ├── base.html         # Base template with navigation
│   ├── index.html        # Main dashboard
│   ├── strategy_detail.html  # Strategy analysis
│   ├── period_detail.html    # Period analysis
│   └── comparison.html   # Multi-run comparison
└── static/
    ├── css/              # Custom CSS (if needed)
    └── js/               # Custom JavaScript (if needed)
```

## API Endpoints

The dashboard provides RESTful API endpoints:

- `GET /api/runs` - List all available backtest runs
- `GET /api/summary/<timestamp>` - Get summary for a specific run
- `GET /api/detailed/<timestamp>` - Get detailed CSV data
- `GET /api/quarterly/<timestamp>` - Get quarterly comparison data
- `GET /api/strategies/<timestamp>` - Get strategy comparison data
- `GET /api/symbols/<timestamp>` - Get symbol performance data
- `GET /api/heatmap/<timestamp>` - Get heatmap data

## Customization

### Changing the Port
Edit `app.py` and modify the last line:
```python
app.run(debug=True, host='0.0.0.0', port=5000)  # Change port here
```

### Adding Custom Metrics
1. Update `DashboardDataLoader` class in `app.py` to calculate new metrics
2. Add new API endpoint
3. Update templates to display the new metrics

### Styling
The dashboard uses Bootstrap 5 and custom CSS defined in `base.html`. You can:
- Modify the gradient colors in the CSS variables
- Add custom CSS files in `static/css/`
- Customize chart colors in the JavaScript sections

## Troubleshooting

### Dashboard shows "No runs available"
- Ensure you have run at least one backtest
- Check that `backtest_results/` directory exists and contains `summary_*.json` files

### Charts not displaying
- Check browser console for JavaScript errors
- Ensure all CDN resources are loading (requires internet connection)

### Port already in use
- Change the port in `app.py` or stop the process using port 5000

### Performance issues with large datasets
- The dashboard loads all data into memory
- For very large backtests (>10,000 rows), consider pagination or filtering

## Browser Compatibility

Tested and working on:
- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Contributing

To add new features:
1. Update the Flask app with new routes/APIs
2. Create or modify templates
3. Add JavaScript for data visualization
4. Update this README with documentation

## License

Same license as the Quantsploit project.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the Quantsploit main documentation
3. Open an issue in the project repository
