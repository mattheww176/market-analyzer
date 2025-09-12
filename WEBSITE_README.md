# 🌐 Market Analyzer Web Interface

A powerful and user-friendly web application for comprehensive stock market analysis, featuring interactive charts, technical indicators, and a customizable watchlist. Now with enhanced basic analysis and detailed technical insights.

## 🚀 Quick Start

1. **Install Dependencies**:
   ```bash
   pip3 install flask yfinance pandas numpy
   ```

2. **Run the Web Server**:
   ```bash
   python3 app.py
   ```
   - Your browser will open automatically to http://127.0.0.1:5000

3. **Start Analyzing**:
   - Enter a stock ticker (e.g., AAPL, MSFT, GOOG)
   - Select your preferred time period
   - Explore the interactive charts and technical indicators

## 🖥️ Features

### 📊 Interactive Charts & Analysis
- **Price Action**: View OHLC candlesticks with volume
- **Enhanced Basic Analysis**:
  - Volatility assessment with risk levels
  - Momentum detection and trend analysis
  - Volume trend indicators
  - Support/Resistance levels
  - Color-coded risk metrics
- **Technical Indicators**:
  - RSI (14-period) with overbought/oversold signals
  - MACD (12/26/9) for trend identification
  - Dual moving averages (50/200) for trend confirmation
  - Price action patterns
- **Chart Controls**:
  - **Zoom**: Click and drag to zoom in, double-click to reset
  - **Pan**: Click and drag to move around the chart
  - **Timeframe**: Choose presets (1M, 3M, 6M, 1Y) or enter custom days (1-365)
  - **Theme**: Toggle between light and dark modes
  - **Tooltips**: Hover for detailed price information

### ⭐ Watchlist Management
- Save and organize your favorite stocks
- One-click analysis of watchlist items
- Persistent storage between sessions

## ⚙️ Server Management

### Starting the Server
```bash
python3 app.py
```

### Stopping the Server
- **Normal Shutdown**: Press `Ctrl+C` in the terminal
- **Force Quit** (if unresponsive):
  ```bash
  # Find the process ID
  lsof -i :5000
  
  # Kill the process (replace PID with actual number)
  kill -9 PID
  ```

## 🔍 Analysis Types

### Basic Analysis
- Quick overview with key metrics
- Volatility and risk assessment
- Volume and momentum analysis
- Support/Resistance levels
- Color-coded trend indicators

### Technical Analysis
- Advanced technical indicators
- Moving average crossovers
- Detailed RSI and MACD analysis
- Price action patterns
- Volume analysis

### Full Analysis
- Combines all analysis types
- Comprehensive market insights
- Detailed technical breakdown
- Actionable trading signals

## 📱 Mobile & Troubleshooting

### Mobile Experience
- Fully responsive design for all screen sizes
- Touch-optimized charts with pinch-to-zoom
- Full feature parity with desktop version

### Common Issues
- **Charts not loading?**
  - Check internet connection and ticker symbol
  - Refresh the page
- **Data issues?**
  - Verify time period selection
  - Note: Some stocks have limited historical data
- **Performance?**
  - Reduce time period
  - Close other tabs
  - Clear browser cache

## 💡 Tips & Best Practices

- **Performance**:
  - Analyze 90-180 days of data for optimal performance
  - Reduce time period if experiencing slowdowns
  - Close unnecessary browser tabs

- **Data Usage**:
  - First load fetches data from Yahoo Finance
  - Subsequent views use cached data when possible
  - Refresh to get the latest market data

### 📁 Project Structure

- `app.py` - Main web application
- `templates/` - Contains the web interface files
  - `index.html` - Main web page
- `market_analyzer.py` - Core analysis functions
- `static/js/` - Frontend JavaScript modules
  - `main.js` - Core application logic
  - `chart.js` - Chart rendering and interactions
  - `technical-analysis.js` - Technical indicator calculations
  - `watchlist.js` - Watchlist management
  - `utils.js` - Helper functions

## 📊 Data Sources
- Real-time and historical market data provided by Yahoo Finance
- Technical indicators calculated in real-time
- All data is for informational purposes only

## 📜 License

This project is for educational purposes only. Use at your own risk. Not intended as financial advice.
