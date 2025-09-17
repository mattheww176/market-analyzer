# 🌐 Market Analyzer Web Interface

A powerful and user-friendly web application for comprehensive stock market analysis, featuring interactive charts, technical indicators, and a customizable watchlist. Now with enhanced basic analysis and detailed technical insights.

## 🆕 Latest Updates (v1.1.0)

### New Features
- **Moving Average Crossover Signals**: Clear buy/sell/hold signals based on 10-day and 30-day moving average crossovers
- **Enhanced Signal Display**: Detailed signals with confidence levels, explanations, and actionable recommendations
- **Improved UI/UX**: Smoother animations, better visual feedback, and enhanced mobile responsiveness
- **Performance Optimizations**: Faster chart rendering and data processing
- **Accessibility Improvements**: Better keyboard navigation and screen reader support

### Bug Fixes
- Fixed chart rendering issues on mobile devices
- Improved data loading reliability
- Fixed theme persistence across page refreshes

## 🚀 Quick Start

1. **Install Dependencies**:
   ```bash
   pip3 install flask yfinance pandas numpy
   ```

2. **Run the Web Server**:
   ```bash
   python3 app.py
   ```
   - Your browser will open automatically to http://127.0.0.1:5001
   - If the browser doesn't open automatically, you can manually navigate to the URL

3. **Start Analyzing**:
   - Enter a stock ticker (e.g., AAPL, MSFT, GOOG)
   - Select your preferred time period
   - Explore the interactive charts and technical indicators

## 🖥️ Features

### 📈 Trading Signals
- **Moving Average Crossover**: Automated buy/sell signals based on 10-day and 30-day moving averages
  - **Bullish Signal**: When 10-day MA crosses above 30-day MA
  - **Bearish Signal**: When 10-day MA crosses below 30-day MA
  - **Confidence Levels**: High/Medium/Low based on volume and trend strength
  - **Recommendations**: Actionable trading suggestions with price targets
  - **Timeframe Analysis**: Signals for different time horizons (short/medium/long term)

### 📊 Interactive Charts & Analysis
- **Price Action**: View OHLC candlesticks with volume
- **Technical Indicators**:
  - RSI (14-period) with overbought/oversold signals
  - MACD (12/26/9) for trend identification
  - Moving Averages (SMA 50/200) for trend confirmation
  - Bollinger Bands for volatility analysis
- **Time Periods**: Analyze data across different timeframes (1D, 5D, 1M, 3M, 6M, 1Y, 5Y, MAX)
- **Chart Controls**:
  - **Zoom**: Use the range selector at the bottom or mouse wheel
  - **Pan**: Click and drag to move around the chart
  - **Reset**: Double-click to reset the view
  - **Toggle Indicators**: Click on legend items to show/hide indicators
  - **Timeframe**: Choose presets (1M, 3M, 6M, 1Y) or enter custom days (1-365)
  - **Theme**: Toggle between light and dark modes
  - **Tooltips**: Hover for detailed price information

### ⭐ Watchlist Management
- Add/remove stocks to your watchlist
- View watchlist performance at a glance
- Click on any watchlist item to view detailed analysis
- Watchlist is saved in your browser's local storage

## ⚙️ Server Management
- **Auto-start**: Server starts automatically when you run the application
- **Auto-browser**: Automatically opens your default web browser to the application
- **Port**: Runs on port 5001 by default (configurable in app.py)
- **Shutdown**: Use the 'Quit' button in the web interface to safely shut down the server

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

### ✨ UI Enhancements
- **Modern Card Design**: Clean, shadowed cards with subtle hover effects
- **Interactive Elements**: Buttons and inputs with smooth transitions
- **Loading States**: Animated spinners and skeleton loaders
- **Copy to Clipboard**: One-click ticker symbol copying
- **Form Validation**: Real-time feedback for input fields
- **Responsive Layout**: Optimized for all screen sizes from mobile to desktop

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
- Moving average signals generated client-side for fast response
- All data is for informational purposes only

## 🛠️ Development

### Added JavaScript Files
- `enhancements.js`: UI improvements and interactive elements
- `signals.js`: Moving average crossover signal generation and display

### Added CSS
- `enhancements.css`: Modern styling and animations

## 📜 License

This project is for educational purposes only. Use at your own risk. Not intended as financial advice.
