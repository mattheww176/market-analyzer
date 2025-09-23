# 🌐 Market Analyzer Web Interface

A powerful and user-friendly web application for comprehensive stock market analysis, featuring interactive charts, technical indicators, and AI-powered price predictions. Get detailed technical analysis, fundamental metrics, and predictive insights for smarter trading decisions.

## 🆕 Latest Updates (v1.3.0)

### New Features
- **AI-Powered Price Predictions**: Advanced machine learning models for 1-day, 3-day, 7-day, 14-day, and 30-day price forecasts with confidence intervals
- **Enhanced Technical Analysis**: Comprehensive technical indicators including RSI, MACD, Bollinger Bands, and moving averages
- **Improved Chart Visualization**: Interactive charts with better tooltips, zooming, and comparison capabilities
- **Responsive Design**: Optimized for all devices with improved mobile experience
- **Moving Average Crossover Signals**: Clear buy/sell/hold signals based on 10-day and 30-day moving average crossovers
- **Enhanced Signal Display**: Detailed signals with confidence levels, explanations, and actionable recommendations

### Improvements
- **UI/UX**: Redesigned interface with dark/light mode support and improved data visualization
- **Performance**: Optimized data fetching and processing for faster analysis
- **Error Handling**: More descriptive error messages and graceful degradation
- **Code Quality**: Refactored codebase with better documentation and type hints
- **Security**: Enhanced input validation and error handling

### Bug Fixes
- Fixed chart rendering issues on mobile devices
- Improved data loading reliability
- Fixed theme persistence across page refreshes

## 📚 Understanding Stock Analysis

### Technical Analysis Terms

#### Moving Averages (MA)
- **Simple Moving Average (SMA)**: The average stock price over a specific period, smoothing out price fluctuations to identify trends.
- **Exponential Moving Average (EMA)**: Similar to SMA but gives more weight to recent prices, making it more responsive to new information.
- **Golden Cross**: When a short-term moving average crosses above a long-term moving average, often seen as a bullish signal.
- **Death Cross**: When a short-term moving average crosses below a long-term moving average, often seen as a bearish signal.

#### Support and Resistance
- **Support Level**: A price level where a stock tends to stop falling and may bounce back up, as demand increases at that price.
- **Resistance Level**: A price level where a stock tends to stop rising and may fall back, as selling pressure increases at that price.

#### Volume
- **Trading Volume**: The number of shares traded during a given period. Higher volume often confirms the strength of a price movement.
- **Volume Weighted Average Price (VWAP)**: The average price a stock has traded at throughout the day, based on both volume and price.

#### Momentum Indicators
- **Relative Strength Index (RSI)**: Measures the speed and change of price movements, indicating overbought (>70) or oversold (<30) conditions.
- **Moving Average Convergence Divergence (MACD)**: Shows the relationship between two moving averages, helping identify trend changes and momentum.

### Fundamental Analysis Terms

#### Valuation Metrics
- **P/E Ratio (Price-to-Earnings)**: Compares a company's stock price to its earnings per share, indicating how much investors are willing to pay per dollar of earnings.
- **P/B Ratio (Price-to-Book)**: Compares a company's market value to its book value, showing how much investors pay for each dollar of net assets.
- **Dividend Yield**: Shows how much a company pays out in dividends each year relative to its stock price.

#### Financial Health
- **Debt-to-Equity Ratio**: Measures a company's financial leverage by comparing its total liabilities to shareholders' equity.
- **Current Ratio**: Indicates a company's ability to pay short-term obligations with its current assets.

#### Growth Metrics
- **Earnings Per Share (EPS) Growth**: The rate at which a company's earnings per share are growing year over year.
- **Revenue Growth**: The rate at which a company's revenue is increasing over time.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/market-analyzer.git
   cd market-analyzer
   ```

2. **Install Dependencies**:
   ```bash
   pip3 install -r requirements.txt
   ```
   or
   ```bash
   pip3 install flask yfinance pandas numpy matplotlib scikit-learn
   ```

3. **Run the Web Server**:
   ```bash
   python3 app.py
   ```
   - Your default browser will open automatically to http://127.0.0.1:5001
   - If the browser doesn't open automatically, you can manually navigate to the URL

4. **Start Analyzing**:
   - Enter a stock ticker (e.g., AAPL, MSFT, GOOG, NVDA)
   - Select your preferred analysis type (Basic, Technical, Full, or Prediction)
   - Choose a time period (1M, 3M, 6M, 1Y, 5Y, or MAX)
   - Explore the interactive charts, technical indicators, and predictions

## 🖥️ Features

### 📊 Analysis Types
- **Basic Analysis**: Quick overview of stock performance and key metrics
- **Technical Analysis**: Detailed technical indicators and chart patterns
- **Full Analysis**: Comprehensive analysis combining technical and fundamental metrics
- **Prediction Analysis**: AI-powered price forecasts with confidence intervals

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
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - Multiple Moving Averages (SMA 10, 20, 50, 200)
- **Prediction Tools**:
  - 1-day to 30-day price forecasts
  - Confidence intervals for predictions
  - Support and resistance levels
  - Volatility analysis

## 🔧 Troubleshooting

### Common Issues

#### Data Not Loading
- Ensure you have a stable internet connection
- Verify the stock ticker is correct and exists
- Check browser console for JavaScript errors (F12 → Console)
- Try clearing your browser cache or using incognito mode

#### Analysis Not Displaying
- Make sure all dependencies are installed correctly
- Check the terminal for any Python errors
- Verify that the Flask server is running without errors
- Ensure you have the required permissions to access financial data

#### Prediction Analysis Issues
- Ensure you have all required Python packages installed
- Check that the model files are in the correct location
- Verify that the historical data is being loaded correctly

## 📞 Support

For additional help or to report issues, please:
1. Check the [GitHub Issues](https://github.com/yourusername/market-analyzer/issues) page
2. Open a new issue with detailed steps to reproduce the problem
3. Include any error messages and your environment details

## 🤝 Contributing

We welcome contributions! Please follow these steps:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with ❤️ using Python, Flask, and modern web technologies
- Powered by Yahoo Finance for market data
- Special thanks to all contributors who have helped improve this project
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
