# 🌐 Market Analyzer Web Interface

A user-friendly web interface for the Market Analyzer tool, allowing you to analyze stocks directly from your browser.

## 🚀 Quick Start

1. **Install Dependencies** (if not already installed):
   ```bash
   pip3 install flask
   ```

2. **Run the Web Server**:
   ```bash
   python3 app.py
   ```
   - Your browser will open automatically to the app
   - Runs on http://127.0.0.1:5000 by default

3. **Shut Down**:
   - Click the red "🛑 Shutdown Server" button at the bottom of the page
   - Confirm the shutdown when prompted
   - The server will stop and the page will close automatically

## 🖥️ Features

- **Simple Interface**: Enter any stock ticker to get started
- **One-Click Shutdown**: Safely stop the server from the web interface
- **Customizable Timeframe**: Analyze 1-365 days of historical data
- **Responsive Design**: Works on both desktop and mobile devices
- **Debug Info**: View connection details in the debug section
- **Real-time Analysis**: Get instant stock analysis
- **No Installation Needed**: Just a web browser required

## 🛠️ How to Use

1. **Enter a Stock Ticker** (e.g., AAPL, MSFT, GOOG)
2. **Select Timeframe**: Choose how many days to analyze (default: 90)
3. **Click "Analyze Stock"**: View the results instantly

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

## 🌟 Tips

- Bookmark [http://127.0.0.1:5000](http://127.0.0.1:5000) for quick access
- The server runs locally - your data stays on your computer
- For best performance, analyze 90-180 days of data

## 🔧 Troubleshooting

- **Port in use?** Make sure no other application is using port 5000
- **Page not loading?** Ensure the server is running in the terminal
- **Getting errors?** Check the terminal for detailed error messages

## 📁 Project Structure

- `app.py` - Main web application
- `templates/` - Contains the web interface files
  - `index.html` - Main web page
- `market_analyzer.py` - Core analysis functions

## 📜 License

This project is for educational purposes only. Use at your own risk.
