# Market Analyzer - Complete Features Guide

A comprehensive Python-based stock analysis tool with advanced technical indicators, predictive modeling, and market scanning capabilities.

## 🚀 Core Features Overview

### 📊 Technical Analysis Suite
- **Moving Averages**: 20-day, 50-day, and 200-day moving averages
- **RSI (Relative Strength Index)**: 14-period RSI with overbought/oversold signals
- **MACD**: Moving Average Convergence Divergence with signal line crossovers
- **Bollinger Bands**: 20-period bands with 2 standard deviations
- **Volume Analysis**: 15-day volume moving average and volume ratios
- **Daily Returns**: Percentage change calculations and statistics

### 🎯 Trading Signals & Strategy
- **RSI Signals**: BUY when RSI < 30 (oversold), SELL when RSI > 70 (overbought)
- **MACD Crossovers**: BUY/SELL signals when MACD crosses above/below signal line
- **Moving Average Crossovers**: BUY when MA20 crosses above MA50, SELL when below
- **Signal Backtesting**: Test trading strategies with historical data and performance metrics
- **Trade Log**: Complete record of all buy/sell signals with dates and prices

### 🔮 Predictive Analytics
- **Price Predictions**: Linear regression model predicting prices 1, 3, 7, and 14 days ahead
- **Trend Analysis**: Identifies UPWARD, DOWNWARD, or SIDEWAYS trends
- **Confidence Scoring**: R² score-based confidence levels (High/Medium/Low)
- **Price Targets**: Multiple scenario-based price targets:
  - Conservative (1σ up movement)
  - Moderate (average recent performance)
  - Optimistic (best 3-month repeat)
  - Resistance (90% of year high)
  - Support (10% above year low)

### 📈 Advanced Momentum Analysis
- **Comprehensive Momentum Score**: 0-100 scoring system based on:
  - Price momentum (1D, 5D, 20D, 50D performance)
  - RSI momentum and trend
  - Volume confirmation (vs 20-day average)
  - MACD signal strength
  - Moving average momentum
  - Bollinger Band position
- **Momentum Categories**: 
  - 🚀 STRONG (75+): Exceptional momentum
  - 📈 GOOD (60-74): Strong momentum
  - ➡️ MODERATE (40-59): Moderate momentum
  - 📉 WEAK (25-39): Weak momentum
  - ⬇️ POOR (<25): Poor momentum

### 🔍 Market Scanning Tools
- **Market Momentum Scanner**: Scans 60+ popular stocks for high momentum opportunities
- **Breakout Scanner**: Identifies stocks breaking 20-day highs with volume confirmation
- **Customizable Thresholds**: Set minimum momentum scores (30-100)
- **Real-time Screening**: Live analysis of market conditions
- **Top Performers**: Ranked results with key signals and metrics

### 📊 Risk & Performance Metrics
- **Sharpe Ratio**: Risk-adjusted return calculation (annualized)
- **Volatility Analysis**: Rolling volatility with customizable windows
- **Maximum Drawdown**: Peak-to-trough decline analysis
- **Cumulative Returns**: Total return performance over time
- **Daily Returns Distribution**: Histogram analysis of return patterns
- **Up/Down Day Statistics**: Win rate and performance insights

### 📈 Visualization & Charts
- **Price History Charts**: Interactive price charts with moving averages
- **Candlestick Charts**: OHLC visualization with volume
- **Technical Indicator Plots**: RSI, MACD, Bollinger Bands overlays
- **Cumulative Returns**: Performance tracking over time
- **Drawdown Charts**: Risk visualization
- **Rolling Volatility**: Risk trend analysis
- **Returns Histogram**: Distribution analysis

### 🏢 Company Fundamentals
- **Basic Information**: Company name, sector, industry, country, exchange
- **Key Metrics**: Market cap, P/E ratio, dividend yield
- **Price Levels**: 52-week high/low, current price
- **Sector Analysis**: Industry classification and peer comparison

### 📊 Data Export & Reporting
- **CSV Export**: Full historical data with all calculated indicators
- **Excel Integration**: Formatted spreadsheet output
- **Summary Statistics**: Key metrics and performance summaries
- **Email Reports**: Automated email delivery of analysis results
- **SMS Alerts**: Twilio integration for mobile notifications
- **Custom Date Ranges**: Flexible time period analysis

### 🔄 Batch Processing
- **Multi-Stock Analysis**: Analyze multiple tickers simultaneously
- **Batch Summary Mode**: Quick overview of multiple stocks
- **Comparative Analysis**: Side-by-side performance comparison
- **Portfolio Screening**: Analyze entire portfolios at once

### ⚙️ Advanced Configuration
- **Flexible Time Periods**: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
- **Customizable Indicators**: Adjustable periods for RSI, MACD, moving averages
- **Signal Sensitivity**: Configurable thresholds for buy/sell signals
- **Output Formats**: Multiple display and export options

## 🛠️ Technical Implementation

### Data Sources
- **Yahoo Finance API**: Real-time and historical stock data via yfinance
- **Robust Error Handling**: Graceful handling of missing or invalid data
- **Data Validation**: Automatic data cleaning and interpolation
- **Multi-Index Support**: Handles complex data structures

### Machine Learning
- **Scikit-learn Integration**: Linear regression for price predictions
- **Model Validation**: R² scoring for prediction confidence
- **Feature Engineering**: Technical indicators as ML features
- **Trend Classification**: Automated trend direction detection

### Performance Optimization
- **Efficient Calculations**: Vectorized operations with pandas/numpy
- **Memory Management**: Optimized data structures for large datasets
- **Caching**: Intelligent data caching for repeated analyses
- **Parallel Processing**: Multi-stock analysis optimization

## 🎯 Use Cases

### Day Trading
- Real-time momentum scanning
- Breakout identification
- Quick signal generation
- Risk assessment tools

### Swing Trading
- Multi-day price predictions
- Trend analysis
- Support/resistance levels
- Signal backtesting

### Long-term Investing
- Fundamental analysis
- Historical performance review
- Risk metrics evaluation
- Portfolio screening

### Research & Analysis
- Market condition assessment
- Sector performance comparison
- Statistical analysis
- Custom indicator development

## 📋 Command Line Interface

### Single Stock Analysis
```bash
# Basic analysis
python market_analyzer.py --ticker AAPL

# With predictions and momentum
python market_analyzer.py --ticker NVDA --predictions --momentum

# Price targets and sector info
python market_analyzer.py --ticker TSLA --price-targets --sector-info
```

### Market Scanning
```bash
# Momentum scanner
python market_analyzer.py --market-scan

# Breakout scanner
python market_analyzer.py --breakout-scan

# Combined analysis
python market_analyzer.py --ticker AAPL --momentum --predictions --price-targets
```

### Batch Processing
```bash
# Multiple stocks
python market_analyzer.py --batch "AAPL,TSLA,NVDA,MSFT" --batch-summary-only

# With email reporting
python market_analyzer.py --ticker AAPL --email your@email.com

# SMS alerts
python market_analyzer.py --ticker NVDA --sms +1234567890
```

## 🔧 Integration Capabilities

### Email Integration
- SMTP configuration for Gmail/Outlook
- Automated report delivery
- Custom formatting and styling
- Attachment support

### SMS Integration
- Twilio API integration
- Real-time alerts
- Customizable message content
- International support

### External APIs
- Yahoo Finance integration
- Extensible for additional data sources
- Rate limiting and error handling
- Data quality validation

## 📊 Output Examples

### Momentum Analysis Output
```
=== 📈 Momentum Analysis for NVDA ===
Overall Momentum: 🟢 STRONG (Score: 78/100)
Current Price: $167.02

📊 Price Momentum:
  1 Day:   -2.70%
  5 Days:  +1.45%
  20 Days: +8.32%
  50 Days: +15.67%

🔍 Technical Momentum:
  RSI: 29.2 (momentum: -5.3)
  Volume Ratio: 1.18x average
  MACD Signal: -0.460
  MA Momentum: +3.12%
  BB Position: 15.4%
```

### Price Predictions Output
```
=== Price Predictions for NVDA ===
Current Price: $167.02
Prediction Model: Linear Regression (30-day trend)
Trend Direction: SIDEWAYS ➡️
Confidence: 🔴 Low (R² = 0.105)

Future Price Predictions:
  Tomorrow: $175.28 (+4.9%)
    3 days: $174.97 (+4.8%)
    7 days: $174.35 (+4.4%)
   14 days: $173.27 (+3.7%)
```

### Market Scanner Output
```
🚀 Found 12 stocks with high momentum:
================================================================================
Rank Ticker Price    Score 1D%   20D%    Category     Key Signals
--------------------------------------------------------------------------------
1    NVDA   $167.02  78    -2.7%  +8.3%   STRONG       Strong trend, High volume
2    AAPL   $225.77  72    +1.2%  +12.1%  GOOD         RSI rising, Breakout
3    TSLA   $248.50  68    +0.8%  +15.2%  GOOD         Strong trend, RSI rising
```

## 🚀 Advanced Features

### Custom Indicator Development
- Extensible framework for new indicators
- Plugin architecture for custom calculations
- Integration with existing signal system

### Portfolio Management
- Multi-stock correlation analysis
- Portfolio-level risk metrics
- Rebalancing recommendations
- Performance attribution

### Alert System
- Customizable alert conditions
- Multiple delivery methods
- Alert history and tracking
- Conditional logic support

### API Integration
- RESTful API endpoints
- JSON data exchange
- Authentication and security
- Rate limiting and quotas

This comprehensive feature set makes Market Analyzer a professional-grade tool suitable for traders, investors, and financial analysts of all levels.

---

# Git File Recovery Guide 🚨

**EMERGENCY FILE RECOVERY - READ THIS IF YOU LOST A FILE!**

## Quick Recovery Commands

### 1. File Just Deleted (Not Committed Yet)
```bash
# Recover specific file
git checkout HEAD -- filename.py

# Recover ALL deleted files
git checkout HEAD -- .

# Example:
git checkout HEAD -- market_analyzer.py
```

### 2. File Deleted and Already Committed
```bash
# See recent commits
git log --oneline

# Recover from previous commit
git checkout HEAD~1 -- filename.py

# Example:
git checkout HEAD~1 -- market_analyzer.py
```

---

## Complete Git Tutorial for Beginners

### What is Git?
Git is like a **time machine** for your files. Every time you "commit" changes, Git takes a snapshot of ALL your files. You can go back to any snapshot anytime.

### Daily Git Workflow (2 Commands Only!)

**After making changes to your files:**
```bash
# 1. Stage your changes (prepare for snapshot)
git add .

# 2. Take the snapshot (commit)
git commit -m "Describe what you changed"
```

**Example:**
```bash
git add .
git commit -m "Added new technical indicators to stock analysis"
```

### Essential Recovery Commands

#### Check Status
```bash
# See what files have changed
git status

# See what specific changes were made
git diff
```

#### View History
```bash
# See all commits (snapshots)
git log --oneline

# See history of a specific file
git log --follow -- filename.py
```

#### Recovery Scenarios

**Scenario 1: Accidentally deleted a file**
```bash
git status                    # Shows "deleted: filename.py"
git checkout HEAD -- filename.py    # Recovers the file
```

**Scenario 2: File was deleted days ago**
```bash
git log --oneline            # Find commit before deletion
git checkout abc1234 -- filename.py    # Use actual commit hash
```

**Scenario 3: Want to see old version without recovering**
```bash
git show HEAD~1:filename.py    # Show file from 1 commit ago
git show HEAD~5:filename.py    # Show file from 5 commits ago
```

**Scenario 4: Committed deletion by mistake**
```bash
git reset --hard HEAD~1      # DANGER: Undoes last commit entirely
```

### Advanced Recovery

#### Find When File Was Deleted
```bash
git log --diff-filter=D --summary
```

#### Recover Specific Version
```bash
# List all commits that touched this file
git log --oneline -- filename.py

# Recover from specific commit
git checkout COMMIT_HASH -- filename.py
```

#### Recover to Different Name
```bash
git show HEAD:filename.py > recovered_filename.py
```

### Git Best Practices

#### Daily Habits
1. **Commit often** - at least once per day
2. **Use descriptive messages** - "Fixed bug in RSI calculation" not "changes"
3. **Check status before committing** - `git status`

#### Good Commit Messages
```bash
git commit -m "Added Bollinger Bands indicator"
git commit -m "Fixed MACD calculation bug"
git commit -m "Added email report functionality"
```

#### Bad Commit Messages
```bash
git commit -m "stuff"
git commit -m "changes"
git commit -m "update"
```

### Emergency Commands Reference

| Situation | Command |
|-----------|---------|
| Just deleted file | `git checkout HEAD -- filename.py` |
| Deleted file yesterday | `git checkout HEAD~1 -- filename.py` |
| See what changed | `git status` |
| See commit history | `git log --oneline` |
| Undo last commit | `git reset --hard HEAD~1` |
| See old file version | `git show HEAD~1:filename.py` |

### Understanding Git Terms

- **Repository (repo)**: Your project folder with git tracking
- **Commit**: A snapshot of all your files at a point in time
- **HEAD**: The current commit (latest snapshot)
- **HEAD~1**: One commit before current
- **HEAD~5**: Five commits before current
- **Staging**: Preparing files for commit with `git add`

### Visual Timeline Example

```
HEAD~3 ← HEAD~2 ← HEAD~1 ← HEAD (current)
  |        |        |        |
Day 1    Day 2    Day 3    Today
```

You can recover files from ANY of these points!

### Troubleshooting

#### "Not a git repository" error
```bash
cd /path/to/your/project
git init
git add .
git commit -m "Initial commit"
```

#### "Nothing to commit" message
This means no files have changed since last commit. This is normal!

#### Can't find deleted file
```bash
# Search for file in all commits
git log --all --full-history -- filename.py
```

### Setup Reminder

**If git isn't set up yet:**
```bash
cd your-project-folder
git init
git add .
git commit -m "Initial commit"
```

---

## 🆘 EMERGENCY CHECKLIST

**File is missing? Follow these steps:**

1. ✅ `cd` to your project directory
2. ✅ Run `git status` - does it show "deleted: filename.py"?
3. ✅ If yes: `git checkout HEAD -- filename.py`
4. ✅ If no: `git log --oneline` then `git checkout HEAD~1 -- filename.py`
5. ✅ Check if file is back: `ls -la filename.py`

**Still can't find it?**
- Try `git log --follow -- filename.py`
- Look for the file in different commits
- Use `git checkout COMMIT_HASH -- filename.py`

---

## Remember: Git Only Protects Committed Files!

**Files are ONLY safe if you've committed them at least once.**

Make it a habit:
```bash
git add .
git commit -m "Daily backup"
```

**Do this every day you work on your project!**
