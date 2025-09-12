import yfinance as yf
import pandas as pd
import numpy as np
import os
import matplotlib
# Try to use Qt5Agg backend first, as it's more reliable
import sys
import platform

# Set the backend before importing pyplot
if 'matplotlib.pyplot' in sys.modules:
    # If pyplot is already imported, we need to restart the kernel
    print("\n⚠️ matplotlib.pyplot already imported. Please restart the Python kernel and try again.")
    sys.exit(1)

try:
    # Try Qt5Agg first (most reliable for interactive plots)
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt
    print("\nℹ️ Using Qt5Agg backend for plotting")
    
    # Enable interactive mode
    plt.ion()
    
except Exception as e:
    print(f"\n⚠️ Could not use Qt5Agg: {e}")
    
    # Fall back to the default backend
    import matplotlib.pyplot as plt
    print("⚠️ Using default matplotlib backend. Plots may not display.")
    print("   To fix this, install PyQt5: pip install PyQt5")

# Try to use IPython's display if available
try:
    from IPython import get_ipython
    ipython = get_ipython()
    if ipython is not None:
        ipython.magic('matplotlib inline')
        print("\nℹ️ Using IPython inline display for plots")
except:
    pass  # Not in IPython, continue with standard matplotlib
from datetime import datetime, timedelta
import sys
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import argparse
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
try:
    from twilio.rest import Client
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False

# ---------------------------
# Technical Indicator Functions
# ---------------------------

def calculate_rsi(series, period=14):
    """Calculate Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(period, min_periods=1).mean()
    avg_loss = loss.rolling(period, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series, short=12, long=26, signal=9):
    short_ema = series.ewm(span=short, adjust=False).mean()
    long_ema = series.ewm(span=long, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_bollinger_bands(series, window=20, num_std=2):
    ma = series.rolling(window, min_periods=1).mean()
    std = series.rolling(window, min_periods=1).std()
    upper_band = ma + num_std * std
    lower_band = ma - num_std * std
    return ma, upper_band, lower_band

def generate_signals(data):
    signals = []
    prev_macd = prev_signal = None
    prev_ma20 = prev_ma50 = None
    for idx, row in data.iterrows():
        signal = None
        def to_scalar(val):
            if isinstance(val, pd.Series):
                if val.size == 1:
                    return val.item()
                else:
                    return np.nan
            return val
        rsi = to_scalar(row.get('RSI', np.nan))
        macd = to_scalar(row.get('MACD', np.nan))
        macd_signal = to_scalar(row.get('MACD_signal', np.nan))
        ma20 = to_scalar(row.get('MA20', np.nan))
        ma50 = to_scalar(row.get('MA50', np.nan))
        # Skip if any are NaN
        if pd.isna(rsi) or pd.isna(macd) or pd.isna(macd_signal) or pd.isna(ma20) or pd.isna(ma50):
            signals.append(signal)
            prev_macd = macd
            prev_signal = macd_signal
            prev_ma20 = ma20
            prev_ma50 = ma50
            continue
        # RSI signals
        if rsi < 30:
            signal = 'BUY (RSI < 30)'
        elif rsi > 70:
            signal = 'SELL (RSI > 70)'
        # MACD crossover signals
        if prev_macd is not None and prev_signal is not None:
            if prev_macd < prev_signal and macd > macd_signal:
                signal = 'BUY (MACD crossover)'
            elif prev_macd > prev_signal and macd < macd_signal:
                signal = 'SELL (MACD crossover)'
        # MA crossover signals
        if prev_ma20 is not None and prev_ma50 is not None:
            if prev_ma20 < prev_ma50 and ma20 > ma50:
                signal = 'BUY (MA20 crosses above MA50)'
            elif prev_ma20 > prev_ma50 and ma20 < ma50:
                signal = 'SELL (MA20 crosses below MA50)'
        signals.append(signal)
        prev_macd = macd
        prev_signal = macd_signal
        prev_ma20 = ma20
        prev_ma50 = ma50
    return signals

def calculate_max_drawdown(series):
    cumulative = series.cummax()
    drawdown = (series - cumulative) / cumulative
    max_drawdown = drawdown.min()
    return drawdown, max_drawdown

def calculate_price_targets(data, ticker):
    """Calculate price targets based on historical performance patterns."""
    current_price = data['Close'].iloc[-1]
    # Ensure current_price is a scalar
    if hasattr(current_price, 'item'):
        current_price = current_price.item()
    
    # Calculate various time period returns
    def get_scalar_price(price_val):
        return price_val.item() if hasattr(price_val, 'item') else price_val
    
    returns_1m = (current_price / get_scalar_price(data['Close'].iloc[-21]) - 1) * 100 if len(data) >= 21 else None
    returns_3m = (current_price / get_scalar_price(data['Close'].iloc[-63]) - 1) * 100 if len(data) >= 63 else None
    returns_6m = (current_price / get_scalar_price(data['Close'].iloc[-126]) - 1) * 100 if len(data) >= 126 else None
    returns_1y = (current_price / get_scalar_price(data['Close'].iloc[-252]) - 1) * 100 if len(data) >= 252 else None
    
    # Calculate average returns for different periods
    daily_returns = data['Daily % Change'].dropna()
    avg_daily_return = daily_returns.mean()
    volatility = daily_returns.std()
    
    # Historical high/low analysis
    year_high = get_scalar_price(data['Close'].max())
    year_low = get_scalar_price(data['Close'].min())
    
    # Price targets based on different scenarios
    targets = {}
    
    # Conservative target (1 standard deviation move up)
    targets['Conservative (1σ up)'] = current_price * (1 + (avg_daily_return + volatility) / 100 * 21)  # 1 month projection
    
    # Moderate target (average of recent performance)
    recent_performance = []
    if returns_1m is not None: recent_performance.append(returns_1m)
    if returns_3m is not None: recent_performance.append(returns_3m)
    if returns_6m is not None: recent_performance.append(returns_6m)
    
    if recent_performance:
        avg_recent = sum(recent_performance) / len(recent_performance)
        targets['Moderate (avg recent)'] = current_price * (1 + avg_recent / 100)
    
    # Optimistic target (if it repeats best 3-month period)
    if len(data) >= 63:
        best_3m_return = 0
        for i in range(63, len(data)):
            period_return = (get_scalar_price(data['Close'].iloc[i]) / get_scalar_price(data['Close'].iloc[i-63]) - 1) * 100
            if period_return > best_3m_return:
                best_3m_return = period_return
        targets['Optimistic (best 3m repeat)'] = current_price * (1 + best_3m_return / 100)
    
    # Resistance level (approach to year high)
    targets['Resistance (90% of year high)'] = year_high * 0.9
    
    # Support level (potential bounce from low)
    targets['Support (10% above year low)'] = year_low * 1.1
    
    print(f"\n=== Price Targets for {ticker} ===")
    print(f"Current Price: ${current_price:.2f}")
    print(f"52-Week High: ${year_high:.2f} (+{((year_high/current_price-1)*100):.1f}%)")
    print(f"52-Week Low: ${year_low:.2f} ({((year_low/current_price-1)*100):.1f}%)")
    print(f"\nPrice Target Scenarios:")
    
    for scenario, target_price in targets.items():
        if target_price and target_price > 0:
            change_pct = ((target_price / current_price) - 1) * 100
            print(f"{scenario:25s}: ${target_price:.2f} ({change_pct:+.1f}%)")
    
    # Recent performance summary
    print(f"\nRecent Performance:")
    if returns_1m is not None: print(f"1 Month:  {returns_1m:+.1f}%")
    if returns_3m is not None: print(f"3 Month:  {returns_3m:+.1f}%")
    if returns_6m is not None: print(f"6 Month:  {returns_6m:+.1f}%")
    if returns_1y is not None: print(f"1 Year:   {returns_1y:+.1f}%")
    
    return targets

def calculate_momentum_score(data, ticker):
    """
    Calculate comprehensive momentum score for a stock.
    Returns momentum analysis with multiple indicators.
    """
    if len(data) < 50:
        print(f"❌ Not enough data for momentum analysis of {ticker}")
        return None
    
    # Price momentum calculations
    current_price = data['Close'].iloc[-1]
    if hasattr(current_price, 'item'):
        current_price = current_price.item()
    
    # 1. Price momentum (% change over different periods)
    price_1d_base = data['Close'].iloc[-2]
    if hasattr(price_1d_base, 'item'):
        price_1d_base = price_1d_base.item()
    price_1d = ((current_price - price_1d_base) / price_1d_base) * 100
    
    price_5d_base = data['Close'].iloc[-6]
    if hasattr(price_5d_base, 'item'):
        price_5d_base = price_5d_base.item()
    price_5d = ((current_price - price_5d_base) / price_5d_base) * 100
    
    price_20d_base = data['Close'].iloc[-21]
    if hasattr(price_20d_base, 'item'):
        price_20d_base = price_20d_base.item()
    price_20d = ((current_price - price_20d_base) / price_20d_base) * 100
    
    price_50d_base = data['Close'].iloc[-51]
    if hasattr(price_50d_base, 'item'):
        price_50d_base = price_50d_base.item()
    price_50d = ((current_price - price_50d_base) / price_50d_base) * 100
    
    # 2. RSI momentum (current RSI and trend)
    current_rsi = data['RSI'].iloc[-1]
    if hasattr(current_rsi, 'item'):
        current_rsi = current_rsi.item()
    rsi_5d_avg = data['RSI'].iloc[-5:].mean()
    if hasattr(rsi_5d_avg, 'item'):
        rsi_5d_avg = rsi_5d_avg.item()
    rsi_momentum = current_rsi - rsi_5d_avg
    
    # 3. Volume momentum
    current_volume = data['Volume'].iloc[-1]
    if hasattr(current_volume, 'item'):
        current_volume = current_volume.item()
    volume_20d_avg = data['Volume'].iloc[-20:].mean()
    if hasattr(volume_20d_avg, 'item'):
        volume_20d_avg = volume_20d_avg.item()
    volume_ratio = current_volume / volume_20d_avg
    
    # 4. MACD momentum
    current_macd = data['MACD'].iloc[-1]
    if hasattr(current_macd, 'item'):
        current_macd = current_macd.item()
    current_macd_signal = data['MACD_signal'].iloc[-1]
    if hasattr(current_macd_signal, 'item'):
        current_macd_signal = current_macd_signal.item()
    macd_momentum = current_macd - current_macd_signal
    
    # 5. Moving Average momentum
    ma20 = data['MA20'].iloc[-1]
    if hasattr(ma20, 'item'):
        ma20 = ma20.item()
    ma50 = data['MA50'].iloc[-1]
    if hasattr(ma50, 'item'):
        ma50 = ma50.item()
    ma_momentum = ((ma20 - ma50) / ma50) * 100
    
    # 6. Bollinger Band position
    bb_upper = data['BB_Upper'].iloc[-1]
    if hasattr(bb_upper, 'item'):
        bb_upper = bb_upper.item()
    bb_lower = data['BB_Lower'].iloc[-1]
    if hasattr(bb_lower, 'item'):
        bb_lower = bb_lower.item()
    bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) * 100
    
    # Calculate composite momentum score (0-100)
    momentum_score = 0
    
    # Price momentum scoring (40% weight)
    if price_1d > 2: momentum_score += 10
    elif price_1d > 0: momentum_score += 5
    
    if price_5d > 5: momentum_score += 10
    elif price_5d > 0: momentum_score += 5
    
    if price_20d > 10: momentum_score += 10
    elif price_20d > 0: momentum_score += 5
    
    if price_50d > 15: momentum_score += 10
    elif price_50d > 0: momentum_score += 5
    
    # RSI momentum scoring (15% weight)
    if 40 < current_rsi < 70 and rsi_momentum > 0: momentum_score += 8
    elif current_rsi > 50: momentum_score += 4
    elif current_rsi < 30: momentum_score += 2  # Oversold bounce potential
    
    # Volume momentum scoring (15% weight)
    if volume_ratio > 1.5: momentum_score += 8
    elif volume_ratio > 1.2: momentum_score += 5
    elif volume_ratio > 1.0: momentum_score += 2
    
    # MACD momentum scoring (15% weight)
    if macd_momentum > 0 and current_macd > 0: momentum_score += 8
    elif macd_momentum > 0: momentum_score += 5
    elif current_macd > current_macd_signal: momentum_score += 2
    
    # MA momentum scoring (10% weight)
    if ma_momentum > 2: momentum_score += 5
    elif ma_momentum > 0: momentum_score += 3
    
    # BB position scoring (5% weight)
    if bb_position > 80: momentum_score += 2  # Near upper band
    elif bb_position > 50: momentum_score += 3
    elif bb_position < 20: momentum_score += 1  # Oversold potential
    
    # Determine momentum category
    if momentum_score >= 75:
        momentum_category = "🚀 STRONG"
        momentum_emoji = "🟢"
    elif momentum_score >= 60:
        momentum_category = "📈 GOOD"
        momentum_emoji = "🟡"
    elif momentum_score >= 40:
        momentum_category = "➡️ MODERATE"
        momentum_emoji = "🟠"
    elif momentum_score >= 25:
        momentum_category = "📉 WEAK"
        momentum_emoji = "🔴"
    else:
        momentum_category = "⬇️ POOR"
        momentum_emoji = "🔴"
    
    return {
        'ticker': ticker,
        'momentum_score': momentum_score,
        'momentum_category': momentum_category,
        'momentum_emoji': momentum_emoji,
        'current_price': current_price,
        'price_1d': price_1d,
        'price_5d': price_5d,
        'price_20d': price_20d,
        'price_50d': price_50d,
        'current_rsi': current_rsi,
        'rsi_momentum': rsi_momentum,
        'volume_ratio': volume_ratio,
        'macd_momentum': macd_momentum,
        'ma_momentum': ma_momentum,
        'bb_position': bb_position
    }

def display_momentum_analysis(momentum_data):
    """Display detailed momentum analysis for a single stock."""
    if not momentum_data:
        return
    
    print(f"\n=== 📈 Momentum Analysis for {momentum_data['ticker']} ===")
    print(f"Overall Momentum: {momentum_data['momentum_emoji']} {momentum_data['momentum_category']} (Score: {momentum_data['momentum_score']}/100)")
    print(f"Current Price: ${momentum_data['current_price']:.2f}")
    
    print(f"\n📊 Price Momentum:")
    print(f"  1 Day:  {momentum_data['price_1d']:+6.2f}%")
    print(f"  5 Days: {momentum_data['price_5d']:+6.2f}%")
    print(f"  20 Days:{momentum_data['price_20d']:+6.2f}%")
    print(f"  50 Days:{momentum_data['price_50d']:+6.2f}%")
    
    print(f"\n🔍 Technical Momentum:")
    print(f"  RSI: {momentum_data['current_rsi']:.1f} (momentum: {momentum_data['rsi_momentum']:+.1f})")
    print(f"  Volume Ratio: {momentum_data['volume_ratio']:.2f}x average")
    print(f"  MACD Signal: {momentum_data['macd_momentum']:+.3f}")
    print(f"  MA Momentum: {momentum_data['ma_momentum']:+.2f}%")
    print(f"  BB Position: {momentum_data['bb_position']:.1f}%")
    
    # Momentum insights
    print(f"\n💡 Momentum Insights:")
    insights = []
    
    if momentum_data['price_20d'] > 15:
        insights.append("• Strong 20-day price trend")
    if momentum_data['volume_ratio'] > 1.5:
        insights.append("• High volume confirmation")
    if momentum_data['rsi_momentum'] > 5:
        insights.append("• RSI showing upward momentum")
    if momentum_data['ma_momentum'] > 3:
        insights.append("• Moving averages trending up")
    if momentum_data['bb_position'] > 80:
        insights.append("• Price near Bollinger Band upper limit")
    elif momentum_data['bb_position'] < 20:
        insights.append("• Potential oversold bounce opportunity")
    
    if not insights:
        insights.append("• Mixed momentum signals")
    
    for insight in insights:
        print(insight)

def scan_market_momentum(num_stocks=10, min_score=None):
    """
    Scan the market for stocks with high momentum.
    Uses popular stock lists and screens for momentum breakouts.
    """
    print(f"\n=== 🔍 Market Momentum Scanner ===")
    
    # Interactive prompt for minimum score if not provided
    if min_score is None:
        print("Choose your minimum momentum score threshold:")
        print("• 75+: Only exceptional momentum (very selective)")
        print("• 60+: Strong momentum (recommended)")
        print("• 45+: Moderate momentum (more results)")
        print("• 30+: Any positive momentum (broad scan)")
        
        while True:
            try:
                min_score = int(input("\nEnter minimum momentum score (30-100): ").strip())
                if 30 <= min_score <= 100:
                    break
                else:
                    print("Please enter a number between 30 and 100")
            except ValueError:
                print("Please enter a valid number")
    
    print(f"\nScanning for stocks with momentum score >= {min_score}")
    print("This may take a few minutes...")
    
    # Popular stock lists to scan
    popular_stocks = [
        # Tech giants
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX',
        # Financial
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'V', 'MA',
        # Healthcare
        'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'LLY',
        # Consumer
        'KO', 'PEP', 'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'DIS',
        # Industrial
        'BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'RTX', 'LMT',
        # Energy
        'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'MPC', 'PSX', 'VLO',
        # Growth stocks
        'ROKU', 'ZOOM', 'SHOP', 'SQ', 'PYPL', 'ADBE', 'CRM', 'NOW',
        # Emerging
        'PLTR', 'SNOW', 'COIN', 'RBLX', 'HOOD', 'SOFI', 'RIVN', 'LCID'
    ]
    
    momentum_results = []
    processed = 0
    
    for ticker in popular_stocks:
        try:
            print(f"Scanning {ticker}... ({processed + 1}/{len(popular_stocks)})")
            
            # Get data for momentum analysis
            start_date = datetime.now() - timedelta(days=365)
            end_date = datetime.now()
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if data.empty or len(data) < 50:
                continue
                
            # Add technical indicators
            data = add_technical_indicators(data)
            
            # Calculate momentum score
            momentum_data = calculate_momentum_score(data, ticker)
            
            if momentum_data and momentum_data['momentum_score'] >= min_score:
                momentum_results.append(momentum_data)
                
        except Exception as e:
            # Skip stocks with errors (delisted, etc.)
            continue
            
        processed += 1
    
    # Sort by momentum score (highest first)
    momentum_results.sort(key=lambda x: x['momentum_score'], reverse=True)
    
    # Display results
    if not momentum_results:
        print(f"\n❌ No stocks found with momentum score >= {min_score}")
        print("Try lowering the minimum score or check back later.")
        return
    
    print(f"\n🚀 Found {len(momentum_results)} stocks with high momentum:")
    print("=" * 80)
    
    # Header
    print(f"{'Rank':<4} {'Ticker':<6} {'Price':<8} {'Score':<5} {'1D%':<6} {'20D%':<7} {'Category':<12} {'Key Signals'}")
    print("-" * 80)
    
    for i, stock in enumerate(momentum_results[:num_stocks], 1):
        key_signals = []
        if stock['price_20d'] > 15:
            key_signals.append("Strong trend")
        if stock['volume_ratio'] > 1.5:
            key_signals.append("High volume")
        if stock['bb_position'] > 80:
            key_signals.append("Breakout")
        if stock['rsi_momentum'] > 5:
            key_signals.append("RSI rising")
        
        signals_str = ", ".join(key_signals[:2]) if key_signals else "Mixed"
        
        print(f"{i:<4} {stock['ticker']:<6} ${stock['current_price']:<7.2f} "
              f"{stock['momentum_score']:<5} {stock['price_1d']:+5.1f}% "
              f"{stock['price_20d']:+6.1f}% {stock['momentum_category']:<12} {signals_str}")
    
    print("=" * 80)
    print(f"💡 Tip: Use --momentum --ticker <TICKER> for detailed analysis of any stock above")
    
    return momentum_results[:num_stocks]

def scan_breakout_stocks(lookback_days=20):
    """
    Scan for stocks breaking out to new highs or breaking key resistance levels.
    """
    print(f"\n=== 📈 Breakout Scanner ===")
    print(f"Scanning for stocks breaking {lookback_days}-day highs...")
    
    # Same stock universe as momentum scanner
    popular_stocks = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX',
        'JPM', 'BAC', 'WFC', 'GS', 'V', 'MA', 'JNJ', 'PFE', 'UNH',
        'KO', 'PEP', 'WMT', 'HD', 'MCD', 'BA', 'CAT', 'XOM', 'CVX',
        'ROKU', 'ZOOM', 'SHOP', 'SQ', 'PYPL', 'ADBE', 'CRM', 'PLTR'
    ]
    
    breakout_stocks = []
    
    for ticker in popular_stocks:
        try:
            print(f"Checking {ticker} for breakouts...")
            
            start_date = datetime.now() - timedelta(days=365)
            end_date = datetime.now()
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if data.empty or len(data) < lookback_days + 5:
                continue
            
            current_price = data['Close'].iloc[-1]
            if hasattr(current_price, 'item'):
                current_price = current_price.item()
                
            # Check for breakout patterns
            recent_high = data['Close'].iloc[-(lookback_days+1):-1].max()
            if hasattr(recent_high, 'item'):
                recent_high = recent_high.item()
                
            volume_avg = data['Volume'].iloc[-10:].mean()
            current_volume = data['Volume'].iloc[-1]
            if hasattr(current_volume, 'item'):
                current_volume = current_volume.item()
            if hasattr(volume_avg, 'item'):
                volume_avg = volume_avg.item()
                
            # Breakout criteria
            price_breakout = current_price > recent_high * 1.02  # 2% above recent high
            volume_confirmation = current_volume > volume_avg * 1.3  # 30% above avg volume
            
            if price_breakout and volume_confirmation:
                breakout_pct = ((current_price - recent_high) / recent_high) * 100
                volume_ratio = current_volume / volume_avg
                
                breakout_stocks.append({
                    'ticker': ticker,
                    'current_price': current_price,
                    'breakout_pct': breakout_pct,
                    'volume_ratio': volume_ratio,
                    'recent_high': recent_high
                })
                
        except Exception:
            continue
    
    # Sort by breakout percentage
    breakout_stocks.sort(key=lambda x: x['breakout_pct'], reverse=True)
    
    if not breakout_stocks:
        print(f"\n❌ No breakout stocks found in the last {lookback_days} days")
        return
    
    print(f"\n🎯 Found {len(breakout_stocks)} stocks breaking out:")
    print("=" * 70)
    print(f"{'Ticker':<6} {'Price':<8} {'Breakout%':<10} {'Volume':<8} {'Recent High'}")
    print("-" * 70)
    
    for stock in breakout_stocks[:10]:
        print(f"{stock['ticker']:<6} ${stock['current_price']:<7.2f} "
              f"+{stock['breakout_pct']:<9.1f}% {stock['volume_ratio']:<7.1f}x "
              f"${stock['recent_high']:.2f}")
    
    print("=" * 70)
    return breakout_stocks[:10]

def predict_stock_prices(data, ticker, days_ahead=[1, 3, 7, 14]):
    """
    Predict future stock prices using multiple technical indicators and market context.
    
    Args:
        data: DataFrame containing stock data (OHLCV)
        ticker: Stock ticker symbol
        days_ahead: List of days to predict ahead
        
    Returns:
        Dictionary containing predictions and analysis
    """
    # Only show debug info if DEBUG environment variable is set
    import os
    if os.environ.get('DEBUG'):
        print(f"\nDEBUG: Inside predict_stock_prices for {ticker}")
        print(f"DEBUG: Data columns: {data.columns.tolist()}")
        print(f"DEBUG: Data shape: {data.shape}")
        print(f"DEBUG: First few rows of data:\n{data.head()}")
    try:
        # Create a clean DataFrame with just the close prices
        if isinstance(data, pd.DataFrame):
            # If data is a DataFrame, try to extract close prices
            if isinstance(data.columns, pd.MultiIndex):
                # Handle MultiIndex columns
                if ('Close', ticker) in data.columns:
                    close_series = data[('Close', ticker)]
                else:
                    close_series = data.xs('Close', axis=1, level=1, drop_level=False).iloc[:, 0]
                df = pd.DataFrame({'Close': close_series.values}, index=close_series.index)
            elif 'Close' in data.columns:
                close_series = data['Close']
                if isinstance(close_series, pd.Series):
                    df = pd.DataFrame({'Close': close_series.values}, index=close_series.index)
                else:
                    # Handle case where data['Close'] is a DataFrame
                    df = close_series.iloc[:, 0].to_frame('Close')
            else:
                print("Error: Could not find 'Close' prices in the data")
                return None
        else:
            print("Error: Invalid data format")
            return None
            
        # Ensure we have enough data
        if len(df) < 30:
            print(f"\n⚠️ Not enough data for predictions (need at least 30 days, have {len(df)})")
            return None
            
        # Calculate basic indicators
        df['Returns'] = df['Close'].pct_change()
        df['Momentum_5'] = df['Close'].pct_change(5)
        df['Momentum_10'] = df['Close'].pct_change(10)
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA50'] = df['Close'].rolling(50).mean()
        df['RSI'] = calculate_rsi(df['Close'])
        df['Volatility'] = df['Returns'].rolling(5).std() * np.sqrt(252)
        
        # Calculate price relative to moving averages
        df['Price_vs_MA20'] = (df['Close'] / df['MA20'] - 1) * 100
        df['Price_vs_MA50'] = (df['Close'] / df['MA50'] - 1) * 100
        
        # Drop NA values
        df = df.dropna()
        
        if len(df) < 30:
            print(f"\n⚠️ Not enough data after cleaning (need at least 30 days, have {len(df)})")
            return None
            
        # Get the most recent data point as a copy to avoid SettingWithCopyWarning
        current = df.iloc[-1].copy()
        current_price = current['Close']
        
        # Calculate additional technical indicators
        current_rsi = current['RSI']
        current_volatility = current['Volatility']
        
        # Calculate mean reversion factor (0.5 when RSI is 50, approaches 0 at extremes)
        mean_reversion = 1 - abs(current_rsi - 50) / 50  # 0-1 scale where 1 means neutral RSI
        
        # Base momentum (weighted average of different timeframes)
        momentum_weights = {
            1: (0.4, 0.3, 0.2, 0.1),  # 1-day weights for (5d, 10d, 20d, 50d) momentum
            3: (0.3, 0.4, 0.2, 0.1),
            7: (0.2, 0.3, 0.3, 0.2),
            14: (0.1, 0.2, 0.4, 0.3)
        }
        
        # Calculate momentum for different timeframes with bounds
        def safe_pct_change(series, periods):
            if len(series) < periods + 1:
                return 0.0
            change = (series.iloc[-1] / series.iloc[-periods-1]) - 1
            # Cap extreme values to ±20%
            return max(-0.2, min(0.2, change))
            
        # Calculate momentum values and assign them properly to avoid warnings
        current.loc['Momentum_5'] = safe_pct_change(df['Close'], 5)
        current.loc['Momentum_10'] = safe_pct_change(df['Close'], 10)
        current.loc['Momentum_20'] = safe_pct_change(df['Close'], 20)
        current.loc['Momentum_50'] = safe_pct_change(df['Close'], 50)
        
        # Calculate average true range (ATR) for volatility scaling
        # Use Close price as fallback if High/Low data is not available
        if all(col in df.columns for col in ['High', 'Low']):
            df['TR'] = np.maximum(
                df['High'] - df['Low'],
                np.maximum(
                    abs(df['High'] - df['Close'].shift(1)),
                    abs(df['Low'] - df['Close'].shift(1))
                )
            )
            atr = df['TR'].rolling(14).mean().iloc[-1]
        else:
            # Fallback: Use daily range based on close prices if OHLC data not available
            df['TR'] = abs(df['Close'] - df['Close'].shift(1))
            atr = df['TR'].rolling(14).mean().iloc[-1] * 2  # Approximate ATR
        
        # Make predictions for each timeframe with better bounds
        predictions = {}
        confidence = {}
        
        for days in sorted(days_ahead):
            # Get appropriate weights for this timeframe
            weights = next((w for d, w in sorted(momentum_weights.items()) if days <= d), (0.2, 0.3, 0.3, 0.2))
            
            # Calculate weighted momentum with bounds, using get() to avoid KeyError
            momentum = (
                current.get('Momentum_5', 0) * weights[0] * 0.5 +  # Shorter-term momentum has less weight
                current.get('Momentum_10', 0) * weights[1] * 0.7 +  # Medium-term momentum
                current.get('Momentum_20', 0) * weights[2] +        # 20-day momentum full weight
                current.get('Momentum_50', 0) * weights[3] * 1.2    # Long-term momentum slightly more weight
            )
            
            # Cap daily momentum to prevent extreme moves
            max_daily_move = 0.05  # 5% max daily move
            momentum = max(-max_daily_move, min(max_daily_move, momentum))
            
            # Adjust for mean reversion based on RSI
            rsi_factor = 1.0
            if current_rsi < 30:  # Oversold - expect bounce
                rsi_factor = 1.1
            elif current_rsi > 70:  # Overbought - expect pullback
                rsi_factor = 0.9
                
            # Scale momentum by days with diminishing returns
            days_factor = min(days, 10) / 10  # Cap at 10 days for scaling
            predicted_change = momentum * days_factor * rsi_factor
            
            # Calculate predicted price with bounds
            predicted_price = current_price * (1 + predicted_change)
            
            # Ensure price doesn't move more than 20% in any direction
            max_move = 0.2  # 20% max move
            predicted_price = max(
                current_price * (1 - max_move),
                min(current_price * (1 + max_move), predicted_price)
            )
            
            # Calculate confidence based on multiple factors
            rsi_confidence = 1 - (abs(current_rsi - 50) / 50)  # 1 when RSI is 50, 0 at extremes
            vol_confidence = 1 / (1 + current_volatility)  # Lower confidence with higher volatility
            
            # Combine confidence factors (0-1 range)
            confidence_score = (rsi_confidence * 0.6 + vol_confidence * 0.4)
            
            # Convert to percentage (30-90% range)
            confidence[days] = int(30 + confidence_score * 60)
            
            predictions[days] = predicted_price
            
        # Prepare the result
        result = {
            'ticker': ticker,
            'current_price': current_price,
            'predictions': predictions,
            'rsi': current['RSI'],
            'momentum_5': current['Momentum_5'],
            'momentum_10': current['Momentum_10'],
            'volatility': current['Volatility'],
            'price_vs_ma20': current['Price_vs_MA20'],
            'price_vs_ma50': current['Price_vs_MA50']
        }
        
        # Print comprehensive analysis
        print(f"\n=== Price Predictions for {ticker} ===")
        print(f"Current Price: ${current_price:.2f}")
        print("\n📊 Technical Indicators:")
        print(f"• RSI: {current['RSI']:.1f} (30=oversold, 70=overbought)")
        print(f"• 5-Day Momentum: {current['Momentum_5']*100:+.1f}%")
        print(f"• 10-Day Momentum: {current['Momentum_10']*100:+.1f}%")
        print(f"• 20-Day Momentum: {current['Momentum_20']*100:+.1f}%")
        print(f"• 50-Day Momentum: {current['Momentum_50']*100:+.1f}%")
        print(f"• Volatility (annualized): {current['Volatility']*100:.1f}%")
        print(f"• Price vs 20-day MA: {current['Price_vs_MA20']:+.1f}%")
        print(f"• Price vs 50-day MA: {current['Price_vs_MA50']:+.1f}%")
        
        print("\n🔮 Price Predictions:")
        for days, price in sorted(predictions.items()):
            change_pct = (price / current_price - 1) * 100
            conf = confidence.get(days, 50)
            print(f"• {days} day{'s' if days > 1 else ''}: ${price:.2f} ({change_pct:+.1f}%)")
            print(f"  Confidence: {'█' * int(conf/10)}{'░' * (10 - int(conf/10))} {conf:.0f}%")
            
        print("\n📈 Market Context:")
        if current['RSI'] > 70:
            print("• RSI indicates overbought conditions (potential pullback risk)")
        elif current['RSI'] < 30:
            print("• RSI indicates oversold conditions (potential rebound opportunity)")
        else:
            print("• RSI in neutral territory")
            
        if current['Momentum_5'] > 0.02:
            print("• Strong positive short-term momentum")
        elif current['Momentum_5'] < -0.02:
            print("• Strong negative short-term momentum")
        else:
            print("• Neutral short-term momentum")
            
        if current['Price_vs_MA20'] > 5:
            print("• Trading significantly above 20-day moving average")
        elif current['Price_vs_MA20'] < -5:
            print("• Trading significantly below 20-day moving average")
            
        print("\n💡 Trading Insights:")
        if current['RSI'] < 30 and current['Momentum_5'] < 0:
            print("• Potential oversold bounce opportunity")
        elif current['RSI'] > 70 and current['Momentum_5'] > 0:
            print("• Consider taking profits - momentum may be exhausting")
        
        volatility_level = current['Volatility']
        if volatility_level > 0.4:
            print("• High volatility environment - expect larger price swings")
        elif volatility_level < 0.2:
            print("• Low volatility environment - expect smaller price movements")
            
        print("\n⚠️ Risk Assessment:")
        risk_score = 0
        if current['RSI'] > 80 or current['RSI'] < 20:
            risk_score += 2
        if abs(current['Price_vs_MA20']) > 10:
            risk_score += 1
        if volatility_level > 0.5:
            risk_score += 1
            
        if risk_score >= 3:
            print("• HIGH RISK: Multiple warning indicators present")
        elif risk_score >= 2:
            print("• MEDIUM RISK: Some caution advised")
        else:
            print("• MODERATE RISK: Normal market conditions")
            
        return result
        
    except Exception as e:
        print(f"\n❌ Error generating predictions: {str(e)}")
        return None

# ---------------------------
# Get Stock Data
# ---------------------------

def get_stock_info(ticker):
    info = {}
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
    except Exception as e:
        print(f"Error fetching info for {ticker}: {e}")
    return info

def plot_cumulative_returns(data, ticker):
    returns = data["Daily % Change"].dropna() / 100
    cum_returns = (1 + returns).cumprod()
    plt.figure(figsize=(10,6))
    plt.plot(cum_returns.index, cum_returns.values, color='green')
    plt.title(f"{ticker} Cumulative Returns")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_drawdown(data, ticker):
    drawdown, max_drawdown = calculate_max_drawdown(data["Close"])
    # Ensure max_drawdown is a scalar for formatting
    if hasattr(max_drawdown, 'item'):
        max_dd_val = max_drawdown.item()
    else:
        max_dd_val = float(max_drawdown)
    plt.figure(figsize=(10,5))
    plt.plot(drawdown.index, drawdown.values, color='red')
    plt.title(f"{ticker} Drawdown (Max: {max_dd_val:.2%})")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_rolling_volatility(data, ticker, window=21):
    returns = data["Daily % Change"].dropna() / 100
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    plt.figure(figsize=(10,5))
    plt.plot(rolling_vol.index, rolling_vol.values, color='orange')
    plt.title(f"{ticker} Rolling Volatility ({window}-day window)")
    plt.xlabel("Date")
    plt.ylabel("Annualized Volatility")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_histogram(data, ticker):
    returns = data["Daily % Change"].dropna()
    plt.figure(figsize=(8,5))
    plt.hist(returns, bins=30, color='skyblue', edgecolor='black')
    plt.title(f"{ticker} Daily Returns Histogram")
    plt.xlabel("Daily % Change (%)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_interactive_candlestick(data, ticker):
    """Create an interactive candlestick chart with volume using Plotly."""
    # Create subplots with shared x-axis
    fig = make_subplots(rows=2, cols=1, 
                       shared_xaxes=True, 
                       vertical_spacing=0.05,
                       row_heights=[0.7, 0.3],
                       subplot_titles=(f'{ticker} Price', 'Volume'))
    
    # Add candlestick trace
    fig.add_trace(go.Candlestick(x=data.index,
                                open=data['Open'],
                                high=data['High'],
                                low=data['Low'],
                                close=data['Close'],
                                name='Price'),
                 row=1, col=1)
    
    # Add moving averages
    for ma_period, color in [(20, 'orange'), (50, 'green')]:
        ma_col = f'MA{ma_period}'
        if ma_col in data.columns:
            fig.add_trace(go.Scatter(x=data.index,
                                   y=data[ma_col],
                                   name=f'MA{ma_period}',
                                   line=dict(color=color, width=1)),
                         row=1, col=1)
    
    # Add Bollinger Bands if available
    if 'BB_Upper' in data.columns and 'BB_Lower' in data.columns:
        fig.add_trace(go.Scatter(x=data.index, 
                               y=data['BB_Upper'],
                               name='BB Upper',
                               line=dict(color='rgba(255, 0, 0, 0.5)', width=1),
                               showlegend=False),
                     row=1, col=1)
        fig.add_trace(go.Scatter(x=data.index,
                               y=data['BB_Lower'],
                               name='BB Lower',
                               line=dict(color='rgba(255, 0, 0, 0.5)', width=1),
                               fill='tonexty',
                               fillcolor='rgba(255, 0, 0, 0.1)',
                               showlegend=False),
                     row=1, col=1)
    
    # Add volume as colored bars
    colors = ['green' if close >= open_ else 'red' 
              for close, open_ in zip(data['Close'], data['Open'])]
    
    fig.add_trace(go.Bar(x=data.index,
                        y=data['Volume'],
                        name='Volume',
                        marker_color=colors,
                        opacity=0.7),
                 row=2, col=1)
    
    # Update layout
    fig.update_layout(
        title=f'{ticker} Interactive Chart',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        showlegend=True,
        height=800,
        template='plotly_white',
        xaxis2_rangeslider_visible=False,
        hovermode='x unified'
    )
    
    # Add range selector buttons
    fig.update_xaxes(
        rangeslider_visible=False,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label='1M', step='month', stepmode='backward'),
                dict(count=3, label='3M', step='month', stepmode='backward'),
                dict(count=6, label='6M', step='month', stepmode='backward'),
                dict(count=1, label='YTD', step='year', stepmode='todate'),
                dict(count=1, label='1Y', step='year', stepmode='backward'),
                dict(step='all', label='All')
            ])
        )
    )
    
    # Customize volume chart
    fig.update_yaxes(title_text='Volume', row=2, col=1)
    
    # Show the plot
    fig.show()

def backtest_signals(data, initial_cash=10000):
    cash = initial_cash
    position = 0
    buy_price = 0
    trades = []
    for idx, row in data.iterrows():
        signal = row.get('Signal', None)
        close = row['Close']
        if signal and 'BUY' in signal and position == 0:
            position = cash // close
            buy_price = close
            cash -= position * close
            trades.append((idx, 'BUY', close))
        elif signal and 'SELL' in signal and position > 0:
            cash += position * close
            trades.append((idx, 'SELL', close))
            position = 0
    # Final value if holding
    final_value = cash + position * data.iloc[-1]['Close']
    total_return = (final_value - initial_cash) / initial_cash
    num_trades = len(trades)
    print(f"\nBacktest Results:")
    print(f"Initial cash: ${initial_cash:,.2f}")
    print(f"Final value: ${final_value:,.2f}")
    print(f"Total return: {total_return:.2%}")
    print(f"Number of trades: {num_trades}")
    if trades:
        print("Trade log:")
        for t in trades:
            print(f"{t[0].date()} - {t[1]} at ${t[2]:.2f}")
    else:
        print("No trades executed.")

def show_historical_data(data, ticker, days=30):
    """Display historical price data for the given ticker.
    
    Args:
        data: DataFrame containing the stock data
        ticker: Stock ticker symbol
        days: Number of days of historical data to show (None for all)
    """
    print(f"\n=== Historical Price Data for {ticker} ===")
    
    # Select columns to display
    columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if all(col in data.columns for col in columns):
        display_data = data[columns].copy()
        
        # Format numbers for better readability
        pd.options.display.float_format = '{:,.2f}'.format
        
        # Show only the most recent days if specified
        if days is not None and len(display_data) > days:
            display_data = display_data.tail(days)
        
        print(display_data)
        print(f"\nShowing {len(display_data)} of {len(data)} available trading days")
    else:
        print("Error: Could not retrieve complete historical data")

def print_status(message, status_type="info"):
    """Print status messages with consistent formatting"""
    icons = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌",
        "progress": "⏳"
    }
    print(f"\n{icons.get(status_type, ' ')} {message}")

def analyze_stock(ticker, skip_plot=False, custom_stats=False, export_excel=False, info=False, cumulative=False, drawdown=False, rolling_vol=False, backtest=False, show_signals=False, histogram=False, latest=None, sector_info=False, summary_only=False, moving_average_plot=False, signal_summary_only=False, price_targets=False, predictions=False, momentum_analysis=False, market_scan=False, breakout_scan=False, interactive_chart=False, show_history=False, history_days=30, show_graphs=True):
    print(f"\nDEBUG: analyze_stock called with predictions={predictions}")
    if predictions:
        print("DEBUG: Predictions are ENABLED")
    else:
        print("DEBUG: Predictions are DISABLED")
    # Handle market scanning modes first (don't need individual stock data)
    if market_scan:
        scan_market_momentum()
        return
    
    if breakout_scan:
        scan_breakout_stocks()
        return
    
    start_date = datetime.now() - timedelta(days=365)
    end_date = datetime.now()
    print("\n" + "="*60)
    print_status(f"Starting analysis for {ticker}", "info")
    print("="*60)
    
    # Show analysis progress
    print_status("Fetching market data...", "progress")
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            print_status(f"No data found for ticker '{ticker}'", "error")
            return
        print_status(f"Successfully fetched {len(data)} days of data", "success")
    except Exception as e:
        print_status(f"Error fetching data: {str(e)}", "error")
        return

    if data.empty:
        print(f"❌ No data found for ticker '{ticker}'. Please check the symbol and try again.")
        sys.exit(1)

    if sector_info:
        stock_info = get_stock_info(ticker)
        print(f"\n=== {ticker} Sector/Industry Info ===")
        for key in ["sector", "industry", "country", "exchange"]:
            print(f"{key}: {stock_info.get(key, 'N/A')}")

    # Ensure all columns exist and are numeric
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        if col in data.columns and isinstance(data[col], pd.Series):
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # Clean Data
    data = data.ffill().bfill()
    if "Close" in data.columns:
        data["Close"] = data["Close"].interpolate()

    # Calculate Indicators
    data["MA20"] = data["Close"].rolling(20, min_periods=1).mean()
    data["MA50"] = data["Close"].rolling(50, min_periods=1).mean()
    data["RSI"] = calculate_rsi(data["Close"])
    data["MACD"], data["MACD_signal"] = calculate_macd(data["Close"])
    data["Vol_MA15"] = data["Volume"].rolling(15, min_periods=1).mean()
    data["Daily % Change"] = data["Close"].pct_change() * 100
    # Bollinger Bands
    data["BB_MA"], data["BB_Upper"], data["BB_Lower"] = calculate_bollinger_bands(data["Close"])
    # Buy/Sell signals
    data["Signal"] = generate_signals(data)

    # Export full processed data
    print(f"\n📄 Full processed data display only (not saved)")
    if export_excel:
        print(f"📄 Full processed data also displayed (not saved)")

    # Show historical data if requested
    if show_history:
        show_historical_data(data, ticker, days=history_days)
    
    # Display Full Table only if show_history is True and not summary_only
    if show_history and not summary_only:
        pd.set_option("display.max_rows", None)
        print(f"\n=== Full trading data for {ticker} (showing {history_days} days) ===\n")
        # Use history_days instead of latest for historical data display
        display_data = data[["Close", "MA20", "MA50", "BB_Upper", "BB_Lower", "Daily % Change", "Volume", "Vol_MA15", "RSI", "MACD", "MACD_signal", "Signal"]].tail(history_days)
        print(display_data)
        print(f"\n[{len(display_data)} rows x {len(display_data.columns)} columns]")

    # Latest Day Summary
    print("\n" + "📊" * 30)
    print("📈 MOST RECENT TRADING DAY SUMMARY".center(60))
    print("📊" * 30)
    
    latest_day = data.iloc[-1]
    try:
        last_date = latest_day.name.strftime('%Y-%m-%d')
    except (AttributeError, ValueError) as e:
        # Handle case where date formatting fails
        last_date = str(latest_day.name)[:10] if hasattr(latest_day, 'name') else 'Unknown'
    
    # Format price and indicators with colors
    def format_price(price, prev_price=None):
        import pandas as pd
        import numpy as np
        
        # Handle case where price is a pandas Series
        if hasattr(price, 'iloc'):
            price = price.iloc[-1] if len(price) > 0 else price
        
        if prev_price is not None:
            if hasattr(prev_price, 'iloc'):
                prev_price = float(prev_price.iloc[0]) if len(prev_price) > 0 else 0.0
                
            if float(price) > float(prev_price):
                return f"\033[92m${float(price):,.2f} ▲\033[0m"
            elif float(price) < float(prev_price):
                return f"\033[91m${float(price):,.2f} ▼\033[0m"
        return f"${float(price):,.2f}" if not pd.isna(price) else "N/A"
    
    def format_change(pct):
        # Convert to float if it's a pandas Series/DataFrame
        if hasattr(pct, 'iloc'):
            pct = float(pct.iloc[0]) if len(pct) > 0 else 0.0
            
        pct_float = float(pct)
        if pct_float > 0:
            return f"\033[92m+{pct_float:.2f}% ▲\033[0m"
        elif pct_float < 0:
            return f"\033[91m{pct_float:.2f}% ▼\033[0m"
        return f"{pct_float:.2f}%"

    # Get previous day's close for comparison
    prev_close = data['Close'].iloc[-2] if len(data) > 1 else None
    
    # Print key metrics in a clean format
    print(f"\n🔹 Date:           {last_date}")
    print(f"🔹 Ticker:        {ticker.upper()}")
    print(f"🔹 Close:         {format_price(latest_day['Close'], prev_close)}")
    if prev_close is not None:
        daily_change = ((latest_day['Close'] - prev_close) / prev_close) * 100
        print(f"🔹 Daily Change:  {format_change(daily_change)}")
    
    # Technical Indicators
    print("\n📊 TECHNICAL INDICATORS")
    print("-" * 30)
    
    # Helper function to safely get values from Series/DataFrame
    def get_value(data, key):
        val = data[key]
        if hasattr(val, 'iloc') and len(val) > 0:
            extracted_val = val.iloc[0]
            if extracted_val is None or pd.isna(extracted_val):
                return None
            try:
                return float(extracted_val)
            except (ValueError, TypeError):
                return extracted_val  # Return as-is if can't convert to float
        return val if val is not None and pd.notna(val) else None
    
    # Get values safely
    ma20 = get_value(latest_day, 'MA20')
    ma50 = get_value(latest_day, 'MA50')
    rsi = get_value(latest_day, 'RSI')
    
    print(f"• 20-Day MA:     ${ma20:,.2f}")
    print(f"• 50-Day MA:     ${ma50:,.2f}")
    print(f"• RSI (14):      {rsi:.2f} " + 
          ("(Overbought)" if rsi > 70 else 
           "(Oversold)" if rsi < 30 else "(Neutral)"))
    
    # Volume Analysis
    print("\n📈 VOLUME")
    print("-" * 30)
    
    # Get volume values safely
    volume = get_value(latest_day, 'Volume')
    vol_ma15 = get_value(latest_day, 'Vol_MA15')
    volume_ratio = volume / vol_ma15 if vol_ma15 != 0 else 1.0
    
    print(f"• Today's Volume:  {volume:,.0f}")
    print(f"• 15-Day Avg:      {vol_ma15:,.0f}")
    print(f"• Volume Ratio:    {volume_ratio:.2f}x " + 
          ("(Above Average)" if volume_ratio > 1.5 else 
           "(Below Average)" if volume_ratio < 0.75 else "(Average)"))
    
    # Trading Signal
    print("\n🚦 TRADING SIGNAL")
    print("-" * 30)
    signal = get_value(latest_day, 'Signal')
    
    if signal is None:
        print("• HOLD (No signal data)")
    elif signal == 'BUY':
        print("\033[92m• STRONG BUY SIGNAL\033[0m")
    elif signal == 'SELL':
        print("\033[91m• STRONG SELL SIGNAL\033[0m")
    else:
        print("• HOLD (No strong signal)")
    
    print("\n" + "📊" * 30)
    
    # Summary Statistics
    print(f"\n📊 SUMMARY STATISTICS - {ticker.upper()}")
    print("-" * 60)
    if custom_stats:
        print("\nEnter custom date range for statistics:")
        start_str = input("Start date (YYYY-MM-DD): ")
        end_str = input("End date (YYYY-MM-DD): ")
        try:
            custom_start = pd.to_datetime(start_str)
            custom_end = pd.to_datetime(end_str)
            stats_data = data.loc[(data.index >= custom_start) & (data.index <= custom_end)]
        except Exception:
            print("Invalid date format. Using full range.")
            stats_data = data
    else:
        stats_data = data

    summary = pd.DataFrame({
        "Highest Close": [stats_data["Close"].max()],
        "Lowest Close": [stats_data["Close"].min()],
        "Average Close": [stats_data["Close"].mean()],
        "Highest % Gain": [stats_data["Daily % Change"].max()],
        "Highest % Loss": [stats_data["Daily % Change"].min()],
        "Highest Volume": [stats_data["Volume"].max()],
        "Lowest Volume": [stats_data["Volume"].min()],
        "Average Volume": [stats_data["Volume"].mean()]
    })
    print(summary)
    
    # Calculate buy/sell signals first (needed for email report)
    buy_signals = data["Signal"].dropna().str.contains("BUY").sum()
    sell_signals = data["Signal"].dropna().str.contains("SELL").sum()
    
    print(f"\n📄 Summary displayed only (not saved)")
    if export_excel:
        print(f"📄 Summary also displayed (not saved)")

    # Sharpe Ratio and Volatility
    daily_returns = data["Daily % Change"].dropna() / 100
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else np.nan
    volatility = daily_returns.std() * np.sqrt(252)
    print(f"\n📈 Sharpe Ratio (annualized): {sharpe_ratio:.3f}")
    print(f"📉 Volatility (annualized): {volatility:.3f}")

    # Top 5 biggest daily gains and losses
    print("\nTop 5 biggest daily gains:")
    print(data.nlargest(5, "Daily % Change")[["Daily % Change", "Close"]])
    print("\nTop 5 biggest daily losses:")
    print(data.nsmallest(5, "Daily % Change")[["Daily % Change", "Close"]])

    # Buy/Sell signal summary
    print(f"\nSignal summary for {ticker}:")
    print(f"Total BUY signals: {buy_signals}")
    print(f"Total SELL signals: {sell_signals}")
    
    # Plot charts if not skipped (moved before early return)
    print(f"\nDEBUG: skip_plot={skip_plot}, show_graphs={show_graphs}")
    if not skip_plot and show_graphs:
        try:
            print("\n📊 Creating analysis dashboard...")
            
            # Check if we can display plots
            import matplotlib
            backend = matplotlib.get_backend()
            print(f"Current backend: {backend}")
            
            # Ask how many days of chart data to display
            while True:
                try:
                    chart_days_input = input("   How many days to display in charts? (1-365, default 90): ").strip()
                    if not chart_days_input:  # Default to 90 days if no input
                        chart_days = 90
                        break
                    chart_days = int(chart_days_input)
                    if 1 <= chart_days <= 365:
                        break
                    print("   Please enter a number between 1 and 365.")
                except (ValueError, KeyboardInterrupt):
                    chart_days = 90
                    break
            
            # Filter data for the selected number of days
            chart_data = data.tail(chart_days) if len(data) > chart_days else data
            
            # Create separate windows for each chart
            figures = []
            
            # 1. Price + Moving Averages + Bollinger Bands
            fig1 = plt.figure(figsize=(12, 6))
            plt.plot(chart_data.index, chart_data["Close"].values, label="Close Price", color="blue")
            plt.plot(chart_data.index, chart_data["MA20"].values, label="MA20", color="orange")
            plt.plot(chart_data.index, chart_data["MA50"].values, label="MA50", color="green")
            plt.plot(chart_data.index, chart_data["BB_Upper"].values, label="BB Upper", color="magenta", linestyle="--")
            plt.plot(chart_data.index, chart_data["BB_Lower"].values, label="BB Lower", color="magenta", linestyle="--")
            plt.title(f"{ticker} - Price, Moving Averages & Bollinger Bands")
            plt.xlabel("Date")
            plt.ylabel("Price")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            figures.append(fig1)
            
            # 2. RSI
            fig2 = plt.figure(figsize=(12, 4))
            plt.plot(chart_data.index, chart_data["RSI"].values, label="RSI", color="purple")
            plt.axhline(70, color="red", linestyle="--", label="Overbought (70)")
            plt.axhline(30, color="green", linestyle="--", label="Oversold (30)")
            plt.title(f"{ticker} - RSI")
            plt.xlabel("Date")
            plt.ylabel("RSI")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            figures.append(fig2)
            
            # 3. MACD
            fig3 = plt.figure(figsize=(12, 4))
            plt.plot(chart_data.index, chart_data["MACD"], label="MACD", color="blue")
            plt.plot(chart_data.index, chart_data["MACD_signal"], label="Signal", color="red")
            plt.bar(chart_data.index, chart_data["MACD"] - chart_data["MACD_signal"], 
                   label="MACD Histogram", color="gray", alpha=0.3)
            plt.axhline(0, color="black", linestyle="-", linewidth=0.5)
            plt.title(f"{ticker} - MACD")
            plt.xlabel("Date")
            plt.ylabel("MACD")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            figures.append(fig3)
            
            # 4. Volume
            fig4 = plt.figure(figsize=(12, 6))
            # Handle volume data - check if it's a MultiIndex column
            if isinstance(chart_data.columns, pd.MultiIndex):
                volume_col = chart_data["Volume"].iloc[:, 0] if len(chart_data["Volume"].columns) > 0 else chart_data["Volume"]
                vol_ma15_col = chart_data["Vol_MA15"] if "Vol_MA15" in chart_data.columns else None
            else:
                volume_col = chart_data["Volume"]
                vol_ma15_col = chart_data["Vol_MA15"] if "Vol_MA15" in chart_data.columns else None
            
            plt.bar(chart_data.index, volume_col.astype(float), color="skyblue", label="Volume")
            if vol_ma15_col is not None:
                plt.plot(chart_data.index, vol_ma15_col.astype(float), color="red", label="Vol MA15")
            plt.title(f"{ticker} - Daily Trading Volume")
            plt.xlabel("Date")
            plt.ylabel("Volume")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            figures.append(fig4)
            
            if show_graphs and not skip_plot and figures:
                print_status("Preparing interactive charts...", "progress")
                print(f"   Using backend: {matplotlib.get_backend()}")
                print("   Close all chart windows to continue...")
                
                try:
                    # Show all figures
                    for i, fig in enumerate(figures, 1):
                        print_status(f"Displaying chart {i} of {len(figures)}...", "progress")
                        plt.figure(fig.number)
                        plt.show(block=False)
                    
                    # Keep the plots open until user closes them
                    print_status("All charts displayed. Close windows to continue...", "info")
                    plt.show(block=True)
                    print_status("Charts closed. Analysis complete!", "success")
                    
                except Exception as e:
                    print_status(f"Error displaying charts: {str(e)}", "error")
                    import traceback
                    traceback.print_exc()
            
        except Exception as e:
            print(f"❌ Error during plotting: {e}")
            import traceback
            traceback.print_exc()
    
    if signal_summary_only and not any([price_targets, predictions, momentum_analysis]):
        return
    
    # Up/Down day insight
    up_days = (data["Daily % Change"] > 0).sum()
    down_days = (data["Daily % Change"] < 0).sum()
    total_days = up_days + down_days
    up_pct = (up_days / total_days * 100) if total_days > 0 else 0
    down_pct = (down_days / total_days * 100) if total_days > 0 else 0
    print(f"\n{ticker} Up/Down Day Insight:")
    print(f"Up days: {up_days} ({up_pct:.1f}%)")
    print(f"Down days: {down_days} ({down_pct:.1f}%)")

    if show_signals:
        print(f"\nRows with BUY/SELL signals:")
        print(data[data["Signal"].notna()][["Close", "MA20", "MA50", "RSI", "MACD", "MACD_signal", "Signal"]])

    if backtest:
        backtest_signals(data)

    if price_targets:
        calculate_price_targets(data, ticker)
    
    print(f"\nDEBUG: predictions={predictions}, type={type(predictions)}")
    if predictions:
        try:
            print("\n" + "="*60)
            print("🔮 GENERATING PRICE PREDICTIONS")
            print("="*60)
            result = predict_stock_prices(data, ticker)
            # The function now handles all the display internally
            print("\n" + "-"*60 + "\n")
        except Exception as e:
            print("\n" + "!"*60)
            print("❌ ERROR in predict_stock_prices:")
            print(str(e))
            print("\nStack trace:")
            import traceback
            traceback.print_exc()
            print("!"*60 + "\n")
    
    if momentum_analysis:
        momentum_data = calculate_momentum_score(data, ticker)
        display_momentum_analysis(momentum_data)

def send_email_report(to_email, report_contents):
    # SMTP configuration (example: Gmail)
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    sender_email = "your_email@gmail.com"  # <-- Replace with your email
    sender_password = "your_app_password"   # <-- Replace with your app password
    subject = "Stock Analysis Report"
    body = "\n\n".join(report_contents)
    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, to_email, msg.as_string())
        server.quit()
        print(f"\n✅ Email report sent to {to_email}")
    except Exception as e:
        print(f"\n❌ Failed to send email: {e}")

def send_sms_report(to_phone, report_contents):
    """Send SMS report using Twilio API"""
    if not TWILIO_AVAILABLE:
        print("\n❌ Twilio not installed. Install with: pip install twilio")
        return
    
    # Twilio configuration - Replace with your credentials
    account_sid = "your_twilio_account_sid"  # <-- Replace with your Twilio Account SID
    auth_token = "your_twilio_auth_token"    # <-- Replace with your Twilio Auth Token
    from_phone = "your_twilio_phone_number"  # <-- Replace with your Twilio phone number
    
    try:
        client = Client(account_sid, auth_token)
        
        # Truncate message if too long (SMS limit is 1600 chars)
        body = "\n\n".join(report_contents)
        if len(body) > 1500:
            body = body[:1500] + "...\n[Report truncated for SMS]"
        
        message = client.messages.create(
            body=f"📊 Stock Analysis Report\n\n{body}",
            from_=from_phone,
            to=to_phone
        )
        print(f"\n✅ SMS report sent to {to_phone} (Message SID: {message.sid})")
    except Exception as e:
        print(f"\n❌ Failed to send SMS: {e}")

def main():
    # Initialize notification variables at the start of main
    email_report = False
    sms_report = False
    email_to = None
    sms_to = None
    
    parser = argparse.ArgumentParser(description="Stock Analysis Tool")
    parser.add_argument("-t", "--ticker", type=str, help="Stock ticker symbol (e.g. AAPL, TSLA, MSFT)")
    parser.add_argument("--batch", type=str, help="Comma-separated list of tickers for batch analysis (e.g. AAPL,TSLA,MSFT)")
    parser.add_argument("--batch-summary-only", action="store_true", help="Show only summary statistics and signal summary for each ticker in batch mode")
    parser.add_argument("--skip-plot", action="store_true", help="Skip plotting charts")
    parser.add_argument("--custom-stats", action="store_true", help="Show statistics for custom date range interactively")
    parser.add_argument("--export-excel", action="store_true", help="Export summary and full data as Excel (.xlsx)")
    parser.add_argument("--info", action="store_true", help="Show sector/industry and other info")
    parser.add_argument("--cumulative", action="store_true", help="Plot cumulative returns chart")
    parser.add_argument("--drawdown", action="store_true", help="Plot drawdown chart and show max drawdown")
    parser.add_argument("--rolling-vol", action="store_true", help="Plot rolling volatility chart")
    parser.add_argument("--backtest", action="store_true", help="Run backtest on buy/sell signals")
    parser.add_argument("--show-signals", action="store_true", help="Display only rows with buy/sell signals after analysis")
    parser.add_argument("--histogram", action="store_true", help="Show histogram of daily returns")
    parser.add_argument("--latest", type=int, help="Show only the latest N days of data after analysis")
    parser.add_argument("--explain", action="store_true", help="Show a quick explanation of what this script does")
    parser.add_argument("--sector-info", action="store_true", help="Show sector, industry, and country info for the chosen stock")
    parser.add_argument("--summary-only", action="store_true", help="Show only summary statistics and skip full table")
    parser.add_argument("--signal-summary-only", action="store_true", help="Show only the buy/sell signal summary and skip other analytics")
    parser.add_argument("--price-targets", action="store_true", help="Calculate and display price targets based on historical patterns")
    parser.add_argument("--predictions", action="store_true", default=False, help="Generate future price predictions using Linear Regression")
    parser.add_argument("--momentum", action="store_true", help="Perform comprehensive momentum analysis with scoring")
    parser.add_argument("--scan-market", action="store_true", help="Scan market for high momentum stocks automatically")
    parser.add_argument("--scan-breakouts", action="store_true", help="Scan market for stocks breaking out to new highs")
    parser.add_argument("--mlb-predict", action="store_true", help="Predict MLB player stats for the next game")
    parser.add_argument("--email-report", action="store_true", help="Email the analysis report")
    parser.add_argument("--email-to", type=str, help="Email address to send report to")
    parser.add_argument("--interactive", action="store_true", help="Show interactive candlestick chart with volume")
    args = parser.parse_args()

    if args.explain:
        print("""
This script fetches 1 year of daily price data for any stock ticker, calculates technical indicators, generates buy/sell signals, prints analytics, and shows interactive charts. Use CLI flags to customize analysis. No files are saved; all results are shown in the terminal and as interactive plots.

Key features:
- Choose any stock with --ticker or interactively
- Show only latest N days with --latest N
- Show sector/industry info with --sector-info
- Show summary only with --summary-only
- Show histogram of returns with --histogram
- Show buy/sell signals with --show-signals
- Run backtest with --backtest
- Show cumulative, drawdown, volatility charts
- Calculate price targets with --price-targets
- Generate price predictions with --predictions
- Analyze momentum with --momentum
- Scan market for high momentum stocks with --scan-market
- Find breakout stocks with --scan-breakouts
""")
        return

    if args.batch:
        tickers = [t.strip().upper() for t in args.batch.split(",") if t.strip()]
        print(f"\n=== Batch Analysis Mode: {', '.join(tickers)} ===")
        for ticker in tickers:
            print(f"\n=== Batch Analysis: {ticker} ===")
            analyze_stock(
                ticker,
                skip_plot=True,  # Always skip plots in batch mode
                custom_stats=False,
                export_excel=False,
                info=args.info,
                cumulative=False,
                drawdown=False,
                rolling_vol=False,
                backtest=False,
                show_signals=False,
                histogram=False,
                latest=args.latest,
                sector_info=args.sector_info,
                summary_only=args.batch_summary_only,
                signal_summary_only=args.batch_summary_only,
                price_targets=args.price_targets,
                predictions=args.predictions,
                momentum_analysis=args.momentum
            )
        return

    # Handle market scanning modes first (don't need ticker input)
    if args.scan_market:
        analyze_stock("", market_scan=True)
        return
    
    if args.scan_breakouts:
        analyze_stock("", breakout_scan=True)
        return
    
    if args.mlb_predict:
        # Load MLB data (replace 'mlb_data.csv' with your actual file path)
        mlb_data_file = 'mlb_data.csv'  # You can also make this a command-line argument
        if not os.path.exists(mlb_data_file):
            print(f"Error: {mlb_data_file} not found.  Please create this file with MLB data.")
            return
            
        mlb_data = load_mlb_data(mlb_data_file)
        if mlb_data is None:
            return

        # Preprocess the data
        X, y = preprocess_data(mlb_data)

        # Train the model
        model = train_model(X, y)

        # Example usage (you'll need to provide the actual player data)
        # ... (rest of the MLB prediction code, see previous response)
        print("\nMLB Prediction functionality integrated. Provide mlb_data.csv")
        return


    # Interactive prompt for tickers if neither --ticker nor --batch is provided
    ticker = args.ticker
    price_targets_choice = args.price_targets
    
    if not ticker:
        try:
            ticker_input = input("Enter one or more stock tickers (comma-separated, e.g. AAPL, TSLA, MSFT): ").strip()
            tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
        except (EOFError, KeyboardInterrupt):
            print("\nUsing default ticker: AAPL")
            tickers = ["AAPL"]
        
        # Ask for price targets if not specified via command line
        if not price_targets_choice:
            try:
                price_targets_input = input("Do you want price target predictions? (y/n): ").strip().lower()
                price_targets_choice = price_targets_input == 'y'
            except (EOFError, KeyboardInterrupt):
                price_targets_choice = False
        
        # Ask for predictions if not specified via command line
        predictions_choice = args.predictions
        if not predictions_choice:
            try:
                predictions_input = input("Do you want future price predictions? (y/n): ").strip().lower()
                predictions_choice = predictions_input == 'y'
            except (EOFError, KeyboardInterrupt):
                predictions_choice = False
        
        # Ask for momentum analysis if not specified via command line
        momentum_choice = args.momentum
        if not momentum_choice:
            try:
                momentum_input = input("Do you want momentum analysis? (y/n): ").strip().lower()
                momentum_choice = momentum_input == 'y'
            except (EOFError, KeyboardInterrupt):
                momentum_choice = False
        
        if len(tickers) > 1:
            print(f"\n=== Batch Analysis Mode: {', '.join(tickers)} ===")
            for ticker in tickers:
                print(f"\n=== Batch Analysis: {ticker} ===")
                analyze_stock(
                    ticker,
                    skip_plot=True,
                    custom_stats=False,
                    export_excel=False,
                    info=args.info,
                    cumulative=False,
                    drawdown=False,
                    rolling_vol=False,
                    backtest=False,
                    show_signals=False,
                    histogram=False,
                    latest=args.latest,
                    show_history=False,  # Don't show history in batch mode
                    sector_info=args.sector_info,
                    summary_only=args.summary_only,
                    signal_summary_only=args.summary_only,
                    price_targets=price_targets_choice,
                    predictions=predictions_choice,
                    momentum_analysis=momentum_choice
                )
            return
        elif len(tickers) == 1:
            ticker = tickers[0]
        else:
            print("No tickers entered. Exiting.")
            return
    else:
        ticker = ticker.upper()
        # Use command-line arguments if provided, otherwise ask interactively
        price_targets_choice = args.price_targets
        predictions_choice = args.predictions
        momentum_choice = args.momentum
        
        # Only prompt if no analysis options were specified
        if not any([args.price_targets, args.predictions, args.momentum]):
            if not price_targets_choice:
                try:
                    price_targets_input = input("Do you want price target predictions? (y/n): ").strip().lower()
                    price_targets_choice = price_targets_input == 'y'
                except (EOFError, KeyboardInterrupt):
                    price_targets_choice = False
            
            if not predictions_choice:
                try:
                    predictions_input = input("Do you want future price predictions? (y/n): ").strip().lower()
                    predictions_choice = predictions_input == 'y'
                except (EOFError, KeyboardInterrupt):
                    predictions_choice = False
            
            if not momentum_choice:
                try:
                    momentum_input = input("Do you want momentum analysis? (y/n): ").strip().lower()
                    momentum_choice = momentum_input == 'y'
                except (EOFError, KeyboardInterrupt):
                    momentum_choice = False

    print("\n" + "="*50)
    print("📊  ANALYSIS OPTIONS")
    print("="*50)
    
    # Initialize display options
    show_graphs = True
    show_history = False
    days_to_show = 30  # Default number of days to show
    show_full_data = False
    
    if not hasattr(args, 'skip_plot') or not args.skip_plot:
        print("\n📈 Display Options:")
        try:
            show_graphs_input = input("   Show interactive charts? [Y/n]: ").lower().strip()
            show_graphs = show_graphs_input in ['y', '']
        except (EOFError, KeyboardInterrupt):
            show_graphs = True
            
        # Always ask about historical data display
        try:
            show_history_input = input("   Show historical price data table? [Y/n]: ").strip().lower()
            show_history = show_history_input in ['y', '']
        except (EOFError, KeyboardInterrupt):
            show_history = True
            
        if show_history:
            while True:
                try:
                    days_input = input("   How many days of history to display? (1-365, default 30): ").strip()
                    if not days_input:  # Default to 30 days if no input
                        days_to_show = 30
                        break
                    try:
                        days_to_show = int(days_input)
                        if 1 <= days_to_show <= 365:
                            break
                        print("   Please enter a number between 1 and 365.")
                    except ValueError:
                        print("   Please enter a valid number.")
                except (EOFError, KeyboardInterrupt):
                    days_to_show = 30
                    break
        
        # Ask about full technical analysis table
        try:
            show_full_data_input = input("   Show full technical analysis table? [y/N]: ").strip().lower()
            show_full_data = show_full_data_input in ['y', 'yes']
        except (EOFError, KeyboardInterrupt):
            show_full_data = False
    
    args.summary_only = not show_full_data
    args.signal_summary_only = not show_full_data

    # Notification options - skip if any analysis options are specified
    if not any([args.price_targets, args.predictions, args.momentum]):
        if not hasattr(args, 'batch') or not args.batch:
            print("\n🔔 Notification Options:")
            
            # Email notification
            if hasattr(args, 'email_report') and args.email_report:
                email_report = True
                email_to = args.email_to if hasattr(args, 'email_to') and args.email_to else None
                print("   ✓ Email notifications enabled")
            else:
                try:
                    email_choice = input("   Send results via email? [y/N]: ").strip().lower()
                    if email_choice in ['y', 'yes']:
                        email_to = input("   Enter your email address: ").strip()
                        email_report = True
                except (EOFError, KeyboardInterrupt):
                    email_report = False
                    email_to = None
            
            # SMS notification (only if email not already set)
            if not email_report or not hasattr(args, 'email_report'):
                try:
                    sms_choice = input("   Send results via SMS? [y/N]: ").strip().lower()
                    if sms_choice in ['y', 'yes']:
                        sms_to = input("   Enter phone number (with country code, e.g., +1234567890): ").strip()
                        sms_report = True
                except (EOFError, KeyboardInterrupt):
                    sms_report = False
                    sms_to = None
    
    # Handle latest days display - use command line arg or default to all
    latest = args.latest
    # Only prompt for latest days if no analysis options are specified and not in batch mode
    if (latest is None and 
        not any([args.price_targets, args.predictions, args.momentum]) and 
        not hasattr(args, 'batch')):
        while True:
            try:
                latest_input = input("\nEnter number of latest days to display (or press Enter for all): ").strip()
                if not latest_input:  # If user just presses Enter
                    latest = None
                    break
                latest = int(latest_input)
                if latest > 0:
                    break
                print("Please enter a positive number or press Enter for all data.")
            except (ValueError, EOFError, KeyboardInterrupt):
                if latest_input == "":
                    latest = None
                    break
                print("Please enter a valid number or press Enter for all data.")

    # Prepare report summary
    report_summary = [f"Stock Analysis Report for {ticker}"]
    
    # Send email if requested
    if email_report and email_to:
        try:
            send_email_report(email_to, report_summary)
        except Exception as e:
            print(f"Error sending email: {e}")
    
    # Send SMS if requested
    if sms_report and sms_to:
        try:
            send_sms_report(sms_to, report_summary)
        except Exception as e:
            print(f"Error sending SMS: {e}")
            # If SMS fails, try to fall back to email if we have an email address
            if email_to and not email_report:  # Only try fallback if we haven't already sent an email
                try:
                    send_email_report(email_to, ["SMS delivery failed. Sending via email instead."] + report_summary)
                except Exception as email_err:
                    print(f"Error sending fallback email: {email_err}")
    
    # Call analyze_stock with the appropriate parameters
    analyze_stock(
        ticker=ticker,
        skip_plot=not show_graphs,
        custom_stats=args.custom_stats if hasattr(args, 'custom_stats') else False,
        show_history=show_history,
        history_days=days_to_show,
        show_graphs=show_graphs,
        predictions=predictions_choice,
        price_targets=price_targets_choice,
        momentum_analysis=momentum_choice
    )

if __name__ == "__main__":
    main()
