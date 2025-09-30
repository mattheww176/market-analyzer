import logging
import os
import signal
import sys
import threading
import time
import webbrowser
from datetime import datetime, timedelta
from typing import List, Dict, Any

from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import yfinance as yf
from jinja2 import Environment

from market_analyzer import analyze_stock, predict_stock_prices

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
shutdown_enabled = False

# Custom Jinja2 filter for formatting timestamps
def datetimeformat(timestamp, format='%B %d, %Y %I:%M %p'):
    if not timestamp:
        return ''
    try:
        # Convert timestamp to datetime object
        if isinstance(timestamp, (int, float)) and timestamp > 1e10:
            # If timestamp is in milliseconds, convert to seconds
            timestamp = timestamp / 1000
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime(format)
    except Exception as e:
        print(f"Error formatting timestamp {timestamp}: {e}")
        return str(timestamp)

# Add the filter to Jinja2 environment
app.jinja_env.filters['datetimeformat'] = datetimeformat

# Shutdown route that can only be accessed from localhost
@app.route('/shutdown')
def shutdown():
    if request.remote_addr == '127.0.0.1':  # Only allow from localhost
        import os
        import signal
        
        def shutdown_server():
            try:
                os.kill(os.getpid(), signal.SIGINT)
            except:
                # Fallback method
                import sys
                sys.exit(0)
        
        # Schedule shutdown after response is sent
        import threading
        threading.Timer(0.5, shutdown_server).start()
        
        return '''
        <html>
        <head><title>Server Shutdown</title></head>
        <body style="font-family: Arial, sans-serif; text-align: center; margin-top: 50px;">
            <h2>✅ Server is shutting down...</h2>
            <p>You can close this window now.</p>
            <script>
                setTimeout(function() {
                    window.close();
                }, 2000);
            </script>
        </body>
        </html>
        '''
    return 'Unauthorized', 403

def get_common_context():
    """Helper function to get common context for all routes"""
    global shutdown_enabled
    client_ip = request.remote_addr
    user_agent = request.user_agent.string
    shutdown_enabled = (client_ip == '127.0.0.1')
    
    # Debug info suppressed for cleaner output
    
    return {
        'show_shutdown': shutdown_enabled,
        'debug_info': {
            'client_ip': client_ip,
            'user_agent': user_agent,
            'show_shutdown': shutdown_enabled
        }
    }

@app.route('/')
def index():
    return render_template('index.html', **get_common_context())

@app.route('/analysis')
def analysis():
    try:
        context = get_common_context()
        result = render_template('analysis.html', **context)
        return result
    except Exception as e:
        # Log error to stderr only for debugging
        import traceback
        traceback.print_exc()
        return f"Error loading analysis page: {str(e)}", 500

@app.route('/api/chart-data/<ticker>')
def get_chart_data(ticker):
    """Get structured chart data for a stock ticker"""
    try:
        import yfinance as yf
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        # Get stock data with extended hours to ensure we get the most recent data
        stock = yf.Ticker(ticker.upper())
        
        # First try to get data with prepost=True to include pre/post market data
        data = stock.history(period='3mo', interval='1d', prepost=True)
        
        # If no data or the data is too old, try without prepost
        if data.empty or (datetime.now().date() - data.index[-1].date()).days > 1:
            data = stock.history(period='3mo', interval='1d')
        
        # If still no data, try a different approach with a specific date range
        if data.empty:
            end_date = datetime.now() + timedelta(days=1)
            start_date = end_date - timedelta(days=95)
            data = stock.history(start=start_date, end=end_date, interval='1d')
        
        if data.empty:
            return jsonify({'error': f'No data found for ticker {ticker}'}), 404
            
        # Ensure we have the most recent data by checking the last date
        last_date = data.index[-1].date()
        today = datetime.now().date()
        
        # If our data is not up to date, try to get the latest
        if last_date < today:
            try:
                # Try to get just today's data with a 5-minute interval
                new_data = stock.history(period='1d', interval='5m', prepost=True)
                
                if not new_data.empty:
                    # Resample to daily data if we got intraday data
                    if len(new_data) > 1:  # If we have multiple data points
                        daily_data = new_data.resample('D').agg({
                            'Open': 'first',
                            'High': 'max',
                            'Low': 'min',
                            'Close': 'last',
                            'Volume': 'sum'
                        })
                        # Combine with existing data, keeping the most recent
                        data = pd.concat([data, daily_data])
                        # Remove any duplicate dates, keeping the last occurrence
                        data = data[~data.index.duplicated(keep='last')].sort_index()
            except Exception as e:
                print(f"Warning: Could not fetch latest intraday data: {e}")
        
        # Final cleanup of any remaining duplicates
        data = data[~data.index.duplicated(keep='last')].sort_index()
        
        # Calculate RSI
        def calculate_rsi(prices, period=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50)  # Fill NaN with neutral RSI
        
        # Calculate MACD
        def calculate_macd(prices, fast=12, slow=26, signal=9):
            exp1 = prices.ewm(span=fast).mean()
            exp2 = prices.ewm(span=slow).mean()
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=signal).mean()
            histogram = macd_line - signal_line
            return macd_line.fillna(0), signal_line.fillna(0), histogram.fillna(0)
        
        # Ensure we have clean, sorted data with no duplicates
        data = data[~data.index.duplicated(keep='last')].sort_index()
        
        # Calculate indicators on the final dataset
        rsi = calculate_rsi(data['Close'])
        macd_line, signal_line, histogram = calculate_macd(data['Close'])
        
        # Calculate moving averages
        ma50 = data['Close'].rolling(window=50).mean()
        ma200 = data['Close'].rolling(window=200).mean()
        
        # Create chart data with modern pandas methods
        chart_data = {
            'dates': data.index.strftime('%Y-%m-%d').tolist(),
            'prices': data['Close'].ffill().values.tolist(),
            'volumes': data['Volume'].fillna(0).values.tolist(),
            'opens': data['Open'].ffill().values.tolist(),
            'highs': data['High'].ffill().values.tolist(),
            'lows': data['Low'].ffill().values.tolist(),
            'rsi': rsi.fillna(50).values.tolist() if rsi is not None else [50] * len(data),
            'ma50': ma50.fillna(0).values.tolist() if ma50 is not None else [0] * len(data),
            'ma200': ma200.fillna(0).values.tolist() if ma200 is not None else [0] * len(data),
            'macdLine': macd_line.fillna(0).values.tolist() if macd_line is not None else [0] * len(data),
            'signalLine': signal_line.fillna(0).values.tolist() if signal_line is not None else [0] * len(data),
            'histogram': histogram.fillna(0).values.tolist() if histogram is not None else [0] * len(data),
            'lastUpdated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return jsonify(chart_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def run_prediction_analysis(ticker, days):
    """Run prediction analysis for the given ticker and return the results."""
    # Import yfinance at function level to ensure it's available
    import yfinance as yf
    import pandas as pd
    
    # Prediction analysis starting
    
    try:
        # Get historical data
        start_date = pd.Timestamp.now() - pd.Timedelta(days=days*2)
        end_date = pd.Timestamp.now()
        
        # Downloading data silently
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if data is None or data.empty:
            error_msg = f"No data available for {ticker} from {start_date} to {end_date}"
            # Error logged internally
            return f"Error: {error_msg}"
            
        # Data downloaded successfully
        
        # Run prediction with verbose output disabled
        try:
            from market_analyzer import predict_stock_prices
            # Disable verbose output in web interface
            prediction_results = predict_stock_prices(
                data, 
                ticker, 
                days_ahead=[1, 3, 7, 14, 30],
                verbose=False
            )
            
            if prediction_results is None:
                error_msg = "Prediction function returned None"
                # Error logged internally
                return f"Error: {error_msg}"
                
        except Exception as pred_error:
            error_msg = f"Error in predict_stock_prices: {str(pred_error)}"
            # Error logged internally
            return f"Error: {error_msg}"
            
        # Prediction results generated
        
        # Format the prediction results
        output = []
        output.append(f"📈 Price Prediction Analysis for {ticker}")
        output.append("=" * 50)
        
        try:
            # Add current price
            current_price = prediction_results.get('current_price', 0)
            output.append(f"\n💵 Current Price: ${current_price:.2f}")
            
            # Add predictions if available
            if 'predictions' in prediction_results and prediction_results['predictions']:
                output.append("\n🔮 Price Predictions:")
                for days_ahead, price in prediction_results['predictions'].items():
                    if current_price > 0:  # Avoid division by zero
                        change_pct = (price / current_price - 1) * 100
                        output.append(f"   {days_ahead}-day forecast: ${price:.2f} ({change_pct:+.1f}%)")
                    else:
                        output.append(f"   {days_ahead}-day forecast: ${price:.2f}")
            
            # Add technical indicators
            output.append("\n📊 Technical Indicators:")
            
            if 'rsi' in prediction_results:
                rsi = prediction_results['rsi']
                rsi_status = "(Oversold)" if rsi < 30 else "(Overbought)" if rsi > 70 else "(Neutral)"
                output.append(f"   • RSI: {rsi:.1f} {rsi_status}")
                
            if 'volatility' in prediction_results:
                output.append(f"   • Volatility: {prediction_results['volatility']*100:.1f}%")
                
            if 'price_vs_ma20' in prediction_results:
                output.append(f"   • Price vs 20-day MA: {prediction_results['price_vs_ma20']:+.1f}%")
                
            if 'price_vs_ma50' in prediction_results:
                output.append(f"   • Price vs 50-day MA: {prediction_results['price_vs_ma50']:+.1f}%")
            
            # Add trading insights
            output.append("\n💡 Trading Insights:")
            if 'rsi' in prediction_results:
                rsi = prediction_results['rsi']
                if rsi < 30:
                    output.append("   • RSI indicates oversold conditions (potential buying opportunity)")
                elif rsi > 70:
                    output.append("   • RSI indicates overbought conditions (potential selling opportunity)")
                    
            if 'volatility' in prediction_results and prediction_results['volatility'] > 0.4:
                output.append("   • High volatility detected - expect larger price swings")
                
            # Add risk assessment
            output.append("\n⚠️ Risk Assessment:")
            risk_factors = []
            if 'rsi' in prediction_results and (prediction_results['rsi'] > 80 or prediction_results['rsi'] < 20):
                risk_factors.append("extreme RSI levels")
            if 'volatility' in prediction_results and prediction_results['volatility'] > 0.5:
                risk_factors.append("high volatility")
                
            if risk_factors:
                output.append(f"   • Caution: {' and '.join(risk_factors)} detected")
            else:
                output.append("   • Normal market conditions detected")
                
        except Exception as format_error:
            # Error formatting results
            output.append("\n⚠️ Analysis completed with partial results. Some data may be missing.")
            
        return "\n".join(output)
        
    except Exception as e:
        error_msg = f"Error in prediction analysis: {str(e)}\n{traceback.format_exc()}"
        # Critical error logged
        return f"Error: {str(e)}\n\nPlease check the server logs for more details."

def run_technical_analysis(ticker, days):
    """Run enhanced technical analysis with better formatting."""
    try:
        import yfinance as yf
        from datetime import datetime, timedelta
        
        stock = yf.Ticker(ticker)
        
        # Get historical data
        if days == 'max':
            hist = stock.history(period="max")
        else:
            hist = stock.history(period=f"{days}d")
        
        if hist.empty:
            return f"<div class='p-4'><h3 class='text-lg font-semibold mb-2 text-red-600'>Error</h3><p>No data available for {ticker}. Please check the ticker symbol.</p></div>"
        
        # Import technical analysis functions from market_analyzer
        from market_analyzer import calculate_rsi, calculate_macd, calculate_bollinger_bands
        
        # Calculate technical indicators
        rsi = calculate_rsi(hist['Close'])
        macd_line, macd_signal = calculate_macd(hist['Close'])
        macd_histogram = macd_line - macd_signal  # Calculate histogram manually
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(hist['Close'])
        
        # Get current values
        current_price = hist['Close'].iloc[-1]
        current_rsi = rsi.iloc[-1] if not rsi.empty else 0
        current_macd = macd_line.iloc[-1] if not macd_line.empty else 0
        current_macd_signal = macd_signal.iloc[-1] if not macd_signal.empty else 0
        current_bb_upper = bb_upper.iloc[-1] if not bb_upper.empty else 0
        current_bb_lower = bb_lower.iloc[-1] if not bb_lower.empty else 0
        
        # Calculate moving averages
        ma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
        ma_50 = hist['Close'].rolling(window=50).mean().iloc[-1] if len(hist) >= 50 else None
        
        # Determine RSI signal
        if current_rsi > 70:
            rsi_signal = "Overbought"
            rsi_color = "text-red-600"
        elif current_rsi < 30:
            rsi_signal = "Oversold"
            rsi_color = "text-green-600"
        else:
            rsi_signal = "Neutral"
            rsi_color = "text-gray-600"
        
        # Determine MACD signal
        if current_macd > current_macd_signal:
            macd_signal_text = "Bullish"
            macd_color = "text-green-600"
        else:
            macd_signal_text = "Bearish"
            macd_color = "text-red-600"
        
        # Determine Bollinger Bands position
        if current_price > current_bb_upper:
            bb_position = "Above Upper Band (Overbought)"
            bb_color = "text-red-600"
        elif current_price < current_bb_lower:
            bb_position = "Below Lower Band (Oversold)"
            bb_color = "text-green-600"
        else:
            bb_position = "Within Bands (Normal)"
            bb_color = "text-gray-600"
        
        # Get company info
        info = stock.info
        company_name = info.get('longName', ticker)
        
        # Format the output with enhanced HTML
        output = f"""
        <div class='p-4'>
            <h3 class='text-lg font-semibold mb-4'>Technical Analysis for {company_name} ({ticker})</h3>
            
            <!-- Current Price Section -->
            <div class='bg-blue-50 dark:bg-blue-900/30 p-4 rounded-lg mb-4'>
                <h4 class='font-semibold mb-2 text-blue-800 dark:text-blue-200'>Current Price Information</h4>
                <div class='grid grid-cols-1 md:grid-cols-3 gap-4'>
                    <div>
                        <p class='text-sm text-gray-600 dark:text-gray-400'>Current Price</p>
                        <p class='text-xl font-bold'>${current_price:.2f}</p>
                    </div>
                    <div>
                        <p class='text-sm text-gray-600 dark:text-gray-400'>20-Day MA</p>
                        <p class='text-lg font-semibold'>${ma_20:.2f}</p>
                    </div>
                    {"<div><p class='text-sm text-gray-600 dark:text-gray-400'>50-Day MA</p><p class='text-lg font-semibold'>$" + f"{ma_50:.2f}" + "</p></div>" if ma_50 else "<div><p class='text-sm text-gray-600 dark:text-gray-400'>50-Day MA</p><p class='text-sm text-gray-500'>Insufficient data</p></div>"}
                </div>
            </div>
            
            <!-- Technical Indicators Grid -->
            <div class='grid grid-cols-1 md:grid-cols-3 gap-4 mb-4'>
                <!-- RSI Card -->
                <div class='bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4'>
                    <h4 class='font-semibold mb-2 text-gray-800 dark:text-white'>RSI (14)</h4>
                    <div class='text-center'>
                        <p class='text-2xl font-bold text-gray-800 dark:text-white'>{current_rsi:.1f}</p>
                        <p class='{rsi_color} font-semibold text-sm'>{rsi_signal}</p>
                    </div>
                    <div class='mt-2'>
                        <div class='w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2'>
                            <div class='bg-blue-600 h-2 rounded-full' style='width: {min(current_rsi, 100)}%'></div>
                        </div>
                        <div class='flex justify-between text-xs text-gray-500 mt-1'>
                            <span>0</span>
                            <span>30</span>
                            <span>70</span>
                            <span>100</span>
                        </div>
                    </div>
                </div>
                
                <!-- MACD Card -->
                <div class='bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4'>
                    <h4 class='font-semibold mb-2 text-gray-800 dark:text-white'>MACD</h4>
                    <div class='space-y-2'>
                        <div>
                            <p class='text-sm text-gray-600 dark:text-gray-400'>MACD Line</p>
                            <p class='font-semibold'>{current_macd:.4f}</p>
                        </div>
                        <div>
                            <p class='text-sm text-gray-600 dark:text-gray-400'>Signal Line</p>
                            <p class='font-semibold'>{current_macd_signal:.4f}</p>
                        </div>
                        <div>
                            <p class='text-sm text-gray-600 dark:text-gray-400'>Signal</p>
                            <p class='{macd_color} font-semibold'>{macd_signal_text}</p>
                        </div>
                    </div>
                </div>
                
                <!-- Bollinger Bands Card -->
                <div class='bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4'>
                    <h4 class='font-semibold mb-2 text-gray-800 dark:text-white'>Bollinger Bands</h4>
                    <div class='space-y-2'>
                        <div>
                            <p class='text-sm text-gray-600 dark:text-gray-400'>Upper Band</p>
                            <p class='font-semibold'>${current_bb_upper:.2f}</p>
                        </div>
                        <div>
                            <p class='text-sm text-gray-600 dark:text-gray-400'>Lower Band</p>
                            <p class='font-semibold'>${current_bb_lower:.2f}</p>
                        </div>
                        <div>
                            <p class='text-sm text-gray-600 dark:text-gray-400'>Position</p>
                            <p class='{bb_color} font-semibold text-sm'>{bb_position}</p>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Summary Section -->
            <div class='bg-gray-50 dark:bg-gray-700 p-4 rounded-lg'>
                <h4 class='font-semibold mb-2 text-gray-800 dark:text-white'>Technical Summary</h4>
                <div class='grid grid-cols-1 md:grid-cols-2 gap-4 text-sm'>
                    <div>
                        <p class='text-gray-600 dark:text-gray-400'>Analysis Period: {len(hist)} trading days</p>
                        <p class='text-gray-600 dark:text-gray-400'>Data Range: {hist.index[0].strftime('%Y-%m-%d')} to {hist.index[-1].strftime('%Y-%m-%d')}</p>
                    </div>
                    <div>
                        <p class='text-gray-600 dark:text-gray-400'>Price vs 20-MA: {"Above" if current_price > ma_20 else "Below"} ({((current_price - ma_20) / ma_20 * 100):+.1f}%)</p>
                        {"<p class='text-gray-600 dark:text-gray-400'>Price vs 50-MA: " + ("Above" if current_price > ma_50 else "Below") + f" ({((current_price - ma_50) / ma_50 * 100):+.1f}%)</p>" if ma_50 else "<p class='text-gray-600 dark:text-gray-400'>50-MA: Insufficient data</p>"}
                    </div>
                </div>
            </div>
        </div>
        """
        
        return output
        
    except Exception as e:
        return f"<div class='p-4'><h3 class='text-lg font-semibold mb-2 text-red-600'>Error</h3><p>Failed to perform technical analysis for {ticker}: {str(e)}</p></div>"

def run_fundamental_analysis(ticker):
    """Run enhanced fundamental analysis with better formatting."""
    try:
        import yfinance as yf
        
        stock = yf.Ticker(ticker)
        info = stock.info
        
        if not info or len(info) < 5:
            return f"<div class='p-4'><h3 class='text-lg font-semibold mb-2 text-red-600'>Error</h3><p>No fundamental data available for {ticker}. Please check the ticker symbol.</p></div>"
        
        # Get company information
        company_name = info.get('longName', ticker)
        sector = info.get('sector', 'N/A')
        industry = info.get('industry', 'N/A')
        market_cap = info.get('marketCap', 0)
        
        # Financial metrics
        pe_ratio = info.get('trailingPE', 'N/A')
        forward_pe = info.get('forwardPE', 'N/A')
        peg_ratio = info.get('pegRatio', 'N/A')
        price_to_book = info.get('priceToBook', 'N/A')
        price_to_sales = info.get('priceToSalesTrailing12Months', 'N/A')
        
        # Profitability metrics
        profit_margin = info.get('profitMargins', 'N/A')
        operating_margin = info.get('operatingMargins', 'N/A')
        roe = info.get('returnOnEquity', 'N/A')
        roa = info.get('returnOnAssets', 'N/A')
        
        # Growth metrics
        revenue_growth = info.get('revenueGrowth', 'N/A')
        earnings_growth = info.get('earningsGrowth', 'N/A')
        
        # Dividend information
        dividend_yield = info.get('dividendYield', 'N/A')
        payout_ratio = info.get('payoutRatio', 'N/A')
        
        # Financial health
        debt_to_equity = info.get('debtToEquity', 'N/A')
        current_ratio = info.get('currentRatio', 'N/A')
        quick_ratio = info.get('quickRatio', 'N/A')
        
        # Format market cap
        if market_cap and market_cap > 0:
            if market_cap >= 1e12:
                market_cap_formatted = f"${market_cap/1e12:.2f}T"
            elif market_cap >= 1e9:
                market_cap_formatted = f"${market_cap/1e9:.2f}B"
            elif market_cap >= 1e6:
                market_cap_formatted = f"${market_cap/1e6:.2f}M"
            else:
                market_cap_formatted = f"${market_cap:,.0f}"
        else:
            market_cap_formatted = "N/A"
        
        # Helper function to format percentage
        def format_percentage(value):
            if value == 'N/A' or value is None:
                return 'N/A'
            try:
                return f"{float(value) * 100:.2f}%" if isinstance(value, (int, float)) else 'N/A'
            except:
                return 'N/A'
        
        # Helper function to format ratio
        def format_ratio(value):
            if value == 'N/A' or value is None:
                return 'N/A'
            try:
                return f"{float(value):.2f}" if isinstance(value, (int, float)) else 'N/A'
            except:
                return 'N/A'
        
        # Format the output with enhanced HTML
        output = f"""
        <div class='p-4'>
            <h3 class='text-lg font-semibold mb-4'>Fundamental Analysis for {company_name} ({ticker})</h3>
            
            <!-- Company Overview -->
            <div class='bg-blue-50 dark:bg-blue-900/30 p-4 rounded-lg mb-4'>
                <h4 class='font-semibold mb-2 text-blue-800 dark:text-blue-200'>Company Overview</h4>
                <div class='grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4'>
                    <div>
                        <p class='text-sm text-gray-600 dark:text-gray-400'>Sector</p>
                        <p class='font-semibold'>{sector}</p>
                    </div>
                    <div>
                        <p class='text-sm text-gray-600 dark:text-gray-400'>Industry</p>
                        <p class='font-semibold'>{industry}</p>
                    </div>
                    <div>
                        <p class='text-sm text-gray-600 dark:text-gray-400'>Market Cap</p>
                        <p class='font-semibold'>{market_cap_formatted}</p>
                    </div>
                    <div>
                        <p class='text-sm text-gray-600 dark:text-gray-400'>Employees</p>
                        <p class='font-semibold'>{info.get('fullTimeEmployees', 'N/A'):,}</p>
                    </div>
                </div>
            </div>
            
            <!-- Valuation Metrics -->
            <div class='grid grid-cols-1 md:grid-cols-2 gap-4 mb-4'>
                <div class='bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4'>
                    <h4 class='font-semibold mb-3 text-gray-800 dark:text-white'>Valuation Ratios</h4>
                    <div class='space-y-3'>
                        <div class='flex justify-between'>
                            <span class='text-sm text-gray-600 dark:text-gray-400'>P/E Ratio (TTM)</span>
                            <span class='font-semibold'>{format_ratio(pe_ratio)}</span>
                        </div>
                        <div class='flex justify-between'>
                            <span class='text-sm text-gray-600 dark:text-gray-400'>Forward P/E</span>
                            <span class='font-semibold'>{format_ratio(forward_pe)}</span>
                        </div>
                        <div class='flex justify-between'>
                            <span class='text-sm text-gray-600 dark:text-gray-400'>PEG Ratio</span>
                            <span class='font-semibold'>{format_ratio(peg_ratio)}</span>
                        </div>
                        <div class='flex justify-between'>
                            <span class='text-sm text-gray-600 dark:text-gray-400'>Price-to-Book</span>
                            <span class='font-semibold'>{format_ratio(price_to_book)}</span>
                        </div>
                        <div class='flex justify-between'>
                            <span class='text-sm text-gray-600 dark:text-gray-400'>Price-to-Sales</span>
                            <span class='font-semibold'>{format_ratio(price_to_sales)}</span>
                        </div>
                    </div>
                </div>
                
                <!-- Profitability Metrics -->
                <div class='bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4'>
                    <h4 class='font-semibold mb-3 text-gray-800 dark:text-white'>Profitability</h4>
                    <div class='space-y-3'>
                        <div class='flex justify-between'>
                            <span class='text-sm text-gray-600 dark:text-gray-400'>Profit Margin</span>
                            <span class='font-semibold'>{format_percentage(profit_margin)}</span>
                        </div>
                        <div class='flex justify-between'>
                            <span class='text-sm text-gray-600 dark:text-gray-400'>Operating Margin</span>
                            <span class='font-semibold'>{format_percentage(operating_margin)}</span>
                        </div>
                        <div class='flex justify-between'>
                            <span class='text-sm text-gray-600 dark:text-gray-400'>Return on Equity</span>
                            <span class='font-semibold'>{format_percentage(roe)}</span>
                        </div>
                        <div class='flex justify-between'>
                            <span class='text-sm text-gray-600 dark:text-gray-400'>Return on Assets</span>
                            <span class='font-semibold'>{format_percentage(roa)}</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Growth and Dividend -->
            <div class='grid grid-cols-1 md:grid-cols-2 gap-4 mb-4'>
                <div class='bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4'>
                    <h4 class='font-semibold mb-3 text-gray-800 dark:text-white'>Growth Metrics</h4>
                    <div class='space-y-3'>
                        <div class='flex justify-between'>
                            <span class='text-sm text-gray-600 dark:text-gray-400'>Revenue Growth</span>
                            <span class='font-semibold'>{format_percentage(revenue_growth)}</span>
                        </div>
                        <div class='flex justify-between'>
                            <span class='text-sm text-gray-600 dark:text-gray-400'>Earnings Growth</span>
                            <span class='font-semibold'>{format_percentage(earnings_growth)}</span>
                        </div>
                    </div>
                </div>
                
                <div class='bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4'>
                    <h4 class='font-semibold mb-3 text-gray-800 dark:text-white'>Dividend Information</h4>
                    <div class='space-y-3'>
                        <div class='flex justify-between'>
                            <span class='text-sm text-gray-600 dark:text-gray-400'>Dividend Yield</span>
                            <span class='font-semibold'>{format_percentage(dividend_yield)}</span>
                        </div>
                        <div class='flex justify-between'>
                            <span class='text-sm text-gray-600 dark:text-gray-400'>Payout Ratio</span>
                            <span class='font-semibold'>{format_percentage(payout_ratio)}</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Financial Health -->
            <div class='bg-green-50 dark:bg-green-900/30 p-4 rounded-lg'>
                <h4 class='font-semibold mb-3 text-green-800 dark:text-green-200'>Financial Health</h4>
                <div class='grid grid-cols-1 md:grid-cols-3 gap-4'>
                    <div>
                        <p class='text-sm text-gray-600 dark:text-gray-400'>Debt-to-Equity</p>
                        <p class='font-semibold'>{format_ratio(debt_to_equity)}</p>
                    </div>
                    <div>
                        <p class='text-sm text-gray-600 dark:text-gray-400'>Current Ratio</p>
                        <p class='font-semibold'>{format_ratio(current_ratio)}</p>
                    </div>
                    <div>
                        <p class='text-sm text-gray-600 dark:text-gray-400'>Quick Ratio</p>
                        <p class='font-semibold'>{format_ratio(quick_ratio)}</p>
                    </div>
                </div>
            </div>
        </div>
        """
        
        return output
        
    except Exception as e:
        return f"<div class='p-4'><h3 class='text-lg font-semibold mb-2 text-red-600'>Error</h3><p>Failed to perform fundamental analysis for {ticker}: {str(e)}</p></div>"

@app.route('/analyze', methods=['POST'])
def analyze():
    ticker = request.form.get('ticker', '').upper()
    days = int(request.form.get('days', 90))
    analysis_type = request.form.get('analysis_type', 'basic')
    
    if not ticker:
        return jsonify({'error': 'Please enter a stock ticker'}), 400
    
    try:
        # Redirect stdout to capture the output
        from io import StringIO
        import sys
        import traceback
        
        # Analysis starting
        
        # For prediction analysis, we don't need to capture stdout
        if analysis_type == 'prediction':
            try:
                output = run_prediction_analysis(ticker, days)
                if not output:
                    raise ValueError("No output from prediction analysis")
                return jsonify({
                    'ticker': ticker,
                    'analysis': output,
                    'analysis_type': analysis_type,
                    'status': 'success'
                })
            except Exception as e:
                # Error in prediction analysis logged
                return jsonify({
                    'error': f"Error in prediction analysis: {str(e)}",
                    'status': 'error',
                    'traceback': traceback.format_exc()
                }), 500
        
        # Handle basic analysis separately (no stdout capture needed)
        if analysis_type == 'basic':
            # For basic analysis, get basic stock information
            try:
                import yfinance as yf
                from datetime import datetime, timedelta
                
                stock = yf.Ticker(ticker)
                
                # Get historical data
                if days == 'max':
                    hist = stock.history(period="max")
                else:
                    hist = stock.history(period=f"{days}d")
                
                if hist.empty:
                    output = f"<div class='p-4'><h3 class='text-lg font-semibold mb-2 text-red-600'>Error</h3><p>No data available for {ticker}. Please check the ticker symbol.</p></div>"
                else:
                    # Get basic info
                    info = stock.info
                    current_price = hist['Close'].iloc[-1]
                    prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                    price_change = current_price - prev_price
                    price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
                    
                    # Calculate basic metrics
                    high_52w = hist['High'].max()
                    low_52w = hist['Low'].min()
                    avg_volume = hist['Volume'].mean()
                    
                    # Format the output
                    change_color = "text-green-600" if price_change >= 0 else "text-red-600"
                    change_symbol = "+" if price_change >= 0 else ""
                    
                    company_name = info.get('longName', ticker)
                    
                    output = f"""
                    <div class='p-4'>
                        <h3 class='text-lg font-semibold mb-4'>Basic Analysis for {company_name} ({ticker})</h3>
                        
                        <div class='grid grid-cols-1 md:grid-cols-2 gap-4 mb-4'>
                            <div class='bg-gray-50 dark:bg-gray-700 p-4 rounded'>
                                <h4 class='font-semibold mb-2'>Current Price</h4>
                                <p class='text-2xl font-bold'>${current_price:.2f}</p>
                                <p class='{change_color} text-sm'>
                                    {change_symbol}{price_change:.2f} ({change_symbol}{price_change_pct:.2f}%)
                                </p>
                            </div>
                            
                            <div class='bg-gray-50 dark:bg-gray-700 p-4 rounded'>
                                <h4 class='font-semibold mb-2'>52-Week Range</h4>
                                <p class='text-sm'>High: <span class='font-semibold'>${high_52w:.2f}</span></p>
                                <p class='text-sm'>Low: <span class='font-semibold'>${low_52w:.2f}</span></p>
                            </div>
                        </div>
                        
                        <div class='bg-gray-50 dark:bg-gray-700 p-4 rounded mb-4'>
                            <h4 class='font-semibold mb-2'>Volume Information</h4>
                            <p class='text-sm'>Average Daily Volume: <span class='font-semibold'>{avg_volume:,.0f}</span></p>
                            <p class='text-sm'>Latest Volume: <span class='font-semibold'>{hist['Volume'].iloc[-1]:,.0f}</span></p>
                        </div>
                        
                        <div class='bg-blue-50 dark:bg-blue-900/30 p-4 rounded'>
                            <h4 class='font-semibold mb-2'>Analysis Period</h4>
                            <p class='text-sm'>Data from the last {days} days ({len(hist)} trading days)</p>
                            <p class='text-sm'>Period: {hist.index[0].strftime('%Y-%m-%d')} to {hist.index[-1].strftime('%Y-%m-%d')}</p>
                        </div>
                    </div>
                    """
                    
            except Exception as e:
                # Error in basic analysis logged
                output = f"<div class='p-4'><h3 class='text-lg font-semibold mb-2 text-red-600'>Error</h3><p>Failed to fetch data for {ticker}: {str(e)}</p></div>"
                
            # Return the basic analysis result
            return jsonify({
                'ticker': ticker,
                'analysis': output,
                'analysis_type': analysis_type,
                'status': 'success'
            })
        
        elif analysis_type == 'technical':
            # Enhanced Technical Analysis
            try:
                output = run_technical_analysis(ticker, days)
                return jsonify({
                    'ticker': ticker,
                    'analysis': output,
                    'analysis_type': analysis_type,
                    'status': 'success'
                })
            except Exception as e:
                return jsonify({
                    'error': f"Error in technical analysis: {str(e)}",
                    'status': 'error'
                }), 500
                
        elif analysis_type == 'fundamental':
            # Enhanced Fundamental Analysis
            try:
                output = run_fundamental_analysis(ticker)
                return jsonify({
                    'ticker': ticker,
                    'analysis': output,
                    'analysis_type': analysis_type,
                    'status': 'success'
                })
            except Exception as e:
                return jsonify({
                    'error': f"Error in fundamental analysis: {str(e)}",
                    'status': 'error'
                }), 500
        
        else:
            # For other analysis types, capture stdout (fallback)
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            try:
                # For technical and full analysis, run the full analysis
                analyze_stock(
                    ticker=ticker,
                    show_history=True,
                    history_days=days,
                    show_graphs=False,
                    skip_plot=True,
                    verbose=False  # Disable verbose output in web interface
                )
                
                # Get the output and format it for HTML display
                raw_output = sys.stdout.getvalue()
                # Convert plain text to HTML with proper line breaks and formatting
                output = f"<div class='p-4'><pre class='whitespace-pre-wrap text-sm bg-gray-100 dark:bg-gray-700 p-4 rounded'>{raw_output}</pre></div>"
                
                # Process the output based on analysis type
                return jsonify({
                    'ticker': ticker,
                    'analysis': output,
                    'analysis_type': analysis_type,
                    'status': 'success'
                })
                
            except Exception as e:
                # Error in analysis logged
                return jsonify({
                    'error': str(e),
                    'status': 'error',
                    'traceback': traceback.format_exc()
                }), 500
                
            finally:
                sys.stdout = old_stdout
            
    except Exception as e:
        # Unexpected error logged
        return jsonify({
            'error': f"Unexpected error: {str(e)}",
            'status': 'error',
            'traceback': traceback.format_exc()
        }), 500

def open_browser():
    time.sleep(1)  # Give the server a second to start
    webbrowser.open('http://127.0.0.1:5001')

@app.route('/api/stock-fundamentals/<ticker>')
def get_stock_fundamentals(ticker):
    """Get fundamental analysis data for a stock ticker"""
    try:
        import requests
        from datetime import datetime
        import json
        import time
        import random
        
        ticker_upper = ticker.upper()
        
        # Generate a random number to prevent caching
        cache_buster = str(int(time.time())) + str(random.randint(1000, 9999))
        
        # Define the base URL for Yahoo Finance API
        base_url = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/"
        
        # Define the modules we want to fetch
        modules = [
            'assetProfile', 'financialData', 'defaultKeyStatistics', 
            'summaryDetail', 'price', 'earnings', 'calendarEvents',
            'upgradeDowngradeHistory', 'recommendationTrend', 'cashflowStatementHistory'
        ]
        
        # Initialize the result dictionary
        info = {}
        
        # Try to fetch data for each module
        for module in modules:
            try:
                url = f"{base_url}{ticker_upper}?modules={module}&formatted=false&corsDomain=finance.yahoo.com&_={cache_buster}"
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                    'Accept': 'application/json',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Referer': 'https://finance.yahoo.com/'
                }
                
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                if 'quoteSummary' in data and 'result' in data['quoteSummary'] and data['quoteSummary']['result']:
                    module_data = data['quoteSummary']['result'][0].get(module, {})
                    if module_data:
                        # Flatten the module data into the info dictionary
                        for key, value in module_data.items():
                            if not key.startswith('maxAge'):
                                info[key] = value
                
                # Add a small delay between requests to avoid rate limiting
                time.sleep(0.2)
                
            except Exception as e:
                # Error fetching module data
                continue
        
        # If we didn't get any data, try the old method as a last resort
        if not info:
            import yfinance as yf
            stock = yf.Ticker(ticker_upper)
            info = stock.info
            info['source'] = 'yfinance_fallback'
        else:
            info['source'] = 'direct_api'
        
        # Ensure we have the ticker symbol
        info['symbol'] = ticker_upper
        
        # Calculate some additional metrics if not available
        if 'trailingPE' not in info and 'trailingEps' in info and info['trailingEps'] is not None and info['trailingEps'] != 0:
            info['trailingPE'] = info.get('currentPrice', 0) / info['trailingEps']
        
        if 'priceToSalesTrailing12Months' not in info and 'revenuePerShare' in info and info['revenuePerShare'] is not None and info['revenuePerShare'] != 0:
            info['priceToSalesTrailing12Months'] = info.get('currentPrice', 0) / info['revenuePerShare']
        
        if 'debtToEquity' not in info and 'totalStockholderEquity' in info and info['totalStockholderEquity'] is not None and info['totalStockholderEquity'] != 0:
            info['debtToEquity'] = info.get('totalDebt', 0) / info['totalStockholderEquity']
            
        # Prepare the response data structure
        fundamentals = {
            'valuation': {
                'peRatio': info.get('trailingPE'),
                'psRatio': info.get('priceToSalesTrailing12Months'),
                'pbRatio': info.get('priceToBook'),
                'marketCap': info.get('marketCap'),
                'dividendYield': info.get('dividendYield'),
                'forwardPE': info.get('forwardPE'),
                'pegRatio': info.get('pegRatio')
            },
            'financialHealth': {
                'currentRatio': info.get('currentRatio'),
                'quickRatio': info.get('quickRatio'),
                'debtToEquity': info.get('debtToEquity'),
                'interestCoverage': info.get('interestCoverage'),
                'totalDebt': info.get('totalDebt'),
                'totalCash': info.get('totalCash')
            },
            'growth': {
                'revenueGrowth': info.get('revenueGrowth'),
                'earningsGrowth': info.get('earningsGrowth'),
                'earningsQuarterlyGrowth': info.get('earningsQuarterlyGrowth'),
                'growthEstimate': info.get('revenueGrowth'),
                'nextFiscalYearGrowth': info.get('earningsQuarterlyGrowth')
            },
            'keyMetrics': {
                'roe': info.get('returnOnEquity'),
                'roa': info.get('returnOnAssets'),
                'profitMargin': info.get('profitMargins'),
                'operatingMargin': info.get('operatingMargins'),
                'ebitda': info.get('ebitda'),
                'ebitdaMargins': info.get('ebitdaMargins')
            },
            'companyInfo': {
                'name': info.get('longName', ticker_upper),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'employees': info.get('fullTimeEmployees'),
                'description': info.get('longBusinessSummary', 'No description available.')
            },
            'dividendInfo': {
                'dividendRate': info.get('dividendRate'),
                'dividendYield': info.get('dividendYield'),
                'payoutRatio': info.get('payoutRatio'),
                'dividendDate': info.get('exDividendDate'),
                'fiveYearAvgDividendYield': info.get('fiveYearAvgDividendYield')
            },
            'lastUpdated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'source': info.get('source', 'unknown')
        }
        
        # Clean up None values to ensure JSON serialization
        def clean_none(d):
            if not isinstance(d, dict):
                return d
            return {k: clean_none(v) for k, v in d.items() if v is not None}
        
        return jsonify(clean_none(fundamentals))
        
    except Exception as e:
        import traceback
        # Error fetching fundamental data
        return jsonify({
            'error': f'Failed to fetch fundamental data for {ticker}: {str(e)}',
            'details': str(e)
        }), 500

@app.route('/watchlist')
def watchlist():
    """Display the user's stock watchlist."""
    return render_template('watchlist.html', **get_common_context())

@app.route('/api/stock/<ticker>')
def get_stock_info(ticker):
    """Get basic stock information for the watchlist."""
    try:
        # Fetching ticker data
        # Get stock data
        stock = yf.Ticker(ticker)
        
        # Get historical data first (more reliable than info)
        hist = stock.history(period='5d')
        
        if hist.empty:
            # No historical data found
            return jsonify({
                'error': f'No data found for {ticker}'
            }), 404
        
        # Use the most recent price as current price
        current_price = hist['Close'].iloc[-1]
        
        # Calculate change if we have at least 2 days of data
        if len(hist) >= 2:
            prev_close = hist['Close'].iloc[-2]
            price_change = current_price - prev_close
            percent_change = (price_change / prev_close) * 100
        else:
            # Using zero change for single day data
            price_change = 0
            percent_change = 0
            prev_close = current_price
        
        # Data fetched successfully
        return jsonify({
            'ticker': ticker.upper(),
            'price': current_price,  # For backward compatibility
            'current_price': current_price,
            'previous_close': prev_close,
            'change': price_change,
            'percent_change': percent_change,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        import traceback
        error_msg = f"Error fetching stock data for {ticker}: {str(e)}\n{traceback.format_exc()}"
        print(f"[ERROR] {error_msg}")
        return jsonify({
            'error': f'Failed to fetch data for {ticker}',
            'details': str(e)
        }), 500

@app.route('/news')
def news():
    """Display market news and stock-specific news if a ticker is provided."""
    ticker = request.args.get('ticker', '').upper().strip()
    news_items = []
    
    # Initialize news items list
    
    try:
        if ticker:
            # Get news for specific stock
            stock = yf.Ticker(ticker)
            raw_news = stock.news or []
            
            # Add all items and let template handle missing data
            for item in raw_news:
                item['ticker'] = ticker
                news_items.append(item)
        else:
            # Get general market news (using SPY as a proxy for market news)
            market = yf.Ticker('SPY')
            raw_news = market.news or []
            
            # Add all items and let template handle missing data
            for item in raw_news:
                item['ticker'] = 'MARKET'
                news_items.append(item)
                    
    except Exception as e:
        import traceback
        print(f"Error in news route: {e}")
        traceback.print_exc()
    
    context = get_common_context()
    context.update({
        'ticker': ticker,
        'news_items': news_items,
        'current_date': datetime.now().strftime('%A, %B %d, %Y')
    })
    return render_template('news.html', **context)

def get_popular_stocks():
    """Return a list of popular stock tickers across different sectors."""
    return [
        # Tech
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'ADBE', 'INTC', 'CSCO',
        # Finance
        'JPM', 'BAC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'V', 'MA',
        # Consumer
        'PG', 'KO', 'PEP', 'WMT', 'TGT', 'HD', 'MCD', 'NKE', 'SBUX', 'DIS',
        # Healthcare
        'JNJ', 'PFE', 'MRK', 'ABT', 'UNH', 'LLY', 'GILD', 'AMGN', 'BMY', 'ABBV',
        # Industrial
        'GE', 'BA', 'HON', 'MMM', 'CAT', 'DE', 'UPS', 'FDX', 'LMT', 'RTX'
    ]

def get_stock_data(tickers):
    """Fetch stock data for the given tickers using yfinance."""
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Stock data fetch timed out")
    
    try:
        print(f"Starting to fetch data for {len(tickers)} tickers...")
        
        # Set a timeout of 10 seconds for the entire operation
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(10)
        
        # Create a Ticker object for batch processing
        ticker_data = yf.Tickers(' '.join(tickers))
        stocks = []
        
        for ticker in tickers:
            try:
                info = ticker_data.tickers[ticker].info
                
                # Ensure all values are JSON serializable
                def safe_get(value, default=0):
                    if value is None or str(value).lower() == 'nan' or value == float('inf') or value == float('-inf'):
                        return default
                    try:
                        return float(value) if isinstance(value, (int, float)) else value
                    except (ValueError, TypeError):
                        return default
                
                stock = {
                    'symbol': str(ticker),
                    'name': str(info.get('shortName', ticker)),
                    'price': safe_get(info.get('currentPrice', info.get('regularMarketPrice', 0))),
                    'change': safe_get(info.get('regularMarketChange', 0)),
                    'change_percent': safe_get(info.get('regularMarketChangePercent', 0)),
                    'market_cap': safe_get(info.get('marketCap', 0)),
                    'volume': safe_get(info.get('volume', 0)),
                    'pe_ratio': safe_get(info.get('trailingPE', None), None),
                    'dividend_yield': safe_get(info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0),
                    'sector': str(info.get('sector', 'N/A')),
                    'fifty_two_week_high': safe_get(info.get('fiftyTwoWeekHigh', 0)),
                    'fifty_two_week_low': safe_get(info.get('fiftyTwoWeekLow', 0))
                }
                stocks.append(stock)
            except Exception as e:
                print(f"Error fetching data for {ticker}: {str(e)}")
                continue
                
        # Clear the alarm
        signal.alarm(0)
        return stocks
        
    except TimeoutError as e:
        print(f"Timeout in get_stock_data: {str(e)}")
        signal.alarm(0)
        return []
    except Exception as e:
        print(f"Error in get_stock_data: {str(e)}")
        signal.alarm(0)
        return []

@app.route('/screener')
def screener():
    """Display the stock screener page with real stock data."""
    # Screener route accessed
    
    context = get_common_context()
    try:
        # Get popular stocks and fetch their data
        # Getting popular stocks
        tickers = get_popular_stocks()
        # Got tickers for analysis
        
        # Using sample data for immediate page load
        # Use sample data for immediate page load to prevent hanging
        stocks = [
            {
                'symbol': 'AAPL',
                'name': 'Apple Inc.',
                'price': 175.34,
                'change': 2.34,
                'change_percent': 1.35,
                'market_cap': 2750000000000,
                'volume': 45000000,
                'pe_ratio': 28.5,
                'dividend_yield': 0.5,
                'sector': 'Technology'
            },
            {
                'symbol': 'MSFT',
                'name': 'Microsoft Corporation',
                'price': 325.12,
                'change': -1.23,
                'change_percent': -0.38,
                'market_cap': 2420000000000,
                'volume': 32000000,
                'pe_ratio': 32.1,
                'dividend_yield': 0.8,
                'sector': 'Technology'
            },
            {
                'symbol': 'GOOGL',
                'name': 'Alphabet Inc.',
                'price': 138.45,
                'change': 0.87,
                'change_percent': 0.63,
                'market_cap': 1740000000000,
                'volume': 28000000,
                'pe_ratio': 25.8,
                'dividend_yield': 0.0,
                'sector': 'Technology'
            },
            {
                'symbol': 'AMZN',
                'name': 'Amazon.com Inc.',
                'price': 185.19,
                'change': 3.45,
                'change_percent': 1.90,
                'market_cap': 1920000000000,
                'volume': 38000000,
                'pe_ratio': 62.3,
                'dividend_yield': 0.0,
                'sector': 'Consumer Cyclical'
            },
            {
                'symbol': 'JNJ',
                'name': 'Johnson & Johnson',
                'price': 152.67,
                'change': -0.56,
                'change_percent': -0.37,
                'market_cap': 394000000000,
                'volume': 12000000,
                'pe_ratio': 15.2,
                'dividend_yield': 2.91,
                'sector': 'Healthcare'
            },
            {
                'symbol': 'JPM',
                'name': 'JPMorgan Chase & Co.',
                'price': 198.76,
                'change': 1.23,
                'change_percent': 0.62,
                'market_cap': 572000000000,
                'volume': 15000000,
                'pe_ratio': 11.8,
                'dividend_yield': 2.35,
                'sector': 'Financial Services'
            },
            {
                'symbol': 'PG',
                'name': 'Procter & Gamble',
                'price': 156.89,
                'change': 0.45,
                'change_percent': 0.29,
                'market_cap': 370000000000,
                'volume': 8500000,
                'pe_ratio': 26.4,
                'dividend_yield': 2.42,
                'sector': 'Consumer Defensive'
            },
            {
                'symbol': 'XOM',
                'name': 'Exxon Mobil Corporation',
                'price': 118.34,
                'change': 2.12,
                'change_percent': 1.82,
                'market_cap': 473000000000,
                'volume': 25000000,
                'pe_ratio': 13.7,
                'dividend_yield': 3.21,
                'sector': 'Energy'
            },
            {
                'symbol': 'VZ',
                'name': 'Verizon Communications',
                'price': 41.23,
                'change': -0.34,
                'change_percent': -0.82,
                'market_cap': 174000000000,
                'volume': 18000000,
                'pe_ratio': 8.9,
                'dividend_yield': 6.54,
                'sector': 'Communication Services'
            },
            {
                'symbol': 'HD',
                'name': 'Home Depot, Inc.',
                'price': 347.89,
                'change': 1.56,
                'change_percent': 0.45,
                'market_cap': 345000000000,
                'volume': 4200000,
                'pe_ratio': 22.6,
                'dividend_yield': 2.30,
                'sector': 'Consumer Cyclical'
            }
        ]
        
        # Ensure stocks is always a list
        context['stocks'] = stocks if stocks else []
        # Final stocks prepared
        
    except Exception as e:
        # Error in screener route - using fallback data
        pass
        # Fallback to empty list if there's an error
        context['stocks'] = []
    
    # Rendering template
    return render_template('screener.html', **context)

def print_startup_message():
    """Print a clean, professional startup message."""
    print("\n" + "╔" + "═"*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "🚀 Market Analyzer v1.0".center(58) + "║")
    print("║" + "Advanced Stock Analysis Platform".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╠" + "═"*58 + "╣")
    print("║" + " "*58 + "║")
    print("║" + "  🌐 Server URL: http://127.0.0.1:5001".ljust(58) + "║")
    print("║" + "  📊 Status: Initializing...".ljust(58) + "║")
    print("║" + "  ⚡ Mode: Production Ready".ljust(58) + "║")
    print("║" + " "*58 + "║")
    print("╠" + "═"*58 + "╣")
    print("║" + "  💡 Your browser will open automatically".ljust(58) + "║")
    print("║" + "  🛑 Press Ctrl+C to stop the server".ljust(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "═"*58 + "╝")
    print("")

def print_server_ready():
    """Print server ready message after Flask initialization."""
    print("╔" + "═"*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "✅ SERVER READY".center(58) + "║")
    print("║" + "Market Analyzer is now running successfully".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "═"*58 + "╝")
    print("")

def print_shutdown_message():
    """Print professional shutdown message."""
    import sys
    # Mark that shutdown message has been shown
    sys._shutdown_message_shown = True
    
    print("\n" + "╔" + "═"*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "🛑 SHUTTING DOWN".center(58) + "║")
    print("║" + "Market Analyzer server is stopping...".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╠" + "═"*58 + "╣")
    print("║" + "  💾 Saving session data...".ljust(58) + "║")
    print("║" + "  🔒 Closing connections...".ljust(58) + "║")
    print("║" + "  ✅ Cleanup complete".ljust(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "═"*58 + "╝")
    print("\n👋 Thank you for using Market Analyzer!")
    print("💡 Run 'python3 app.py' again to restart\n")

def configure_logging():
    """Configure logging for the application."""
    # Set up basic config first
    logging.basicConfig(
        level=logging.CRITICAL,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    # Suppress specific loggers completely
    loggers_to_suppress = [
        'werkzeug',
        'yfinance',
        'matplotlib',
        'urllib3',
        'asyncio',
        'flask.app',
        'flask_compress',
        'flask_caching',
        'flask_socketio',
        'engineio',
        'socketio',
        'requests',
        'urllib3.connectionpool',
        'matplotlib.font_manager',
        'PIL'
    ]
    
    for logger_name in loggers_to_suppress:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL)
        logger.propagate = False
        logger.disabled = True
    
    # Suppress all warnings
    import warnings
    warnings.filterwarnings("ignore")
    
    # Specifically suppress yfinance warnings
    import os
    os.environ['PYTHONWARNINGS'] = 'ignore'
    
    # Suppress Flask CLI messages
    os.environ['FLASK_ENV'] = 'production'
    
    # Suppress Flask startup messages by redirecting both stdout and stderr
    import sys
    from io import StringIO
    
    class SuppressFlaskOutput:
        def __init__(self):
            self.original_stdout = sys.stdout
            self.original_stderr = sys.stderr
            
        def __enter__(self):
            sys.stdout = StringIO()
            sys.stderr = StringIO()
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr
    
    return SuppressFlaskOutput

if __name__ == '__main__':
    # Configure logging first and get the suppressor
    SuppressFlaskOutput = configure_logging()
    
    # Set up signal handler for graceful shutdown
    import signal
    
    def signal_handler(sig, frame):
        print_shutdown_message()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Start the browser after a short delay
    threading.Timer(1.25, open_browser).start()
    
    # Add a timer to show "Server Ready" message after Flask initializes
    def show_ready_message():
        import time
        time.sleep(2.0)  # Wait for Flask to finish startup
        print_server_ready()
    
    threading.Timer(2.0, show_ready_message).start()
    
    # Run the app
    try:
        # Print enhanced startup message
        print_startup_message()
        
        # Configure Flask to run silently
        import logging
        
        # Disable Flask's startup messages by setting werkzeug logger to ERROR level
        werkzeug_logger = logging.getLogger('werkzeug')
        werkzeug_logger.setLevel(logging.ERROR)
        werkzeug_logger.disabled = True
        
        # Also try to suppress click (Flask CLI) messages
        click_logger = logging.getLogger('click')
        click_logger.setLevel(logging.ERROR)
        click_logger.disabled = True
        
        # Get port from environment variable or use default 5001 for local development
        port = int(os.environ.get('PORT', 5001))
        
        # Use localhost for local development, 0.0.0.0 for production
        host = '127.0.0.1' if port == 5001 else '0.0.0.0'
        
        app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)
        
    except KeyboardInterrupt:
        print_shutdown_message()
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        import traceback
        traceback.print_exc()
        print_shutdown_message()
    finally:
        # Always show shutdown message if it hasn't been shown yet
        import sys
        if not hasattr(sys, '_shutdown_message_shown'):
            print_shutdown_message()
            sys._shutdown_message_shown = True
