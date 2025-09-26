from flask import Flask, render_template, request, jsonify, redirect, url_for
import sys
import os
import threading
import time
import traceback
import pandas as pd
import numpy as np
import yfinance as yf
from market_analyzer import analyze_stock, predict_stock_prices
from datetime import datetime, timedelta
import webbrowser
from typing import List, Dict, Any
from jinja2 import Environment

app = Flask(__name__)
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
    
    print(f"\n=== DEBUG ===")
    print(f"Client IP: {client_ip}")
    print(f"User Agent: {user_agent}")
    print(f"Show Shutdown: {shutdown_enabled}")
    print("============\n")
    
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
    print("Debug: /analysis route was accessed!")
    try:
        context = get_common_context()
        print(f"Debug: Context created successfully: {context}")
        result = render_template('analysis.html', **context)
        print("Debug: Template rendered successfully")
        return result
    except Exception as e:
        print(f"Debug: Error in analysis route: {str(e)}")
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
    
    # Debug info
    print(f"[DEBUG] Starting prediction analysis for {ticker} (last {days} days)", file=sys.stderr)
    
    try:
        # Get historical data
        start_date = pd.Timestamp.now() - pd.Timedelta(days=days*2)
        end_date = pd.Timestamp.now()
        
        print(f"[DEBUG] Downloading data from {start_date} to {end_date}", file=sys.stderr)
        data = yf.download(ticker, start=start_date, end=end_date)
        
        if data is None or data.empty:
            error_msg = f"No data available for {ticker} from {start_date} to {end_date}"
            print(f"[ERROR] {error_msg}", file=sys.stderr)
            return f"Error: {error_msg}"
            
        print(f"[DEBUG] Downloaded {len(data)} rows of data", file=sys.stderr)
        print(f"[DEBUG] Data columns: {data.columns.tolist()}", file=sys.stderr)
        
        # Run prediction
        print("[DEBUG] Calling predict_stock_prices...", file=sys.stderr)
        try:
            from market_analyzer import predict_stock_prices
            prediction_results = predict_stock_prices(data, ticker, days_ahead=[1, 3, 7, 14, 30])
            
            if prediction_results is None:
                error_msg = "Prediction function returned None"
                print(f"[ERROR] {error_msg}", file=sys.stderr)
                return f"Error: {error_msg}"
                
        except Exception as pred_error:
            error_msg = f"Error in predict_stock_prices: {str(pred_error)}"
            print(f"[ERROR] {error_msg}\n{traceback.format_exc()}", file=sys.stderr)
            return f"Error: {error_msg}"
            
        print(f"[DEBUG] Prediction results: {prediction_results}", file=sys.stderr)
        
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
            print(f"[ERROR] Error formatting results: {str(format_error)}\n{traceback.format_exc()}", file=sys.stderr)
            output.append("\n⚠️ Analysis completed with partial results. Some data may be missing.")
            
        return "\n".join(output)
        
    except Exception as e:
        error_msg = f"Error in prediction analysis: {str(e)}\n{traceback.format_exc()}"
        print(f"[CRITICAL] {error_msg}", file=sys.stderr)
        return f"Error: {str(e)}\n\nPlease check the server logs for more details."

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
        
        # Debug log
        print(f"Starting {analysis_type} analysis for {ticker} (last {days} days)", file=sys.stderr)
        
        # For prediction analysis, we don't need to capture stdout
        if analysis_type == 'prediction':
            try:
                print(f"[DEBUG] Running prediction analysis for {ticker}", file=sys.stderr)
                output = run_prediction_analysis(ticker, days)
                print(f"[DEBUG] Prediction output length: {len(output) if output else 0}", file=sys.stderr)
                print(f"[DEBUG] Prediction output preview: {output[:200] if output else 'None'}...", file=sys.stderr)
                if not output:
                    raise ValueError("No output from prediction analysis")
                return jsonify({
                    'ticker': ticker,
                    'analysis': output,
                    'analysis_type': analysis_type,
                    'status': 'success'
                })
            except Exception as e:
                print(f"Error in prediction analysis: {str(e)}\n{traceback.format_exc()}", file=sys.stderr)
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
                print(f"Error in basic analysis: {str(e)}", file=sys.stderr)
                output = f"<div class='p-4'><h3 class='text-lg font-semibold mb-2 text-red-600'>Error</h3><p>Failed to fetch data for {ticker}: {str(e)}</p></div>"
                
            # Return the basic analysis result
            return jsonify({
                'ticker': ticker,
                'analysis': output,
                'analysis_type': analysis_type,
                'status': 'success'
            })
        
        else:
            # For technical and full analysis, capture stdout
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            
            try:
                # For technical and full analysis, run the full analysis
                analyze_stock(
                    ticker=ticker,
                    show_history=True,
                    history_days=days,
                    show_graphs=False,
                    skip_plot=True
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
                print(f"Error in analysis: {str(e)}\n{traceback.format_exc()}", file=sys.stderr)
                return jsonify({
                    'error': str(e),
                    'status': 'error',
                    'traceback': traceback.format_exc()
                }), 500
                
            finally:
                sys.stdout = old_stdout
            
    except Exception as e:
        print(f"Unexpected error in analyze route: {str(e)}\n{traceback.format_exc()}", file=sys.stderr)
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
                print(f"Error fetching {module} for {ticker_upper}: {str(e)}")
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
        print(f"Error fetching fundamental data for {ticker}: {str(e)}\n{traceback.format_exc()}")
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
        print(f"[DEBUG] Fetching data for ticker: {ticker}")
        # Get stock data
        stock = yf.Ticker(ticker)
        
        # Get historical data first (more reliable than info)
        hist = stock.history(period='5d')
        
        if hist.empty:
            print(f"[DEBUG] No historical data found for {ticker}")
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
            print(f"[DEBUG] Only one day of data for {ticker}, using zero change")
            price_change = 0
            percent_change = 0
            prev_close = current_price
        
        print(f"[DEBUG] Successfully fetched data for {ticker}")
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
    print("=== SCREENER DEBUG ===")
    print("Screener route accessed")
    
    context = get_common_context()
    try:
        # Get popular stocks and fetch their data
        print("Getting popular stocks...")
        tickers = get_popular_stocks()
        print(f"Got {len(tickers)} tickers: {tickers[:5]}...")  # Show first 5
        
        print("Using sample data for immediate page load...")
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
        print(f"Final stocks count: {len(context['stocks'])}")
        
    except Exception as e:
        print(f"Error in screener route: {str(e)}")
        import traceback
        traceback.print_exc()
        # Fallback to empty list if there's an error
        context['stocks'] = []
    
    print("Rendering template...")
    return render_template('screener.html', **context)

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Start the browser after a short delay
    if os.environ.get('WERKZEUG_RUN_MAIN') != 'true':  # Prevent duplicate tabs in debug mode
        threading.Timer(1.25, open_browser).start()
    
    # Run the app
    try:
        app.run(debug=True, use_reloader=False, port=5001)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        print("\nServer has been shut down. You can close this window.")
