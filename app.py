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

app = Flask(__name__)
shutdown_enabled = False

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

@app.route('/')
def index():
    global shutdown_enabled
    # Debug information
    client_ip = request.remote_addr
    user_agent = request.user_agent.string
    shutdown_enabled = (client_ip == '127.0.0.1')
    
    print(f"\n=== DEBUG ===")
    print(f"Client IP: {client_ip}")
    print(f"User Agent: {user_agent}")
    print(f"Show Shutdown: {shutdown_enabled}")
    print("============\n")
    
    return render_template('index.html', 
                         show_shutdown=shutdown_enabled,
                         debug_info={
                             'client_ip': client_ip,
                             'user_agent': user_agent,
                             'show_shutdown': shutdown_enabled
                         })

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
    analysis_type = request.form.get('analysisType', 'basic')
    
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
                print(f"Error in prediction analysis: {str(e)}\n{traceback.format_exc()}", file=sys.stderr)
                return jsonify({
                    'error': f"Error in prediction analysis: {str(e)}",
                    'status': 'error',
                    'traceback': traceback.format_exc()
                }), 500
        
        # For other analysis types, capture stdout
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            if analysis_type == 'basic':
                # For basic analysis, return minimal output
                output = f"Basic analysis completed for {ticker}. Chart data available for {days} days."
            else:
                # For technical and full analysis, run the full analysis
                analyze_stock(
                    ticker=ticker,
                    show_history=True,
                    history_days=days,
                    show_graphs=False,
                    skip_plot=True
                )
                
                # Get the output
                output = sys.stdout.getvalue()
            
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

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Start the browser automatically
    threading.Thread(target=open_browser).start()
    
    # Run the app
    try:
        app.run(debug=True, use_reloader=False, port=5001)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        print("\nServer has been shut down. You can close this window.")
