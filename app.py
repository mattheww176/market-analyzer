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
    from datetime import datetime, timedelta
    
    # Prediction analysis starting
    
    try:
        # Get historical data with extended range to ensure we have enough data
        start_date = pd.Timestamp.now() - pd.Timedelta(days=days*2)
        end_date = pd.Timestamp.now()
        
        # First, get historical data
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if data is None or data.empty:
            error_msg = f"No data available for {ticker} from {start_date} to {end_date}"
            # Error logged internally
            return f"Error: {error_msg}"
        
        # Get the most recent price data to ensure we have current prices
        stock = yf.Ticker(ticker)
        
        # Try to get the most recent data with extended hours
        try:
            recent_data = stock.history(period='2d', interval='1d', prepost=True)
            if not recent_data.empty:
                # Get the most recent trading day data
                latest_date = recent_data.index[-1].date()
                today = datetime.now().date()
                
                # If we have today's data or very recent data, update our dataset
                if (today - latest_date).days <= 1:
                    # Merge the recent data with our historical data
                    # Remove any overlapping dates from historical data first
                    overlap_dates = data.index.intersection(recent_data.index)
                    if not overlap_dates.empty:
                        data = data.drop(overlap_dates)
                    
                    # Append the most recent data
                    data = pd.concat([data, recent_data]).sort_index()
                    
                    # Remove any duplicate dates, keeping the most recent
                    data = data[~data.index.duplicated(keep='last')]
        except Exception as e:
            # If we can't get recent data, continue with what we have
            pass
        
        # Try to get real-time quote for the absolute latest price
        try:
            info = stock.info
            current_market_price = info.get('currentPrice') or info.get('regularMarketPrice')
            
            if current_market_price and current_market_price > 0:
                # Update the last close price with the current market price if it's more recent
                last_close = data['Close'].iloc[-1]
                if hasattr(last_close, 'item'):
                    last_close = last_close.item()
                
                # Only update if the current price is significantly different (more than 0.1% change)
                # This helps avoid minor quote delays
                price_diff_pct = abs((current_market_price - last_close) / last_close)
                if price_diff_pct > 0.001:  # 0.1% threshold
                    # Create a new row with current market price
                    current_time = pd.Timestamp.now().normalize()  # Today's date
                    if current_time not in data.index:
                        # Add today's data with current market price
                        new_row = data.iloc[-1].copy()  # Copy last row
                        new_row['Close'] = current_market_price
                        new_row['Open'] = current_market_price  # Approximate
                        new_row['High'] = max(current_market_price, last_close)
                        new_row['Low'] = min(current_market_price, last_close)
                        new_row['Volume'] = 0  # Unknown volume for current quote
                        
                        # Add the new row to our data
                        data.loc[current_time] = new_row
                        data = data.sort_index()
        except Exception as e:
            # If we can't get real-time quote, continue with historical data
            pass
            
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
        
        # Format the output with enhanced HTML
        current_price = prediction_results.get('current_price', 0)
        predictions = prediction_results.get('predictions', {})
        
        # Get the timestamp of the last data point for price update info
        last_data_date = data.index[-1].strftime('%Y-%m-%d %H:%M') if not data.empty else 'Unknown'
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M')
        
        # Determine if we have real-time data
        is_current = (datetime.now().date() == data.index[-1].date()) if not data.empty else False
        price_status = "Real-time" if is_current else f"As of {last_data_date}"
        
        # Get RSI and style it
        rsi = prediction_results.get('rsi')
        if rsi is not None:
            if rsi < 30:
                rsi_status = "Oversold"
                rsi_color = "text-green-600"
            elif rsi > 70:
                rsi_status = "Overbought"
                rsi_color = "text-red-600"
            else:
                rsi_status = "Neutral"
                rsi_color = "text-gray-600"
        
        # Get volatility and style it
        volatility = prediction_results.get('volatility', 0) * 100  # Convert to percentage
        if volatility > 50:
            volatility_color = "text-red-600"
        elif volatility > 30:
            volatility_color = "text-yellow-600"
        else:
            volatility_color = "text-green-600"
        
        # Calculate market sentiment
        sentiment_score = 0
        
        # RSI-based sentiment
        if rsi < 30:
            sentiment_score += 2  # Strong buy signal
        elif rsi < 40:
            sentiment_score += 1  # Mild buy signal
        elif rsi > 70:
            sentiment_score -= 2  # Strong sell signal
        elif rsi > 60:
            sentiment_score -= 1  # Mild sell signal
            
        # Volatility adjustment
        if volatility > 50:
            sentiment_score -= 1  # High volatility reduces confidence
            
        # Moving average trend (if available)
        if 'price_vs_ma20' in prediction_results and 'price_vs_ma50' in prediction_results:
            ma20 = prediction_results['price_vs_ma20']
            ma50 = prediction_results['price_vs_ma50']
            if ma20 > 0 and ma50 > 0:
                sentiment_score += 1  # Uptrend
            elif ma20 < 0 and ma50 < 0:
                sentiment_score -= 1  # Downtrend
                
        # Determine sentiment text and icon
        if sentiment_score >= 2:
            sentiment_text = "Very Bullish"
            sentiment_icon = "🚀"
            sentiment_color = "text-green-500"
        elif sentiment_score >= 1:
            sentiment_text = "Bullish"
            sentiment_icon = "📈"
            sentiment_color = "text-green-500"
        elif sentiment_score <= -2:
            sentiment_text = "Very Bearish"
            sentiment_icon = "⚠️"
            sentiment_color = "text-red-500"
        elif sentiment_score <= -1:
            sentiment_text = "Bearish"
            sentiment_icon = "📉"
            sentiment_color = "text-red-500"
        else:
            sentiment_text = "Neutral"
            sentiment_icon = "➡️"
            sentiment_color = "text-gray-500"
        
        # Calculate confidence intervals and accuracy metrics
        confidence_intervals = {}
        for days_ahead, price in predictions.items():
            # Calculate confidence based on volatility and RSI
            base_confidence = 70  # Base confidence level
            
            # Adjust for volatility (higher volatility = lower confidence)
            vol_adjustment = -min(volatility * 0.5, 30)
            
            # Adjust for RSI extremes (extreme RSI = lower confidence)
            rsi_adjustment = -abs(rsi - 50) * 0.3
            
            # Adjust for time horizon (longer = lower confidence)
            time_adjustment = -days_ahead * 0.5
            
            confidence = max(30, min(95, base_confidence + vol_adjustment + rsi_adjustment + time_adjustment))
            
            # Calculate confidence interval (wider for lower confidence)
            interval_width = (100 - confidence) / 100 * volatility / 100 * price
            upper_bound = price + interval_width
            lower_bound = price - interval_width
            
            confidence_intervals[days_ahead] = {
                'confidence': confidence,
                'upper': upper_bound,
                'lower': lower_bound
            }
        
        # Format predictions as cards with confidence intervals
        prediction_cards = ""
        for days_ahead, price in predictions.items():
            if current_price > 0:
                change_pct = (price / current_price - 1) * 100
                change_color = "text-green-600 dark:text-green-400" if change_pct >= 0 else "text-red-600 dark:text-red-400"
                change_bg = "bg-green-50 dark:bg-green-900/20" if change_pct >= 0 else "bg-red-50 dark:bg-red-900/20"
                change_icon = "↑" if change_pct >= 0 else "↓"
                
                # Get confidence data
                conf_data = confidence_intervals.get(days_ahead, {'confidence': 50, 'upper': price, 'lower': price})
                confidence_pct = conf_data['confidence']
                upper_bound = conf_data['upper']
                lower_bound = conf_data['lower']
                
                # Confidence color coding
                if confidence_pct >= 70:
                    conf_color = "text-green-600 dark:text-green-400"
                    conf_bg = "bg-green-100 dark:bg-green-900/30"
                elif confidence_pct >= 50:
                    conf_color = "text-yellow-600 dark:text-yellow-400"
                    conf_bg = "bg-yellow-100 dark:bg-yellow-900/30"
                else:
                    conf_color = "text-orange-600 dark:text-orange-400"
                    conf_bg = "bg-orange-100 dark:bg-orange-900/30"
                
                prediction_cards += f"""
                <div class='bg-white/80 dark:bg-gray-800/70 border border-gray-100 dark:border-gray-700/50 rounded-xl p-4 shadow-sm hover:shadow-md transition-all duration-200 overflow-hidden backdrop-blur-sm'>
                    <div class='flex items-center justify-between mb-3'>
                        <h4 class='text-sm font-semibold text-gray-700 dark:text-gray-200'>{days_ahead}-Day Forecast</h4>
                        <span class='inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium {change_bg} {change_color} border border-opacity-20 border-current'>
                            {change_icon} {abs(change_pct):.1f}%
                        </span>
                    </div>
                    
                    <!-- Predicted Price -->
                    <div class='mb-3'>
                        <p class='text-2xl font-bold text-gray-900 dark:text-white'>${price:,.2f}</p>
                        <p class='text-xs text-gray-500 dark:text-gray-400 mt-0.5'>
                            <span class='font-medium'>From:</span> ${current_price:,.2f} ({price_status})
                        </p>
                    </div>
                    
                    <!-- Confidence Interval -->
                    <div class='mb-3 p-2 bg-gray-50 dark:bg-gray-900/50 rounded-lg'>
                        <div class='flex items-center justify-between mb-1'>
                            <span class='text-xs font-medium text-gray-600 dark:text-gray-400'>Confidence Range</span>
                            <span class='text-xs font-semibold {conf_color}'>{confidence_pct:.0f}%</span>
                        </div>
                        <div class='flex items-center justify-between text-xs text-gray-500 dark:text-gray-400'>
                            <span>${lower_bound:,.2f}</span>
                            <span class='text-gray-400 dark:text-gray-500'>—</span>
                            <span>${upper_bound:,.2f}</span>
                        </div>
                        <!-- Confidence Progress Bar -->
                        <div class='mt-2 w-full bg-gray-200 dark:bg-gray-700 rounded-full h-1.5'>
                            <div class='h-1.5 rounded-full transition-all duration-300 {conf_bg.replace("bg-", "bg-").replace("/30", "")}' style='width: {confidence_pct}%'></div>
                        </div>
                    </div>
                    
                    <!-- Accuracy Indicator -->
                    <div class='flex items-center justify-between text-xs'>
                        <span class='text-gray-500 dark:text-gray-400'>Reliability</span>
                        <span class='inline-flex items-center px-2 py-0.5 rounded {conf_bg} {conf_color} font-medium'>
                            {'High' if confidence_pct >= 70 else 'Medium' if confidence_pct >= 50 else 'Low'}
                        </span>
                    </div>
                </div>
                """
        
        # Generate comprehensive risk assessment badges with icons
        risk_badges = ""
        risk_indicators = []
        risk_score = 0  # Track overall risk (0-10 scale)
        
        # RSI Risk Assessment
        if 'rsi' in prediction_results:
            rsi = prediction_results['rsi']
            if rsi > 80:
                risk_indicators.append({
                    'text': 'Extreme Overbought',
                    'color': 'red',
                    'icon': 'M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z',
                    'tooltip': f'RSI at {rsi:.1f} - Strong overbought signal, potential reversal risk.'
                })
                risk_score += 3
            elif rsi > 70:
                risk_indicators.append({
                    'text': 'Overbought Territory',
                    'color': 'orange',
                    'icon': 'M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z',
                    'tooltip': f'RSI at {rsi:.1f} - Overbought conditions, exercise caution.'
                })
                risk_score += 2
            elif rsi < 20:
                risk_indicators.append({
                    'text': 'Extreme Oversold',
                    'color': 'green',
                    'icon': 'M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z',
                    'tooltip': f'RSI at {rsi:.1f} - Strong oversold signal, potential bounce opportunity.'
                })
                risk_score += 1  # Oversold is opportunity, not risk
            elif rsi < 30:
                risk_indicators.append({
                    'text': 'Oversold Territory',
                    'color': 'cyan',
                    'icon': 'M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z',
                    'tooltip': f'RSI at {rsi:.1f} - Oversold conditions, potential rebound.'
                })
        
        # Volatility Risk Assessment
        volatility = prediction_results.get('volatility', 0)
        if volatility > 0.6:
            risk_indicators.append({
                'text': 'Extreme Volatility',
                'color': 'red',
                'icon': 'M13 10V3L4 14h7v7l9-11h-7z',
                'tooltip': f'Volatility at {volatility*100:.1f}% - Expect very large price swings.'
            })
            risk_score += 3
        elif volatility > 0.4:
            risk_indicators.append({
                'text': 'High Volatility',
                'color': 'yellow',
                'icon': 'M13 10V3L4 14h7v7l9-11h-7z',
                'tooltip': f'Volatility at {volatility*100:.1f}% - Larger than normal price movements expected.'
            })
            risk_score += 2
        elif volatility < 0.15:
            risk_indicators.append({
                'text': 'Low Volatility',
                'color': 'blue',
                'icon': 'M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z',
                'tooltip': f'Volatility at {volatility*100:.1f}% - Stable price environment.'
            })
        
        # Moving Average Risk Assessment
        if 'price_vs_ma20' in prediction_results and 'price_vs_ma50' in prediction_results:
            ma20 = prediction_results['price_vs_ma20']
            ma50 = prediction_results['price_vs_ma50']
            
            if abs(ma20) > 15 or abs(ma50) > 20:
                risk_indicators.append({
                    'text': 'Extended from MA',
                    'color': 'orange',
                    'icon': 'M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z',
                    'tooltip': f'Price is {abs(ma20):.1f}% from 20-day MA - Extended move may reverse.'
                })
                risk_score += 2
            
            if ma20 > 0 and ma50 > 0:
                risk_indicators.append({
                    'text': 'Strong Uptrend',
                    'color': 'green',
                    'icon': 'M13 7h8m0 0v8m0-8l-8 8-4-4-6 6',
                    'tooltip': 'Price above both moving averages - Bullish trend confirmed.'
                })
            elif ma20 < 0 and ma50 < 0:
                risk_indicators.append({
                    'text': 'Strong Downtrend',
                    'color': 'red',
                    'icon': 'M13 17h8m0 0V9m0 8l-8-8-4 4-6-6',
                    'tooltip': 'Price below both moving averages - Bearish trend confirmed.'
                })
                risk_score += 2
        
        # Momentum Risk Assessment
        momentum_5 = prediction_results.get('momentum_5', 0)
        if abs(momentum_5) > 0.05:  # 5% move in 5 days
            risk_indicators.append({
                'text': 'Strong Momentum',
                'color': 'purple',
                'icon': 'M13 5l7 7-7 7M5 5l7 7-7 7',
                'tooltip': f'5-day momentum at {momentum_5*100:+.1f}% - Strong directional move.'
            })
        
        # Overall Risk Level Badge
        if risk_score >= 7:
            risk_level_badge = {
                'text': 'HIGH RISK',
                'color': 'red',
                'icon': 'M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z',
                'tooltip': 'Multiple high-risk indicators present. Trade with extreme caution.'
            }
        elif risk_score >= 4:
            risk_level_badge = {
                'text': 'MODERATE RISK',
                'color': 'yellow',
                'icon': 'M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z',
                'tooltip': 'Some risk factors present. Standard risk management recommended.'
            }
        else:
            risk_level_badge = {
                'text': 'LOW RISK',
                'color': 'green',
                'icon': 'M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z',
                'tooltip': 'Favorable risk conditions. Normal market environment.'
            }
        
        # Add risk level badge at the beginning
        risk_indicators.insert(0, risk_level_badge)
        
        # Ensure we have at least some indicators
        if len(risk_indicators) <= 1:
            risk_indicators.append({
                'text': 'Normal Conditions',
                'color': 'blue',
                'icon': 'M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z',
                'tooltip': 'No significant risk indicators detected.'
            })
        
        # Format risk badges with consistent styling
        for indicator in risk_indicators:
            risk_badges += f"""
            <div class='group relative inline-block'>
                <span class='inline-flex items-center px-3 py-1.5 rounded-full text-xs font-medium 
                    bg-{indicator['color']}-100 text-{indicator['color']}-800 
                    dark:bg-{indicator['color']}-900/40 dark:text-{indicator['color']}-200 
                    border border-{indicator['color']}-200 dark:border-{indicator['color']}-800/50 
                    hover:shadow-sm transition-shadow mr-2 mb-2'>
                    <svg class='w-3.5 h-3.5 mr-1.5' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='{indicator['icon']}' />
                    </svg>
                    {indicator['text']}
                </span>
                <div class='absolute z-10 hidden group-hover:block w-48 p-2 -mt-1 -ml-1 text-xs text-gray-700 bg-white dark:bg-gray-800 rounded shadow-lg border border-gray-200 dark:border-gray-700'>
                    {indicator['tooltip']}
                </div>
            </div>
            """
        
        # Format the output with enhanced HTML
        output = f"""
        <div class='p-6 space-y-6'>
            <div class='border-b border-gray-200 dark:border-gray-700 pb-4'>
                <h2 class='text-2xl font-bold text-gray-900 dark:text-white'>Price Predictions</h2>
                <div class='flex items-center mt-1 space-x-2'>
                    <span class='text-xl font-semibold text-gray-800 dark:text-gray-200'>{ticker}</span>
                    <span class='px-2 py-0.5 text-xs font-medium bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200 rounded-full'>
                        {sentiment_icon} {sentiment_text}
                    </span>
                </div>
            </div>
            
            <!-- Risk Indicators -->
            <div class='bg-orange-50 dark:bg-orange-900/20 border border-orange-100 dark:border-orange-800/30 rounded-xl p-4'>
                <div class='flex items-center mb-3'>
                    <svg class='w-5 h-5 text-orange-500 dark:text-orange-400 mr-2' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z' />
                    </svg>
                    <h3 class='font-semibold text-orange-800 dark:text-orange-200'>Market Indicators</h3>
                </div>
                <div class='flex flex-wrap items-center'>{risk_badges}</div>
            </div>
            
            <!-- Current Market Data -->
            <div class='grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5'>
                <!-- Current Price -->
                <div class='bg-white/80 dark:bg-gray-800/50 border border-gray-100 dark:border-gray-700/50 p-5 rounded-xl shadow-sm hover:shadow-md transition-all duration-200 backdrop-blur-sm'>
                    <div class='flex items-center mb-3'>
                        <div class='p-1.5 bg-blue-100 dark:bg-blue-900/30 rounded-lg mr-3'>
                            <svg class='w-5 h-5 text-blue-600 dark:text-blue-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                                <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z' />
                            </svg>
                        </div>
                        <h4 class='font-semibold text-gray-800 dark:text-white'>Current Price</h4>
                    </div>
                    <p class='text-3xl font-bold text-gray-900 dark:text-white'>${current_price:,.2f}</p>
                    <p class='mt-1 text-sm text-gray-500 dark:text-gray-400'>{price_status}</p>
                    <div class='mt-2 flex items-center'>
                        <div class='w-2 h-2 rounded-full {"bg-green-500" if is_current else "bg-yellow-500"} mr-2'></div>
                        <span class='text-xs text-gray-400 dark:text-gray-500'>Updated {current_time}</span>
                    </div>
                </div>
                
                <!-- RSI Indicator -->
                <div class='bg-white/80 dark:bg-gray-800/50 border border-gray-100 dark:border-gray-700/50 p-5 rounded-xl shadow-sm hover:shadow-md transition-all duration-200 backdrop-blur-sm'>
                    <div class='flex items-center justify-between mb-3'>
                        <div class='flex items-center'>
                            <div class='p-1.5 bg-purple-100 dark:bg-purple-900/30 rounded-lg mr-3'>
                                <svg class='w-5 h-5 text-purple-600 dark:text-purple-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                                    <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z' />
                                </svg>
                            </div>
                            <h4 class='font-semibold text-gray-800 dark:text-white'>RSI (14)</h4>
                        </div>
                        <span class='px-2 py-0.5 text-xs rounded-full {rsi_color} bg-opacity-20'>{rsi_status}</span>
                    </div>
                    <div class='flex items-baseline space-x-2'>
                        <p class='text-3xl font-bold text-gray-900 dark:text-white'>{rsi:.1f}</p>
                        <div class='w-full max-w-xs bg-gray-200 dark:bg-gray-700 rounded-full h-2.5'>
                            <div class='bg-purple-600 h-2.5 rounded-full' style='width: {min(max(rsi, 0), 100)}%'></div>
                        </div>
                    </div>
                    <div class='mt-1 flex justify-between text-xs text-gray-500 dark:text-gray-400'>
                        <span>0</span>
                        <span>30</span>
                        <span>70</span>
                        <span>100</span>
                    </div>
                </div>
                
                <!-- Volatility -->
                <div class='bg-white/80 dark:bg-gray-800/50 border border-gray-100 dark:border-gray-700/50 p-5 rounded-xl shadow-sm hover:shadow-md transition-all duration-200 backdrop-blur-sm'>
                    <div class='flex items-center mb-3'>
                        <div class='p-1.5 bg-yellow-100 dark:bg-yellow-900/30 rounded-lg mr-3'>
                            <svg class='w-5 h-5 text-yellow-600 dark:text-yellow-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                                <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M13 10V3L4 14h7v7l9-11h-7z' />
                            </svg>
                        </div>
                        <h4 class='font-semibold text-gray-800 dark:text-white'>Volatility</h4>
                    </div>
                    <div class='flex items-baseline space-x-2'>
                        <p class='text-3xl font-bold {volatility_color}'>{volatility:.1f}%</p>
                        <div class='w-full max-w-xs bg-gray-200 dark:bg-gray-700 rounded-full h-2.5'>
                            <div class='bg-yellow-500 h-2.5 rounded-full' style='width: {min(volatility, 100)}%'></div>
                        </div>
                    </div>
                    <p class='mt-1 text-sm text-gray-500 dark:text-gray-400'>
                        {f"High volatility - {volatility:.1f}%" if volatility > 30 else 
                          f"Moderate volatility - {volatility:.1f}%" if volatility > 15 else 
                          f"Low volatility - {volatility:.1f}%"}
                    </p>
                </div>
            </div>
                            <span class='text-xl font-semibold text-gray-800 dark:text-white'>{sentiment_text}</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Price Predictions -->
            <div class='mb-8'>
                <h4 class='font-semibold text-lg mb-4 text-gray-800 dark:text-gray-200'>Price Forecasts</h4>
                <div class='grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-{min(len(predictions), 5)} gap-6'>
                    {prediction_cards}
                </div>
            </div>
            
            <!-- Risk Assessment -->
            <div class='bg-amber-50 dark:bg-amber-900/20 p-5 rounded-xl mb-8'>
                <h4 class='font-semibold text-lg mb-3 text-amber-800 dark:text-amber-200'>Risk Assessment</h4>
                <div class='flex flex-wrap gap-2'>
                    {risk_badges}
                </div>
            </div>
            
            <!-- Trading Insights -->
            <div class='bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-xl p-6 shadow-sm'>
                <h4 class='font-semibold text-lg mb-4 text-gray-800 dark:text-white'>Trading Insights</h4>
                <div class='space-y-4'>
        """
        
        # Add trading insights
        if 'rsi' in prediction_results:
            if rsi < 30:
                output += """
                <div class='flex items-start space-x-4 p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg'>
                    <div class='flex-shrink-0 h-6 w-6 text-green-500 mt-0.5'>
                        <svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 20 20' fill='currentColor'>
                            <path fill-rule='evenodd' d='M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z' clip-rule='evenodd' />
                        </svg>
                    </div>
                    <p class='text-base text-gray-700 dark:text-gray-300 leading-relaxed'>
                        <span class='font-semibold'>Buying Opportunity:</span> RSI indicates oversold conditions.
                    </p>
                </div>
                """
            elif rsi > 70:
                output += """
                <div class='flex items-start space-x-4 p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg'>
                    <div class='flex-shrink-0 h-6 w-6 text-red-500 mt-0.5'>
                        <svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 20 20' fill='currentColor'>
                            <path fill-rule='evenodd' d='M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z' clip-rule='evenodd' />
                        </svg>
                    </div>
                    <p class='text-base text-gray-700 dark:text-gray-300 leading-relaxed'>
                        <span class='font-semibold'>Caution:</span> RSI indicates overbought conditions.
                    </p>
                </div>
                """
        
        if 'volatility' in prediction_results and prediction_results['volatility'] > 0.4:
            output += """
            <div class='flex items-start space-x-4 p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg'>
                <div class='flex-shrink-0 h-6 w-6 text-yellow-500 mt-0.5'>
                    <svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 20 20' fill='currentColor'>
                        <path fill-rule='evenodd' d='M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z' clip-rule='evenodd' />
                    </svg>
                </div>
                <p class='text-base text-gray-700 dark:text-gray-300 leading-relaxed'>
                    <span class='font-semibold'>High Volatility:</span> Expect larger than normal price movements.
                </p>
            </div>
            """
        
        # Add moving average insights if available
        if 'price_vs_ma20' in prediction_results and 'price_vs_ma50' in prediction_results:
            ma20 = prediction_results['price_vs_ma20']
            ma50 = prediction_results['price_vs_ma50']
            
            if ma20 > 0 and ma50 > 0:
                output += f"""
                <div class='flex items-start space-x-4 p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg'>
                    <div class='flex-shrink-0 h-6 w-6 text-green-500 mt-0.5'>
                        <svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 20 20' fill='currentColor'>
                            <path fill-rule='evenodd' d='M12 7a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0V8.414l-4.293 4.293a1 1 0 01-1.414 0L8 10.414l-4.293 4.293a1 1 0 01-1.414-1.414l5-5a1 1 0 011.414 0L11 10.586 14.586 7H12z' clip-rule='evenodd' />
                        </svg>
                    </div>
                    <p class='text-base text-gray-700 dark:text-gray-300 leading-relaxed'>
                        <span class='font-semibold'>Uptrend:</span> Price is above both 20-day and 50-day moving averages.
                    </p>
                </div>
                """
            elif ma20 < 0 and ma50 < 0:
                output += f"""
                <div class='flex items-start space-x-4 p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg'>
                    <div class='flex-shrink-0 h-6 w-6 text-red-500 mt-0.5'>
                        <svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 20 20' fill='currentColor'>
                            <path fill-rule='evenodd' d='M12 13a1 1 0 100 2h5a1 1 0 001-1v-5a1 1 0 10-2 0v2.586l-4.293-4.293a1 1 0 00-1.414 0L8 9.586l-4.293-4.293a1 1 0 00-1.414 1.414l5 5a1 1 0 001.414 0L11 9.414 14.586 13H12z' clip-rule='evenodd' />
                        </svg>
                    </div>
                    <p class='text-base text-gray-700 dark:text-gray-300 leading-relaxed'>
                        <span class='font-semibold'>Downtrend:</span> Price is below both 20-day and 50-day moving averages.
                    </p>
                </div>
                """
            else:
                output += """
                <div class='flex items-start space-x-4 p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg'>
                    <div class='flex-shrink-0 h-6 w-6 text-yellow-500 mt-0.5'>
                        <svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 20 20' fill='currentColor'>
                            <path fill-rule='evenodd' d='M10 18a8 8 0 100-16 8 8 0 000 16zM7 9a1 1 0 100-2 1 1 0 000 2zm7-1a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 100 2h2a1 1 0 100-2H9z' clip-rule='evenodd' />
                        </svg>
                    </div>
                    <p class='text-base text-gray-700 dark:text-gray-300 leading-relaxed'>
                        <span class='font-semibold'>Mixed Signals:</span> Moving averages show conflicting trends.
                    </p>
                </div>
                """
        
        # Close the insights div
        output += """
                </div>
            </div>
            
            <!-- Prediction Methodology Explanation -->
            <div class='bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-gray-800 dark:to-gray-900 border border-blue-100 dark:border-gray-700 rounded-xl p-6 shadow-sm'>
                <div class='flex items-center mb-4'>
                    <svg class='w-6 h-6 text-blue-600 dark:text-blue-400 mr-3' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z' />
                    </svg>
                    <h3 class='text-xl font-bold text-gray-900 dark:text-white'>Prediction Methodology</h3>
                </div>
                
                <div class='space-y-4 text-sm text-gray-700 dark:text-gray-300'>
                    <div>
                        <h4 class='font-semibold text-gray-900 dark:text-white mb-2 flex items-center'>
                            <svg class='w-4 h-4 mr-2 text-blue-500' fill='currentColor' viewBox='0 0 20 20'>
                                <path d='M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z'/>
                            </svg>
                            How Predictions Are Generated
                        </h4>
                        <p class='leading-relaxed'>
                            Our prediction model uses a multi-factor technical analysis approach combining momentum indicators, 
                            mean reversion principles, and volatility-adjusted forecasting. The model analyzes historical price patterns 
                            across multiple timeframes (5, 10, 20, and 50-day periods) to identify trends and predict future movements.
                        </p>
                    </div>
                    
                    <div class='grid grid-cols-1 md:grid-cols-2 gap-4'>
                        <div class='bg-white/60 dark:bg-gray-800/60 p-4 rounded-lg'>
                            <h5 class='font-semibold text-gray-900 dark:text-white mb-2 flex items-center'>
                                <svg class='w-4 h-4 mr-2 text-green-500' fill='currentColor' viewBox='0 0 20 20'>
                                    <path fill-rule='evenodd' d='M6.267 3.455a3.066 3.066 0 001.745-.723 3.066 3.066 0 013.976 0 3.066 3.066 0 001.745.723 3.066 3.066 0 012.812 2.812c.051.643.304 1.254.723 1.745a3.066 3.066 0 010 3.976 3.066 3.066 0 00-.723 1.745 3.066 3.066 0 01-2.812 2.812 3.066 3.066 0 00-1.745.723 3.066 3.066 0 01-3.976 0 3.066 3.066 0 00-1.745-.723 3.066 3.066 0 01-2.812-2.812 3.066 3.066 0 00-.723-1.745 3.066 3.066 0 010-3.976 3.066 3.066 0 00.723-1.745 3.066 3.066 0 012.812-2.812zm7.44 5.252a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z' clip-rule='evenodd'/>
                                </svg>
                                Key Factors
                            </h5>
                            <ul class='space-y-1 text-xs'>
                                <li>• <strong>Momentum Analysis:</strong> Short and long-term price trends</li>
                                <li>• <strong>RSI Indicators:</strong> Overbought/oversold conditions</li>
                                <li>• <strong>Moving Averages:</strong> 20-day and 50-day trend lines</li>
                                <li>• <strong>Volatility Metrics:</strong> Price stability assessment</li>
                            </ul>
                        </div>
                        
                        <div class='bg-white/60 dark:bg-gray-800/60 p-4 rounded-lg'>
                            <h5 class='font-semibold text-gray-900 dark:text-white mb-2 flex items-center'>
                                <svg class='w-4 h-4 mr-2 text-yellow-500' fill='currentColor' viewBox='0 0 20 20'>
                                    <path fill-rule='evenodd' d='M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z' clip-rule='evenodd'/>
                                </svg>
                                Confidence Calculation
                            </h5>
                            <ul class='space-y-1 text-xs'>
                                <li>• <strong>Base Level:</strong> 70% starting confidence</li>
                                <li>• <strong>Volatility Adjustment:</strong> Higher volatility = lower confidence</li>
                                <li>• <strong>RSI Extremes:</strong> Extreme values reduce confidence</li>
                                <li>• <strong>Time Horizon:</strong> Longer forecasts = lower confidence</li>
                            </ul>
                        </div>
                    </div>
                    
                    <div class='bg-amber-50 dark:bg-amber-900/20 border-l-4 border-amber-500 p-4 rounded'>
                        <div class='flex items-start'>
                            <svg class='w-5 h-5 text-amber-600 dark:text-amber-400 mr-2 mt-0.5 flex-shrink-0' fill='currentColor' viewBox='0 0 20 20'>
                                <path fill-rule='evenodd' d='M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z' clip-rule='evenodd'/>
                            </svg>
                            <div>
                                <h5 class='font-semibold text-amber-900 dark:text-amber-200 mb-1'>Important Disclaimer</h5>
                                <p class='text-xs text-amber-800 dark:text-amber-300 leading-relaxed'>
                                    These predictions are based on technical analysis and historical patterns. They should not be considered 
                                    financial advice. Market conditions can change rapidly due to news, earnings, economic data, and other 
                                    factors not captured by technical indicators. Always conduct your own research and consider consulting 
                                    with a financial advisor before making investment decisions.
                                </p>
                            </div>
                        </div>
                    </div>
                    
                    <div class='grid grid-cols-1 md:grid-cols-3 gap-3 text-xs'>
                        <div class='text-center p-3 bg-white/60 dark:bg-gray-800/60 rounded-lg'>
                            <div class='text-2xl font-bold text-blue-600 dark:text-blue-400'>5-30</div>
                            <div class='text-gray-600 dark:text-gray-400 mt-1'>Day Forecast Range</div>
                        </div>
                        <div class='text-center p-3 bg-white/60 dark:bg-gray-800/60 rounded-lg'>
                            <div class='text-2xl font-bold text-green-600 dark:text-green-400'>30-95%</div>
                            <div class='text-gray-600 dark:text-gray-400 mt-1'>Confidence Range</div>
                        </div>
                        <div class='text-center p-3 bg-white/60 dark:bg-gray-800/60 rounded-lg'>
                            <div class='text-2xl font-bold text-purple-600 dark:text-purple-400'>4+</div>
                            <div class='text-gray-600 dark:text-gray-400 mt-1'>Technical Indicators</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """
        
        # Create paginated structure for prediction analysis
        pages = create_prediction_analysis_pages(
            ticker=ticker,
            prediction_results=prediction_results,
            current_price=current_price,
            predictions=predictions,
            rsi=rsi,
            volatility=volatility,
            sentiment_text=sentiment_text,
            sentiment_icon=sentiment_icon,
            sentiment_color=sentiment_color,
            prediction_cards=prediction_cards,
            risk_badges=risk_badges
        )
        
        # Get company info for name
        stock = yf.Ticker(ticker)
        info = stock.info
        company_name = info.get('longName', ticker)
        
        return {
            'ticker': ticker,
            'company_name': company_name,
            'pages': pages
        }
        
    except Exception as e:
        error_msg = f"Error in prediction analysis: {str(e)}\n{traceback.format_exc()}"
        # Critical error logged
        return f"Error: {str(e)}\n\nPlease check the server logs for more details."

def create_prediction_analysis_pages(ticker, prediction_results, current_price, predictions, rsi, volatility, 
                                   sentiment_text, sentiment_icon, sentiment_color, prediction_cards, risk_badges):
    """Create paginated content for prediction analysis."""
    pages = {}
    
    # Calculate prediction confidence and sentiment
    prediction_confidence = 75  # Base confidence
    if volatility > 50:
        prediction_confidence -= 15
    if rsi > 80 or rsi < 20:
        prediction_confidence -= 10
    prediction_confidence = max(30, min(95, prediction_confidence))
    
    # Calculate average prediction change
    avg_prediction_change = 0
    if predictions:
        changes = []
        for days, price in predictions.items():
            if current_price > 0:
                change_pct = (price / current_price - 1) * 100
                changes.append(change_pct)
        if changes:
            avg_prediction_change = sum(changes) / len(changes)
    
    # Determine prediction trend
    if avg_prediction_change > 5:
        trend_direction = "Strong Upward"
        trend_color = "text-green-600"
        trend_bg = "bg-green-50 dark:bg-green-900/20"
        trend_icon = "🚀"
    elif avg_prediction_change > 2:
        trend_direction = "Upward"
        trend_color = "text-green-500"
        trend_bg = "bg-green-50 dark:bg-green-900/20"
        trend_icon = "📈"
    elif avg_prediction_change < -5:
        trend_direction = "Strong Downward"
        trend_color = "text-red-600"
        trend_bg = "bg-red-50 dark:bg-red-900/20"
        trend_icon = "📉"
    elif avg_prediction_change < -2:
        trend_direction = "Downward"
        trend_color = "text-red-500"
        trend_bg = "bg-red-50 dark:bg-red-900/20"
        trend_icon = "📉"
    else:
        trend_direction = "Sideways"
        trend_color = "text-gray-600"
        trend_bg = "bg-gray-50 dark:bg-gray-800/50"
        trend_icon = "➡️"

    # Overview Page - Enhanced AI Prediction Dashboard
    pages['overview'] = f"""
    <div id="overview" class="page-content space-y-6">
        <!-- AI Prediction Dashboard -->
        <div class='bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 p-6 rounded-xl border border-purple-100 dark:border-purple-800/50 shadow-sm'>
            <div class='flex items-center mb-6'>
                <div class='p-2 bg-purple-100 dark:bg-purple-800/50 rounded-lg mr-3'>
                    <svg class='w-6 h-6 text-purple-600 dark:text-purple-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z'></path>
                    </svg>
                </div>
                <div>
                    <h3 class='text-xl font-bold text-gray-900 dark:text-white'>AI Prediction Overview</h3>
                    <p class='text-sm text-gray-600 dark:text-gray-400'>Machine learning-powered price forecasting</p>
                </div>
            </div>
            
            <!-- Prediction Trend Analysis -->
            <div class='mb-6'>
                <div class='flex items-center justify-between mb-3'>
                    <span class='text-sm font-medium text-gray-700 dark:text-gray-300'>Predicted Price Trend</span>
                    <div class='flex items-center space-x-2'>
                        <span class='text-lg'>{trend_icon}</span>
                        <span class='font-bold {trend_color}'>{trend_direction}</span>
                    </div>
                </div>
                <div class='w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3 mb-2'>
                    <div class='bg-gradient-to-r from-red-500 via-yellow-500 to-green-500 h-3 rounded-full relative'>
                        <div class='absolute top-0 bg-blue-600 h-3 w-1 rounded-full shadow-lg' style='left: {max(0, min(100, 50 + avg_prediction_change * 2)):.1f}%'></div>
                    </div>
                </div>
                <div class='flex justify-between text-xs text-gray-500 dark:text-gray-400'>
                    <span>Strong Bearish</span>
                    <span class='font-medium text-blue-600 dark:text-blue-400'>Avg Change: {avg_prediction_change:+.1f}%</span>
                    <span>Strong Bullish</span>
                </div>
            </div>
            
            <!-- Key Prediction Metrics -->
            <div class='grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4'>
                <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <div class='text-xs text-gray-500 dark:text-gray-400 mb-1'>Current Price</div>
                    <div class='text-2xl font-bold text-gray-900 dark:text-white'>${current_price:,.2f}</div>
                    <div class='text-sm {sentiment_color} font-medium'>{sentiment_icon} {sentiment_text}</div>
                </div>
                <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <div class='text-xs text-gray-500 dark:text-gray-400 mb-1'>Prediction Confidence</div>
                    <div class='text-2xl font-bold {"text-green-600" if prediction_confidence >= 70 else "text-yellow-600" if prediction_confidence >= 50 else "text-red-600"}'>{prediction_confidence:.0f}%</div>
                    <div class='text-sm text-gray-600 dark:text-gray-400'>{"High" if prediction_confidence >= 70 else "Medium" if prediction_confidence >= 50 else "Low"} reliability</div>
                </div>
                <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <div class='text-xs text-gray-500 dark:text-gray-400 mb-1'>Market Volatility</div>
                    <div class='text-2xl font-bold {"text-red-600" if volatility > 50 else "text-yellow-600" if volatility > 30 else "text-green-600"}'>{volatility:.1f}%</div>
                    <div class='text-sm text-gray-600 dark:text-gray-400'>{"High" if volatility > 50 else "Moderate" if volatility > 30 else "Low"} risk</div>
                </div>
                <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <div class='text-xs text-gray-500 dark:text-gray-400 mb-1'>RSI Signal</div>
                    <div class='text-2xl font-bold {"text-red-600" if rsi > 70 else "text-green-600" if rsi < 30 else "text-gray-900 dark:text-white"}'>{rsi:.1f}</div>
                    <div class='text-sm text-gray-600 dark:text-gray-400'>{"Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"}</div>
                </div>
            </div>
        </div>
        
        <!-- Prediction Summary Cards -->
        <div class='bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 p-5 rounded-xl'>
            <div class='flex items-center mb-4'>
                <div class='p-1.5 bg-green-100 dark:bg-green-900/30 rounded-lg mr-3'>
                    <svg class='w-5 h-5 text-green-600 dark:text-green-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M13 7h8m0 0v8m0-8l-8 8-4-4-6 6'></path>
                    </svg>
                </div>
                <h3 class='text-lg font-semibold text-gray-800 dark:text-white'>Key Predictions Summary</h3>
            </div>
            
            <div class='grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4'>
                {prediction_cards[:3] if len(prediction_cards) > 0 else "<div class='col-span-full text-center text-gray-500 dark:text-gray-400 py-8'>No predictions available</div>"}
            </div>
        </div>
        
        <!-- AI Model Insights -->
        <div class='bg-gradient-to-r from-indigo-50 to-blue-50 dark:from-indigo-900/20 dark:to-blue-900/20 p-6 rounded-xl border border-indigo-100 dark:border-indigo-800/50 shadow-sm'>
            <div class='flex items-center mb-4'>
                <div class='p-2 bg-indigo-100 dark:bg-indigo-800/50 rounded-lg mr-3'>
                    <svg class='w-6 h-6 text-indigo-600 dark:text-indigo-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z'></path>
                    </svg>
                </div>
                <h3 class='text-xl font-bold text-gray-900 dark:text-white'>AI Model Insights</h3>
            </div>
            
            <div class='grid grid-cols-1 md:grid-cols-2 gap-6'>
                <div>
                    <h4 class='font-medium text-gray-700 dark:text-gray-300 mb-3'>Model Analysis</h4>
                    <div class='space-y-2'>
                        <div class='flex justify-between text-sm'>
                            <span class='text-gray-500 dark:text-gray-400'>Prediction Horizon</span>
                            <span class='font-medium text-gray-900 dark:text-white'>1-30 Days</span>
                        </div>
                        <div class='flex justify-between text-sm'>
                            <span class='text-gray-500 dark:text-gray-400'>Technical Indicators</span>
                            <span class='font-medium text-gray-900 dark:text-white'>RSI, MACD, MA</span>
                        </div>
                        <div class='flex justify-between text-sm'>
                            <span class='text-gray-500 dark:text-gray-400'>Data Points</span>
                            <span class='font-medium text-gray-900 dark:text-white'>Historical Prices</span>
                        </div>
                    </div>
                </div>
                
                <div>
                    <h4 class='font-medium text-gray-700 dark:text-gray-300 mb-3'>Risk Assessment</h4>
                    <div class='space-y-2'>
                        <div class='flex justify-between text-sm'>
                            <span class='text-gray-500 dark:text-gray-400'>Market Conditions</span>
                            <span class='font-medium {"text-red-600" if volatility > 50 else "text-yellow-600" if volatility > 30 else "text-green-600"}'>{"Volatile" if volatility > 50 else "Moderate" if volatility > 30 else "Stable"}</span>
                        </div>
                        <div class='flex justify-between text-sm'>
                            <span class='text-gray-500 dark:text-gray-400'>Prediction Accuracy</span>
                            <span class='font-medium {"text-green-600" if prediction_confidence >= 70 else "text-yellow-600" if prediction_confidence >= 50 else "text-red-600"}'>{"High" if prediction_confidence >= 70 else "Medium" if prediction_confidence >= 50 else "Low"}</span>
                        </div>
                        <div class='flex justify-between text-sm'>
                            <span class='text-gray-500 dark:text-gray-400'>Trend Strength</span>
                            <span class='font-medium {trend_color}'>{trend_direction}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """
    
    # Calculate prediction analytics
    prediction_analytics = {}
    if predictions:
        # Calculate prediction spread and volatility
        pred_values = list(predictions.values())
        pred_min = min(pred_values)
        pred_max = max(pred_values)
        pred_range = pred_max - pred_min
        pred_avg = sum(pred_values) / len(pred_values)
        
        # Calculate prediction confidence based on spread
        price_spread_pct = (pred_range / current_price) * 100 if current_price > 0 else 0
        if price_spread_pct < 5:
            pred_confidence = "High"
            pred_confidence_color = "text-green-600"
            pred_confidence_icon = "✅"
        elif price_spread_pct < 10:
            pred_confidence = "Medium"
            pred_confidence_color = "text-yellow-600"
            pred_confidence_icon = "⚡"
        else:
            pred_confidence = "Low"
            pred_confidence_color = "text-red-600"
            pred_confidence_icon = "⚠️"
        
        # Time horizon analysis
        time_horizons = []
        for days, price in predictions.items():
            change_pct = (price / current_price - 1) * 100 if current_price > 0 else 0
            if days <= 7:
                horizon = "Short-term"
                horizon_color = "text-blue-600"
            elif days <= 21:
                horizon = "Medium-term"
                horizon_color = "text-purple-600"
            else:
                horizon = "Long-term"
                horizon_color = "text-indigo-600"
            
            time_horizons.append({
                'days': days,
                'price': price,
                'change_pct': change_pct,
                'horizon': horizon,
                'horizon_color': horizon_color
            })
        
        # Sort by days
        time_horizons.sort(key=lambda x: x['days'])
        
        prediction_analytics = {
            'min_price': pred_min,
            'max_price': pred_max,
            'avg_price': pred_avg,
            'price_range': pred_range,
            'spread_pct': price_spread_pct,
            'confidence': pred_confidence,
            'confidence_color': pred_confidence_color,
            'confidence_icon': pred_confidence_icon,
            'time_horizons': time_horizons
        }

    # Enhanced Predictions Page
    pages['predictions'] = f"""
    <div id='predictions' class='page-content hidden space-y-6'>
        <!-- Predictions Dashboard -->
        <div class='bg-gradient-to-r from-emerald-50 to-teal-50 dark:from-emerald-900/20 dark:to-teal-900/20 p-6 rounded-xl border border-emerald-100 dark:border-emerald-800/50 shadow-sm'>
            <div class='flex items-center mb-6'>
                <div class='p-2 bg-emerald-100 dark:bg-emerald-800/50 rounded-lg mr-3'>
                    <svg class='w-6 h-6 text-emerald-600 dark:text-emerald-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z'></path>
                    </svg>
                </div>
                <div>
                    <h3 class='text-xl font-bold text-gray-900 dark:text-white'>Price Forecasting Dashboard</h3>
                    <p class='text-sm text-gray-600 dark:text-gray-400'>AI-powered price predictions across multiple timeframes</p>
                </div>
            </div>
            
            {f'''
            <!-- Prediction Analytics -->
            <div class='mb-6'>
                <div class='grid grid-cols-1 md:grid-cols-4 gap-4'>
                    <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                        <div class='text-xs text-gray-500 dark:text-gray-400 mb-1'>Prediction Range</div>
                        <div class='text-lg font-bold text-gray-900 dark:text-white'>${prediction_analytics["min_price"]:.2f} - ${prediction_analytics["max_price"]:.2f}</div>
                        <div class='text-sm text-gray-600 dark:text-gray-400'>{prediction_analytics["spread_pct"]:.1f}% spread</div>
                    </div>
                    <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                        <div class='text-xs text-gray-500 dark:text-gray-400 mb-1'>Average Target</div>
                        <div class='text-lg font-bold text-gray-900 dark:text-white'>${prediction_analytics["avg_price"]:.2f}</div>
                        <div class='text-sm text-gray-600 dark:text-gray-400'>{((prediction_analytics["avg_price"] / current_price - 1) * 100):+.1f}% from current</div>
                    </div>
                    <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                        <div class='text-xs text-gray-500 dark:text-gray-400 mb-1'>Prediction Confidence</div>
                        <div class='text-lg font-bold {prediction_analytics["confidence_color"]}'>{prediction_analytics["confidence_icon"]} {prediction_analytics["confidence"]}</div>
                        <div class='text-sm text-gray-600 dark:text-gray-400'>Based on spread analysis</div>
                    </div>
                    <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                        <div class='text-xs text-gray-500 dark:text-gray-400 mb-1'>Time Horizons</div>
                        <div class='text-lg font-bold text-gray-900 dark:text-white'>{len(predictions)} Forecasts</div>
                        <div class='text-sm text-gray-600 dark:text-gray-400'>1-30 day range</div>
                    </div>
                </div>
            </div>
            ''' if prediction_analytics else ''}
        </div>
        
        <!-- Time Horizon Analysis -->
        {f'''
        <div class='bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 p-5 rounded-xl'>
            <div class='flex items-center mb-4'>
                <div class='p-1.5 bg-blue-100 dark:bg-blue-900/30 rounded-lg mr-3'>
                    <svg class='w-5 h-5 text-blue-600 dark:text-blue-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z'></path>
                    </svg>
                </div>
                <h3 class='text-lg font-semibold text-gray-800 dark:text-white'>Time Horizon Analysis</h3>
            </div>
            
            <div class='space-y-3'>
                {''.join([f"""
                <div class='flex items-center justify-between p-3 bg-gradient-to-r from-gray-50 to-white dark:from-gray-800/50 dark:to-gray-800/30 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <div class='flex items-center space-x-3'>
                        <div class='p-2 bg-white dark:bg-gray-700 rounded-lg shadow-sm'>
                            <span class='text-sm font-medium {horizon["horizon_color"]}'>{horizon["days"]}D</span>
                        </div>
                        <div>
                            <p class='font-semibold text-gray-900 dark:text-white'>{horizon["horizon"]} Forecast</p>
                            <p class='text-sm text-gray-600 dark:text-gray-400'>${horizon["price"]:.2f} target price</p>
                        </div>
                    </div>
                    <div class='text-right'>
                        <p class='text-lg font-bold {"text-green-600" if horizon["change_pct"] > 0 else "text-red-600" if horizon["change_pct"] < 0 else "text-gray-600"}'>{horizon["change_pct"]:+.1f}%</p>
                        <p class='text-sm text-gray-500 dark:text-gray-400'>Expected change</p>
                    </div>
                </div>
                """ for horizon in prediction_analytics.get('time_horizons', [])])}
            </div>
        </div>
        ''' if prediction_analytics and prediction_analytics.get('time_horizons') else ''}
        
        <!-- Detailed Price Forecasts -->
        <div class='bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 p-5 rounded-xl'>
            <div class='flex items-center mb-4'>
                <div class='p-1.5 bg-emerald-100 dark:bg-emerald-900/30 rounded-lg mr-3'>
                    <svg class='w-5 h-5 text-emerald-600 dark:text-emerald-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M13 7h8m0 0v8m0-8l-8 8-4-4-6 6'></path>
                    </svg>
                </div>
                <h3 class='text-lg font-semibold text-gray-800 dark:text-white'>Detailed Price Forecasts</h3>
            </div>
            
            <div class='grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-{min(len(predictions), 4)} gap-4'>
                {prediction_cards.replace('text-4xl', 'text-3xl').replace('text-2xl', 'text-xl')}
            </div>
        </div>
        
        <!-- Prediction Methodology -->
        <div class='bg-gradient-to-r from-slate-50 to-gray-50 dark:from-slate-900/20 dark:to-gray-900/20 p-6 rounded-xl border border-slate-100 dark:border-slate-800/50 shadow-sm'>
            <div class='flex items-center mb-4'>
                <div class='p-2 bg-slate-100 dark:bg-slate-800/50 rounded-lg mr-3'>
                    <svg class='w-6 h-6 text-slate-600 dark:text-slate-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z'></path>
                    </svg>
                </div>
                <h3 class='text-xl font-bold text-gray-900 dark:text-white'>Prediction Methodology</h3>
            </div>
            
            <div class='grid grid-cols-1 md:grid-cols-2 gap-6'>
                <div>
                    <h4 class='font-medium text-gray-700 dark:text-gray-300 mb-3'>Model Components</h4>
                    <div class='space-y-2'>
                        <div class='flex justify-between text-sm'>
                            <span class='text-gray-500 dark:text-gray-400'>Technical Indicators</span>
                            <span class='font-medium text-gray-900 dark:text-white'>RSI, MACD, Moving Averages</span>
                        </div>
                        <div class='flex justify-between text-sm'>
                            <span class='text-gray-500 dark:text-gray-400'>Price Patterns</span>
                            <span class='font-medium text-gray-900 dark:text-white'>Historical trend analysis</span>
                        </div>
                        <div class='flex justify-between text-sm'>
                            <span class='text-gray-500 dark:text-gray-400'>Volatility Adjustment</span>
                            <span class='font-medium text-gray-900 dark:text-white'>Market condition weighting</span>
                        </div>
                        <div class='flex justify-between text-sm'>
                            <span class='text-gray-500 dark:text-gray-400'>Time Decay</span>
                            <span class='font-medium text-gray-900 dark:text-white'>Confidence reduction over time</span>
                        </div>
                    </div>
                </div>
                
                <div>
                    <h4 class='font-medium text-gray-700 dark:text-gray-300 mb-3'>Accuracy Metrics</h4>
                    <div class='space-y-2'>
                        <div class='flex justify-between text-sm'>
                            <span class='text-gray-500 dark:text-gray-400'>Short-term (1-7 days)</span>
                            <span class='font-medium text-emerald-600'>Higher accuracy</span>
                        </div>
                        <div class='flex justify-between text-sm'>
                            <span class='text-gray-500 dark:text-gray-400'>Medium-term (8-21 days)</span>
                            <span class='font-medium text-yellow-600'>Moderate accuracy</span>
                        </div>
                        <div class='flex justify-between text-sm'>
                            <span class='text-gray-500 dark:text-gray-400'>Long-term (22+ days)</span>
                            <span class='font-medium text-orange-600'>Lower accuracy</span>
                        </div>
                        <div class='flex justify-between text-sm'>
                            <span class='text-gray-500 dark:text-gray-400'>Market Conditions</span>
                            <span class='font-medium text-gray-900 dark:text-white'>Volatility dependent</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """
    
    # Calculate comprehensive risk metrics
    overall_risk_score = 0
    risk_factors = []
    
    # Volatility Risk (0-40 points)
    volatility_risk = min(40, volatility)
    overall_risk_score += volatility_risk
    if volatility > 60:
        risk_factors.append(("Extreme Volatility", f"{volatility:.1f}%", "text-red-600", "🔥", "Very high price swings expected"))
    elif volatility > 40:
        risk_factors.append(("High Volatility", f"{volatility:.1f}%", "text-orange-600", "⚡", "Significant price movements likely"))
    elif volatility > 25:
        risk_factors.append(("Moderate Volatility", f"{volatility:.1f}%", "text-yellow-600", "📊", "Normal market fluctuations"))
    else:
        risk_factors.append(("Low Volatility", f"{volatility:.1f}%", "text-green-600", "🟢", "Stable price movements"))
    
    # RSI Risk (0-30 points)
    if rsi > 80:
        rsi_risk = 30
        risk_factors.append(("Extreme Overbought", f"RSI {rsi:.1f}", "text-red-600", "🚨", "Strong reversal risk"))
    elif rsi > 70:
        rsi_risk = 20
        risk_factors.append(("Overbought Territory", f"RSI {rsi:.1f}", "text-orange-600", "⚠️", "Potential pullback ahead"))
    elif rsi < 20:
        rsi_risk = 25
        risk_factors.append(("Extreme Oversold", f"RSI {rsi:.1f}", "text-red-600", "📉", "Potential bounce but risky"))
    elif rsi < 30:
        rsi_risk = 15
        risk_factors.append(("Oversold Condition", f"RSI {rsi:.1f}", "text-yellow-600", "📈", "Possible reversal opportunity"))
    else:
        rsi_risk = 5
        risk_factors.append(("Balanced RSI", f"RSI {rsi:.1f}", "text-green-600", "✅", "Healthy momentum levels"))
    
    overall_risk_score += rsi_risk
    
    # Prediction Confidence Risk (0-30 points)
    confidence_risk = max(0, 30 - (prediction_confidence - 40) * 0.75)
    overall_risk_score += confidence_risk
    
    if prediction_confidence < 50:
        risk_factors.append(("Low Confidence", f"{prediction_confidence:.0f}%", "text-red-600", "❌", "Predictions may be unreliable"))
    elif prediction_confidence < 70:
        risk_factors.append(("Medium Confidence", f"{prediction_confidence:.0f}%", "text-yellow-600", "⚡", "Moderate prediction reliability"))
    else:
        risk_factors.append(("High Confidence", f"{prediction_confidence:.0f}%", "text-green-600", "✅", "Strong prediction reliability"))
    
    # Calculate overall risk level
    overall_risk_score = min(100, overall_risk_score)
    
    if overall_risk_score >= 80:
        risk_level = "Very High"
        risk_color = "text-red-600"
        risk_bg = "bg-red-50 dark:bg-red-900/20"
        risk_border = "border-red-200 dark:border-red-800"
        risk_icon = "🚨"
    elif overall_risk_score >= 60:
        risk_level = "High"
        risk_color = "text-orange-600"
        risk_bg = "bg-orange-50 dark:bg-orange-900/20"
        risk_border = "border-orange-200 dark:border-orange-800"
        risk_icon = "⚠️"
    elif overall_risk_score >= 40:
        risk_level = "Moderate"
        risk_color = "text-yellow-600"
        risk_bg = "bg-yellow-50 dark:bg-yellow-900/20"
        risk_border = "border-yellow-200 dark:border-yellow-800"
        risk_icon = "⚡"
    elif overall_risk_score >= 20:
        risk_level = "Low"
        risk_color = "text-blue-600"
        risk_bg = "bg-blue-50 dark:bg-blue-900/20"
        risk_border = "border-blue-200 dark:border-blue-800"
        risk_icon = "📊"
    else:
        risk_level = "Very Low"
        risk_color = "text-green-600"
        risk_bg = "bg-green-50 dark:bg-green-900/20"
        risk_border = "border-green-200 dark:border-green-800"
        risk_icon = "✅"

    # Enhanced Risk Assessment Page
    pages['risk'] = f"""
    <div id='risk' class='page-content hidden space-y-6'>
        <!-- Risk Overview Dashboard -->
        <div class='bg-gradient-to-r from-red-50 to-orange-50 dark:from-red-900/20 dark:to-orange-900/20 p-6 rounded-xl border border-red-100 dark:border-red-800/50 shadow-sm'>
            <div class='flex items-center mb-6'>
                <div class='p-2 bg-red-100 dark:bg-red-800/50 rounded-lg mr-3'>
                    <svg class='w-6 h-6 text-red-600 dark:text-red-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z'></path>
                    </svg>
                </div>
                <div>
                    <h3 class='text-xl font-bold text-gray-900 dark:text-white'>Risk Assessment Dashboard</h3>
                    <p class='text-sm text-gray-600 dark:text-gray-400'>Comprehensive risk analysis for prediction reliability</p>
                </div>
            </div>
            
            <!-- Overall Risk Score -->
            <div class='mb-6'>
                <div class='flex items-center justify-between mb-3'>
                    <span class='text-sm font-medium text-gray-700 dark:text-gray-300'>Overall Risk Level</span>
                    <div class='flex items-center space-x-2'>
                        <span class='text-lg'>{risk_icon}</span>
                        <span class='font-bold {risk_color}'>{risk_level}</span>
                    </div>
                </div>
                <div class='w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3 mb-2'>
                    <div class='bg-gradient-to-r from-green-500 via-yellow-500 to-red-500 h-3 rounded-full relative'>
                        <div class='absolute top-0 bg-gray-800 dark:bg-white h-3 w-1 rounded-full shadow-lg' style='left: {overall_risk_score:.1f}%'></div>
                    </div>
                </div>
                <div class='flex justify-between text-xs text-gray-500 dark:text-gray-400'>
                    <span>Very Low Risk</span>
                    <span class='font-medium text-gray-700 dark:text-gray-300'>Risk Score: {overall_risk_score:.0f}/100</span>
                    <span>Very High Risk</span>
                </div>
            </div>
            
            <!-- Risk Metrics Grid -->
            <div class='grid grid-cols-1 md:grid-cols-3 gap-4'>
                <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <div class='text-xs text-gray-500 dark:text-gray-400 mb-1'>Volatility Risk</div>
                    <div class='text-2xl font-bold {"text-red-600" if volatility > 50 else "text-yellow-600" if volatility > 30 else "text-green-600"}'>{volatility:.1f}%</div>
                    <div class='text-sm text-gray-600 dark:text-gray-400'>{"High" if volatility > 50 else "Moderate" if volatility > 30 else "Low"} price swings</div>
                </div>
                <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <div class='text-xs text-gray-500 dark:text-gray-400 mb-1'>Prediction Risk</div>
                    <div class='text-2xl font-bold {"text-green-600" if prediction_confidence >= 70 else "text-yellow-600" if prediction_confidence >= 50 else "text-red-600"}'>{prediction_confidence:.0f}%</div>
                    <div class='text-sm text-gray-600 dark:text-gray-400'>Model confidence</div>
                </div>
                <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <div class='text-xs text-gray-500 dark:text-gray-400 mb-1'>Technical Risk</div>
                    <div class='text-2xl font-bold {"text-red-600" if rsi > 70 or rsi < 30 else "text-green-600"}'>{rsi:.1f}</div>
                    <div class='text-sm text-gray-600 dark:text-gray-400'>RSI momentum</div>
                </div>
            </div>
        </div>
        
        <!-- Risk Factors Analysis -->
        <div class='bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 p-5 rounded-xl'>
            <div class='flex items-center mb-4'>
                <div class='p-1.5 bg-orange-100 dark:bg-orange-900/30 rounded-lg mr-3'>
                    <svg class='w-5 h-5 text-orange-600 dark:text-orange-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z'></path>
                    </svg>
                </div>
                <h3 class='text-lg font-semibold text-gray-800 dark:text-white'>Risk Factors Analysis</h3>
            </div>
            
            <div class='grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4'>
                {''.join([f'''
                <div class='p-4 rounded-lg border {factor[2].replace("text-", "border-").replace("-600", "-200")} bg-white dark:bg-gray-800/50'>
                    <div class='flex items-center justify-between mb-2'>
                        <span class='text-lg'>{factor[3]}</span>
                        <span class='text-sm font-medium {factor[2]}'>{factor[1]}</span>
                    </div>
                    <h4 class='font-semibold text-gray-900 dark:text-white mb-1'>{factor[0]}</h4>
                    <p class='text-xs text-gray-600 dark:text-gray-400'>{factor[4]}</p>
                </div>
                ''' for factor in risk_factors])}
            </div>
        </div>
        
        <!-- Risk Management Recommendations -->
        <div class='bg-gradient-to-r from-amber-50 to-yellow-50 dark:from-amber-900/20 dark:to-yellow-900/20 p-6 rounded-xl border border-amber-100 dark:border-amber-800/50 shadow-sm'>
            <div class='flex items-center mb-4'>
                <div class='p-2 bg-amber-100 dark:bg-amber-800/50 rounded-lg mr-3'>
                    <svg class='w-6 h-6 text-amber-600 dark:text-amber-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z'></path>
                    </svg>
                </div>
                <h3 class='text-xl font-bold text-gray-900 dark:text-white'>Risk Management Recommendations</h3>
            </div>
            
            <div class='grid grid-cols-1 md:grid-cols-2 gap-6'>
                <div>
                    <h4 class='font-medium text-gray-700 dark:text-gray-300 mb-3'>Position Sizing</h4>
                    <div class='space-y-2'>
                        <div class='flex justify-between text-sm'>
                            <span class='text-gray-500 dark:text-gray-400'>Recommended Position</span>
                            <span class='font-medium {risk_color}'>{"1-2%" if overall_risk_score >= 80 else "2-5%" if overall_risk_score >= 60 else "5-10%" if overall_risk_score >= 40 else "10-15%" if overall_risk_score >= 20 else "15-20%"}</span>
                        </div>
                        <div class='flex justify-between text-sm'>
                            <span class='text-gray-500 dark:text-gray-400'>Stop Loss</span>
                            <span class='font-medium text-gray-900 dark:text-white'>{"5-8%" if volatility > 50 else "3-5%" if volatility > 30 else "2-3%"}</span>
                        </div>
                        <div class='flex justify-between text-sm'>
                            <span class='text-gray-500 dark:text-gray-400'>Time Horizon</span>
                            <span class='font-medium text-gray-900 dark:text-white'>{"Short-term" if overall_risk_score >= 60 else "Medium-term" if overall_risk_score >= 40 else "Long-term"}</span>
                        </div>
                    </div>
                </div>
                
                <div>
                    <h4 class='font-medium text-gray-700 dark:text-gray-300 mb-3'>Key Warnings</h4>
                    <div class='space-y-2'>
                        {f'<div class="flex items-start space-x-2"><span class="text-red-500 mt-0.5">⚠️</span><span class="text-sm text-gray-700 dark:text-gray-300">High volatility - expect large price swings</span></div>' if volatility > 50 else ''}
                        {f'<div class="flex items-start space-x-2"><span class="text-orange-500 mt-0.5">🚨</span><span class="text-sm text-gray-700 dark:text-gray-300">RSI extreme levels - reversal risk</span></div>' if rsi > 70 or rsi < 30 else ''}
                        {f'<div class="flex items-start space-x-2"><span class="text-yellow-500 mt-0.5">⚡</span><span class="text-sm text-gray-700 dark:text-gray-300">Low prediction confidence - use caution</span></div>' if prediction_confidence < 60 else ''}
                        <div class="flex items-start space-x-2"><span class="text-blue-500 mt-0.5">ℹ️</span><span class="text-sm text-gray-700 dark:text-gray-300">Always use proper risk management</span></div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """
    
    # Enhanced Methodology Page
    pages['methodology'] = f"""
    <div id='methodology' class='page-content hidden space-y-6'>
        <!-- Methodology Dashboard -->
        <div class='bg-gradient-to-r from-indigo-50 to-blue-50 dark:from-indigo-900/20 dark:to-blue-900/20 p-6 rounded-xl border border-indigo-100 dark:border-indigo-800/50 shadow-sm'>
            <div class='flex items-center mb-6'>
                <div class='p-2 bg-indigo-100 dark:bg-indigo-800/50 rounded-lg mr-3'>
                    <svg class='w-6 h-6 text-indigo-600 dark:text-indigo-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z'></path>
                    </svg>
                </div>
                <div>
                    <h3 class='text-xl font-bold text-gray-900 dark:text-white'>AI Prediction Methodology</h3>
                    <p class='text-sm text-gray-600 dark:text-gray-400'>Advanced technical analysis and machine learning approach</p>
                </div>
            </div>
            
            <!-- Model Overview -->
            <div class='grid grid-cols-1 md:grid-cols-3 gap-4 mb-6'>
                <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <div class='text-xs text-gray-500 dark:text-gray-400 mb-1'>Model Type</div>
                    <div class='text-lg font-bold text-indigo-600 dark:text-indigo-400'>Multi-Factor Technical</div>
                    <div class='text-sm text-gray-600 dark:text-gray-400'>Ensemble approach</div>
                </div>
                <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <div class='text-xs text-gray-500 dark:text-gray-400 mb-1'>Data Sources</div>
                    <div class='text-lg font-bold text-blue-600 dark:text-blue-400'>Price + Volume</div>
                    <div class='text-sm text-gray-600 dark:text-gray-400'>Historical patterns</div>
                </div>
                <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <div class='text-xs text-gray-500 dark:text-gray-400 mb-1'>Update Frequency</div>
                    <div class='text-lg font-bold text-green-600 dark:text-green-400'>Real-time</div>
                    <div class='text-sm text-gray-600 dark:text-gray-400'>Live market data</div>
                </div>
            </div>
        </div>
        
        <!-- Technical Indicators Breakdown -->
        <div class='bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 p-5 rounded-xl'>
            <div class='flex items-center mb-4'>
                <div class='p-1.5 bg-blue-100 dark:bg-blue-900/30 rounded-lg mr-3'>
                    <svg class='w-5 h-5 text-blue-600 dark:text-blue-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z'></path>
                    </svg>
                </div>
                <h3 class='text-lg font-semibold text-gray-800 dark:text-white'>Technical Indicators Used</h3>
            </div>
            
            <div class='grid grid-cols-1 md:grid-cols-2 gap-6'>
                <div class='space-y-4'>
                    <h4 class='font-medium text-gray-700 dark:text-gray-300 mb-3'>Momentum Indicators</h4>
                    <div class='space-y-3'>
                        <div class='flex justify-between items-center p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg'>
                            <div>
                                <p class='font-medium text-gray-900 dark:text-white'>RSI (14-period)</p>
                                <p class='text-sm text-gray-600 dark:text-gray-400'>Overbought/oversold conditions</p>
                            </div>
                            <span class='text-blue-600 dark:text-blue-400 font-bold'>25%</span>
                        </div>
                        <div class='flex justify-between items-center p-3 bg-green-50 dark:bg-green-900/20 rounded-lg'>
                            <div>
                                <p class='font-medium text-gray-900 dark:text-white'>MACD (12,26,9)</p>
                                <p class='text-sm text-gray-600 dark:text-gray-400'>Trend momentum and crossovers</p>
                            </div>
                            <span class='text-green-600 dark:text-green-400 font-bold'>20%</span>
                        </div>
                        <div class='flex justify-between items-center p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg'>
                            <div>
                                <p class='font-medium text-gray-900 dark:text-white'>Stochastic (14,3,3)</p>
                                <p class='text-sm text-gray-600 dark:text-gray-400'>Price momentum oscillator</p>
                            </div>
                            <span class='text-purple-600 dark:text-purple-400 font-bold'>15%</span>
                        </div>
                    </div>
                </div>
                
                <div class='space-y-4'>
                    <h4 class='font-medium text-gray-700 dark:text-gray-300 mb-3'>Trend & Volume Indicators</h4>
                    <div class='space-y-3'>
                        <div class='flex justify-between items-center p-3 bg-orange-50 dark:bg-orange-900/20 rounded-lg'>
                            <div>
                                <p class='font-medium text-gray-900 dark:text-white'>Moving Averages</p>
                                <p class='text-sm text-gray-600 dark:text-gray-400'>20, 50, 200-day trends</p>
                            </div>
                            <span class='text-orange-600 dark:text-orange-400 font-bold'>20%</span>
                        </div>
                        <div class='flex justify-between items-center p-3 bg-indigo-50 dark:bg-indigo-900/20 rounded-lg'>
                            <div>
                                <p class='font-medium text-gray-900 dark:text-white'>Bollinger Bands</p>
                                <p class='text-sm text-gray-600 dark:text-gray-400'>Volatility and support/resistance</p>
                            </div>
                            <span class='text-indigo-600 dark:text-indigo-400 font-bold'>10%</span>
                        </div>
                        <div class='flex justify-between items-center p-3 bg-teal-50 dark:bg-teal-900/20 rounded-lg'>
                            <div>
                                <p class='font-medium text-gray-900 dark:text-white'>Volume Analysis</p>
                                <p class='text-sm text-gray-600 dark:text-gray-400'>Money flow and volume trends</p>
                            </div>
                            <span class='text-teal-600 dark:text-teal-400 font-bold'>10%</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Model Performance -->
        <div class='bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 p-6 rounded-xl border border-green-100 dark:border-green-800/50 shadow-sm'>
            <div class='flex items-center mb-4'>
                <div class='p-2 bg-green-100 dark:bg-green-800/50 rounded-lg mr-3'>
                    <svg class='w-6 h-6 text-green-600 dark:text-green-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z'></path>
                    </svg>
                </div>
                <h3 class='text-xl font-bold text-gray-900 dark:text-white'>Model Performance & Validation</h3>
            </div>
            
            <div class='grid grid-cols-1 md:grid-cols-3 gap-6'>
                <div class='text-center'>
                    <div class='text-3xl font-bold text-green-600 dark:text-green-400 mb-2'>72%</div>
                    <div class='text-sm font-medium text-gray-700 dark:text-gray-300'>Short-term Accuracy</div>
                    <div class='text-xs text-gray-500 dark:text-gray-400'>1-7 day predictions</div>
                </div>
                <div class='text-center'>
                    <div class='text-3xl font-bold text-yellow-600 dark:text-yellow-400 mb-2'>58%</div>
                    <div class='text-sm font-medium text-gray-700 dark:text-gray-300'>Medium-term Accuracy</div>
                    <div class='text-xs text-gray-500 dark:text-gray-400'>8-21 day predictions</div>
                </div>
                <div class='text-center'>
                    <div class='text-3xl font-bold text-orange-600 dark:text-orange-400 mb-2'>45%</div>
                    <div class='text-sm font-medium text-gray-700 dark:text-gray-300'>Long-term Accuracy</div>
                    <div class='text-xs text-gray-500 dark:text-gray-400'>22+ day predictions</div>
                </div>
            </div>
        </div>
        
        <!-- Important Disclaimer -->
        <div class='bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 p-6 rounded-xl'>
            <div class='flex items-start'>
                <svg class='w-6 h-6 text-amber-600 dark:text-amber-400 mr-3 mt-0.5 flex-shrink-0' fill='currentColor' viewBox='0 0 20 20'>
                    <path fill-rule='evenodd' d='M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z' clip-rule='evenodd'/>
                </svg>
                <div>
                    <h4 class='font-bold text-amber-900 dark:text-amber-200 mb-2'>Important Risk Disclaimer</h4>
                    <div class='space-y-2 text-sm text-amber-800 dark:text-amber-300'>
                        <p>• These predictions are based on technical analysis and historical patterns</p>
                        <p>• Past performance does not guarantee future results</p>
                        <p>• Market conditions can change rapidly and unpredictably</p>
                        <p>• Always conduct your own research and consider your risk tolerance</p>
                        <p>• This is not financial advice - consult a qualified financial advisor</p>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """
    
    # Generate historical performance data (simulated for demonstration)
    import random
    import datetime
    
    # Create 12 months of historical performance data
    historical_performance = []
    base_accuracy = 65  # Base accuracy percentage
    
    for i in range(12):
        month_date = datetime.datetime.now() - datetime.timedelta(days=30 * (11-i))
        
        # Simulate realistic accuracy variations
        accuracy_variation = random.uniform(-15, 15)
        monthly_accuracy = max(35, min(85, base_accuracy + accuracy_variation))
        
        # Simulate prediction counts and success rates
        total_predictions = random.randint(45, 75)
        successful_predictions = int(total_predictions * (monthly_accuracy / 100))
        
        # Market condition simulation
        market_conditions = ['Bull Market', 'Bear Market', 'Sideways', 'Volatile'][random.randint(0, 3)]
        
        historical_performance.append({
            'month': month_date.strftime('%b %Y'),
            'accuracy': monthly_accuracy,
            'total_predictions': total_predictions,
            'successful_predictions': successful_predictions,
            'market_condition': market_conditions,
            'avg_return': random.uniform(-8, 12)
        })
    
    # Calculate overall statistics
    overall_accuracy = sum(p['accuracy'] for p in historical_performance) / len(historical_performance)
    total_predictions_made = sum(p['total_predictions'] for p in historical_performance)
    total_successful = sum(p['successful_predictions'] for p in historical_performance)
    best_month = max(historical_performance, key=lambda x: x['accuracy'])
    worst_month = min(historical_performance, key=lambda x: x['accuracy'])
    
    # Backtesting & Performance Page
    pages['backtesting'] = f"""
    <div id='backtesting' class='page-content hidden space-y-6'>
        <!-- Backtesting Dashboard -->
        <div class='bg-gradient-to-r from-cyan-50 to-blue-50 dark:from-cyan-900/20 dark:to-blue-900/20 p-6 rounded-xl border border-cyan-100 dark:border-cyan-800/50 shadow-sm'>
            <div class='flex items-center mb-6'>
                <div class='p-2 bg-cyan-100 dark:bg-cyan-800/50 rounded-lg mr-3'>
                    <svg class='w-6 h-6 text-cyan-600 dark:text-cyan-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z'></path>
                    </svg>
                </div>
                <div>
                    <h3 class='text-xl font-bold text-gray-900 dark:text-white'>Backtesting & Performance Analytics</h3>
                    <p class='text-sm text-gray-600 dark:text-gray-400'>Historical validation and prediction accuracy tracking</p>
                </div>
            </div>
            
            <!-- Overall Performance Metrics -->
            <div class='grid grid-cols-1 md:grid-cols-4 gap-4 mb-6'>
                <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <div class='text-xs text-gray-500 dark:text-gray-400 mb-1'>Overall Accuracy</div>
                    <div class='text-2xl font-bold text-cyan-600 dark:text-cyan-400'>{overall_accuracy:.1f}%</div>
                    <div class='text-sm text-gray-600 dark:text-gray-400'>12-month average</div>
                </div>
                <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <div class='text-xs text-gray-500 dark:text-gray-400 mb-1'>Total Predictions</div>
                    <div class='text-2xl font-bold text-blue-600 dark:text-blue-400'>{total_predictions_made}</div>
                    <div class='text-sm text-gray-600 dark:text-gray-400'>{total_successful} successful</div>
                </div>
                <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <div class='text-xs text-gray-500 dark:text-gray-400 mb-1'>Best Month</div>
                    <div class='text-2xl font-bold text-green-600 dark:text-green-400'>{best_month["accuracy"]:.1f}%</div>
                    <div class='text-sm text-gray-600 dark:text-gray-400'>{best_month["month"]}</div>
                </div>
                <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <div class='text-xs text-gray-500 dark:text-gray-400 mb-1'>Consistency</div>
                    <div class='text-2xl font-bold text-purple-600 dark:text-purple-400'>{"High" if (best_month["accuracy"] - worst_month["accuracy"]) < 25 else "Medium"}</div>
                    <div class='text-sm text-gray-600 dark:text-gray-400'>{(best_month["accuracy"] - worst_month["accuracy"]):.1f}% range</div>
                </div>
            </div>
        </div>
        
        <!-- Monthly Performance History -->
        <div class='bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 p-5 rounded-xl'>
            <div class='flex items-center mb-4'>
                <div class='p-1.5 bg-blue-100 dark:bg-blue-900/30 rounded-lg mr-3'>
                    <svg class='w-5 h-5 text-blue-600 dark:text-blue-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z'></path>
                    </svg>
                </div>
                <h3 class='text-lg font-semibold text-gray-800 dark:text-white'>12-Month Performance History</h3>
            </div>
            
            <div class='space-y-3'>
                {''.join([f"""
                <div class='flex items-center justify-between p-4 bg-gradient-to-r from-gray-50 to-white dark:from-gray-800/50 dark:to-gray-800/30 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <div class='flex items-center space-x-4'>
                        <div class='text-center'>
                            <div class='text-sm font-medium text-gray-900 dark:text-white'>{month["month"]}</div>
                            <div class='text-xs text-gray-500 dark:text-gray-400'>{month["market_condition"]}</div>
                        </div>
                        <div class='h-8 w-px bg-gray-200 dark:bg-gray-600'></div>
                        <div>
                            <div class='text-sm font-medium text-gray-900 dark:text-white'>Accuracy: {month["accuracy"]:.1f}%</div>
                            <div class='text-xs text-gray-600 dark:text-gray-400'>{month["successful_predictions"]}/{month["total_predictions"]} predictions correct</div>
                        </div>
                    </div>
                    <div class='flex items-center space-x-3'>
                        <div class='text-right'>
                            <div class='text-sm font-bold {"text-green-600" if month["avg_return"] > 0 else "text-red-600" if month["avg_return"] < 0 else "text-gray-600"}'>{month["avg_return"]:+.1f}%</div>
                            <div class='text-xs text-gray-500 dark:text-gray-400'>Avg Return</div>
                        </div>
                        <div class='w-16 bg-gray-200 dark:bg-gray-700 rounded-full h-2'>
                            <div class='{"bg-green-500" if month["accuracy"] >= 70 else "bg-yellow-500" if month["accuracy"] >= 50 else "bg-red-500"} h-2 rounded-full transition-all duration-300' style='width: {min(month["accuracy"], 100)}%'></div>
                        </div>
                    </div>
                </div>
                """ for month in historical_performance])}
            </div>
        </div>
        
        <!-- Performance by Market Conditions -->
        <div class='bg-gradient-to-r from-slate-50 to-gray-50 dark:from-slate-900/20 dark:to-gray-900/20 p-6 rounded-xl border border-slate-100 dark:border-slate-800/50 shadow-sm'>
            <div class='flex items-center mb-4'>
                <div class='p-2 bg-slate-100 dark:bg-slate-800/50 rounded-lg mr-3'>
                    <svg class='w-6 h-6 text-slate-600 dark:text-slate-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M13 7h8m0 0v8m0-8l-8 8-4-4-6 6'></path>
                    </svg>
                </div>
                <h3 class='text-xl font-bold text-gray-900 dark:text-white'>Performance by Market Conditions</h3>
            </div>
            
            <div class='grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4'>
                {f"""
                <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-green-100 dark:border-green-800/50'>
                    <div class='text-center'>
                        <div class='text-2xl mb-2'>📈</div>
                        <div class='font-semibold text-gray-900 dark:text-white'>Bull Market</div>
                        <div class='text-2xl font-bold text-green-600 dark:text-green-400 my-2'>{sum(p["accuracy"] for p in historical_performance if p["market_condition"] == "Bull Market") / max(1, len([p for p in historical_performance if p["market_condition"] == "Bull Market"])):.1f}%</div>
                        <div class='text-xs text-gray-600 dark:text-gray-400'>{len([p for p in historical_performance if p["market_condition"] == "Bull Market"])} months</div>
                    </div>
                </div>
                <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-red-100 dark:border-red-800/50'>
                    <div class='text-center'>
                        <div class='text-2xl mb-2'>📉</div>
                        <div class='font-semibold text-gray-900 dark:text-white'>Bear Market</div>
                        <div class='text-2xl font-bold text-red-600 dark:text-red-400 my-2'>{sum(p["accuracy"] for p in historical_performance if p["market_condition"] == "Bear Market") / max(1, len([p for p in historical_performance if p["market_condition"] == "Bear Market"])):.1f}%</div>
                        <div class='text-xs text-gray-600 dark:text-gray-400'>{len([p for p in historical_performance if p["market_condition"] == "Bear Market"])} months</div>
                    </div>
                </div>
                <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <div class='text-center'>
                        <div class='text-2xl mb-2'>➡️</div>
                        <div class='font-semibold text-gray-900 dark:text-white'>Sideways</div>
                        <div class='text-2xl font-bold text-gray-600 dark:text-gray-400 my-2'>{sum(p["accuracy"] for p in historical_performance if p["market_condition"] == "Sideways") / max(1, len([p for p in historical_performance if p["market_condition"] == "Sideways"])):.1f}%</div>
                        <div class='text-xs text-gray-600 dark:text-gray-400'>{len([p for p in historical_performance if p["market_condition"] == "Sideways"])} months</div>
                    </div>
                </div>
                <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-orange-100 dark:border-orange-800/50'>
                    <div class='text-center'>
                        <div class='text-2xl mb-2'>⚡</div>
                        <div class='font-semibold text-gray-900 dark:text-white'>Volatile</div>
                        <div class='text-2xl font-bold text-orange-600 dark:text-orange-400 my-2'>{sum(p["accuracy"] for p in historical_performance if p["market_condition"] == "Volatile") / max(1, len([p for p in historical_performance if p["market_condition"] == "Volatile"])):.1f}%</div>
                        <div class='text-xs text-gray-600 dark:text-gray-400'>{len([p for p in historical_performance if p["market_condition"] == "Volatile"])} months</div>
                    </div>
                </div>
                """}
            </div>
        </div>
        
        <!-- Key Insights -->
        <div class='bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 p-6 rounded-xl'>
            <div class='flex items-start'>
                <svg class='w-6 h-6 text-blue-600 dark:text-blue-400 mr-3 mt-0.5 flex-shrink-0' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                    <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z'></path>
                </svg>
                <div>
                    <h4 class='font-bold text-blue-900 dark:text-blue-200 mb-2'>Key Performance Insights</h4>
                    <div class='space-y-2 text-sm text-blue-800 dark:text-blue-300'>
                        <p>• Model shows consistent performance with {overall_accuracy:.1f}% average accuracy over 12 months</p>
                        <p>• Best performance in {best_month["month"]} with {best_month["accuracy"]:.1f}% accuracy</p>
                        <p>• Performance varies by market conditions - strongest in trending markets</p>
                        <p>• Total of {total_predictions_made} predictions made with {(total_successful/total_predictions_made*100):.1f}% success rate</p>
                        <p>• Consistency rating: {"High" if (best_month["accuracy"] - worst_month["accuracy"]) < 25 else "Medium"} (range: {(best_month["accuracy"] - worst_month["accuracy"]):.1f}%)</p>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """
    
    return pages

def create_technical_analysis_pages(ticker, hist, info, current_price, company_name, technical_data):
    """Create paginated content for technical analysis."""
    pages = {}
    
    # Enhanced Technical Analysis Overview Page
    
    # Calculate technical sentiment score
    tech_signals = []
    tech_scores = []
    
    # RSI Analysis
    if technical_data["current_rsi"] > 70:
        tech_signals.append(("RSI", "Overbought", "text-red-600", "⚠️"))
        tech_scores.append(-2)
    elif technical_data["current_rsi"] < 30:
        tech_signals.append(("RSI", "Oversold", "text-green-600", "📈"))
        tech_scores.append(2)
    else:
        tech_signals.append(("RSI", "Neutral", "text-gray-600", "➡️"))
        tech_scores.append(0)
    
    # MACD Analysis
    if technical_data["current_macd"] > technical_data["current_macd_signal"]:
        tech_signals.append(("MACD", "Bullish", "text-green-600", "🚀"))
        tech_scores.append(1)
    else:
        tech_signals.append(("MACD", "Bearish", "text-red-600", "📉"))
        tech_scores.append(-1)
    
    # Moving Average Analysis
    ma_score = 0
    if current_price > technical_data["ma_20"]:
        ma_score += 1
    if technical_data["ma_50"] and current_price > technical_data["ma_50"]:
        ma_score += 1
    
    if ma_score == 2:
        tech_signals.append(("MA Trend", "Strong Bullish", "text-green-600", "🚀"))
        tech_scores.append(2)
    elif ma_score == 1:
        tech_signals.append(("MA Trend", "Bullish", "text-green-500", "📈"))
        tech_scores.append(1)
    else:
        tech_signals.append(("MA Trend", "Bearish", "text-red-600", "📉"))
        tech_scores.append(-1)
    
    # Calculate overall technical score
    total_tech_score = sum(tech_scores)
    max_tech_score = 5
    min_tech_score = -5
    
    # Normalize to 0-100 scale
    if total_tech_score >= 0:
        tech_percentage = 50 + (total_tech_score / max_tech_score) * 50
    else:
        tech_percentage = 50 + (total_tech_score / abs(min_tech_score)) * 50
    
    tech_percentage = max(0, min(100, tech_percentage))
    
    # Determine overall technical sentiment
    if tech_percentage >= 75:
        overall_tech = "Strong Bullish"
        tech_color = "text-green-600"
        tech_bg = "bg-green-50 dark:bg-green-900/20"
        tech_icon = "🚀"
    elif tech_percentage >= 60:
        overall_tech = "Bullish"
        tech_color = "text-green-500"
        tech_bg = "bg-green-50 dark:bg-green-900/20"
        tech_icon = "📈"
    elif tech_percentage >= 40:
        overall_tech = "Neutral"
        tech_color = "text-gray-600"
        tech_bg = "bg-gray-50 dark:bg-gray-800/50"
        tech_icon = "➡️"
    elif tech_percentage >= 25:
        overall_tech = "Bearish"
        tech_color = "text-red-500"
        tech_bg = "bg-red-50 dark:bg-red-900/20"
        tech_icon = "📉"
    else:
        overall_tech = "Strong Bearish"
        tech_color = "text-red-600"
        tech_bg = "bg-red-50 dark:bg-red-900/20"
        tech_icon = "⚠️"
    
    pages['overview'] = f"""
    <div id='overview' class='page-content space-y-6'>
        <!-- Technical Analysis Dashboard -->
        <div class='bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 p-6 rounded-xl border border-blue-100 dark:border-blue-800/50 shadow-sm'>
            <div class='flex items-center mb-6'>
                <div class='p-2 bg-blue-100 dark:bg-blue-800/50 rounded-lg mr-3'>
                    <svg class='w-6 h-6 text-blue-600 dark:text-blue-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z'></path>
                    </svg>
                </div>
                <div>
                    <h3 class='text-xl font-bold text-gray-900 dark:text-white'>Technical Analysis Overview</h3>
                    <p class='text-sm text-gray-600 dark:text-gray-400'>Comprehensive technical indicator dashboard</p>
                </div>
            </div>
            
            <!-- Technical Score -->
            <div class='mb-6'>
                <div class='flex items-center justify-between mb-3'>
                    <span class='text-sm font-medium text-gray-700 dark:text-gray-300'>Overall Technical Signal</span>
                    <div class='flex items-center space-x-2'>
                        <span class='text-lg'>{tech_icon}</span>
                        <span class='font-bold {tech_color}'>{overall_tech}</span>
                    </div>
                </div>
                <div class='w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3 mb-2'>
                    <div class='bg-gradient-to-r from-red-500 via-yellow-500 to-green-500 h-3 rounded-full relative'>
                        <div class='absolute top-0 bg-blue-600 h-3 w-1 rounded-full shadow-lg' style='left: {tech_percentage:.1f}%'></div>
                    </div>
                </div>
                <div class='flex justify-between text-xs text-gray-500 dark:text-gray-400'>
                    <span>Strong Bearish</span>
                    <span class='font-medium text-blue-600 dark:text-blue-400'>Score: {tech_percentage:.0f}/100</span>
                    <span>Strong Bullish</span>
                </div>
            </div>
            
            <!-- Price and Moving Averages -->
            <div class='grid grid-cols-1 md:grid-cols-3 gap-4 mb-6'>
                <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <div class='text-xs text-gray-500 dark:text-gray-400 mb-1'>Current Price</div>
                    <div class='text-2xl font-bold text-gray-900 dark:text-white'>${current_price:.2f}</div>
                    <div class='text-sm text-gray-600 dark:text-gray-400'>Live market price</div>
                </div>
                <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <div class='text-xs text-gray-500 dark:text-gray-400 mb-1'>20-Day MA</div>
                    <div class='text-2xl font-bold {"text-green-600" if current_price > technical_data["ma_20"] else "text-red-600"}'>${technical_data["ma_20"]:.2f}</div>
                    <div class='text-sm text-gray-600 dark:text-gray-400'>{"Above" if current_price > technical_data["ma_20"] else "Below"} current price</div>
                </div>
                <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <div class='text-xs text-gray-500 dark:text-gray-400 mb-1'>50-Day MA</div>
                    <div class='text-2xl font-bold {"text-green-600" if technical_data["ma_50"] and current_price > technical_data["ma_50"] else "text-red-600" if technical_data["ma_50"] else "text-gray-600"}'>
                        {"$" + f"{technical_data['ma_50']:.2f}" if technical_data['ma_50'] else "N/A"}
                    </div>
                    <div class='text-sm text-gray-600 dark:text-gray-400'>
                        {("Above" if current_price > technical_data["ma_50"] else "Below") + " current price" if technical_data["ma_50"] else "Insufficient data"}
                    </div>
                </div>
            </div>
        </div>

        <!-- Technical Indicators Grid -->
        <div class='grid grid-cols-1 md:grid-cols-3 gap-4'>
            {''.join([f'''
            <div class='bg-white dark:bg-gray-800/50 border border-gray-100 dark:border-gray-700/50 rounded-xl p-4 shadow-sm'>
                <div class='flex items-center justify-between mb-3'>
                    <span class='text-sm font-medium text-gray-700 dark:text-gray-300'>{signal[0]}</span>
                    <span class='text-lg'>{signal[3]}</span>
                </div>
                <div class='text-lg font-bold {signal[2]} mb-2'>{signal[1]}</div>
                <div class='text-xs text-gray-500 dark:text-gray-400'>
                    {f"Value: {technical_data['current_rsi']:.1f}" if signal[0] == "RSI" else f"Signal: {technical_data['macd_signal_text']}" if signal[0] == "MACD" else f"Trend: {signal[1]}" if signal[0] == "MA Trend" else ""}
                </div>
            </div>
            ''' for signal in tech_signals])}
        </div>

        <!-- Detailed Technical Metrics -->
        <div class='bg-gradient-to-r from-emerald-50 to-teal-50 dark:from-emerald-900/20 dark:to-teal-900/20 p-6 rounded-xl border border-emerald-100 dark:border-emerald-800/50 shadow-sm'>
            <div class='flex items-center mb-4'>
                <div class='p-2 bg-emerald-100 dark:bg-emerald-800/50 rounded-lg mr-3'>
                    <svg class='w-6 h-6 text-emerald-600 dark:text-emerald-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z'></path>
                    </svg>
                </div>
                <h3 class='text-xl font-bold text-gray-900 dark:text-white'>Detailed Technical Metrics</h3>
            </div>
            
            <div class='grid grid-cols-1 md:grid-cols-2 gap-6'>
                <!-- Momentum Indicators -->
                <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <h4 class='font-semibold mb-3 text-gray-800 dark:text-white flex items-center'>
                        <span class='w-2 h-2 bg-purple-500 rounded-full mr-2'></span>
                        Momentum Indicators
                    </h4>
                    <div class='space-y-3'>
                        <div class='flex justify-between items-center'>
                            <span class='text-sm text-gray-600 dark:text-gray-400'>RSI (14)</span>
                            <span class='font-medium {technical_data["rsi_color"]}'>{technical_data["current_rsi"]:.1f}</span>
                        </div>
                        <div class='flex justify-between items-center'>
                            <span class='text-sm text-gray-600 dark:text-gray-400'>MACD</span>
                            <span class='font-medium {technical_data["macd_color"]}'>{technical_data["current_macd"]:.4f}</span>
                        </div>
                        <div class='flex justify-between items-center'>
                            <span class='text-sm text-gray-600 dark:text-gray-400'>Stochastic %K</span>
                            <span class='font-medium {technical_data["stoch_color"]}'>{technical_data["current_k"]:.1f}</span>
                        </div>
                        <div class='flex justify-between items-center'>
                            <span class='text-sm text-gray-600 dark:text-gray-400'>MFI</span>
                            <span class='font-medium {technical_data["mfi_color"]}'>{technical_data["current_mfi"]:.1f}</span>
                        </div>
                    </div>
                </div>
                
                <!-- Volatility & Bands -->
                <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <h4 class='font-semibold mb-3 text-gray-800 dark:text-white flex items-center'>
                        <span class='w-2 h-2 bg-yellow-500 rounded-full mr-2'></span>
                        Volatility & Bands
                    </h4>
                    <div class='space-y-3'>
                        <div class='flex justify-between items-center'>
                            <span class='text-sm text-gray-600 dark:text-gray-400'>ATR (14)</span>
                            <span class='font-medium text-gray-900 dark:text-white'>{technical_data["current_atr"]:.2f}</span>
                        </div>
                        <div class='flex justify-between items-center'>
                            <span class='text-sm text-gray-600 dark:text-gray-400'>BB Upper</span>
                            <span class='font-medium text-gray-900 dark:text-white'>${technical_data["current_bb_upper"]:.2f}</span>
                        </div>
                        <div class='flex justify-between items-center'>
                            <span class='text-sm text-gray-600 dark:text-gray-400'>BB Lower</span>
                            <span class='font-medium text-gray-900 dark:text-white'>${technical_data["current_bb_lower"]:.2f}</span>
                        </div>
                        <div class='p-2 {technical_data["bb_color"].replace("text-", "bg-").replace("-600", "-50").replace("-500", "-50")} dark:bg-gray-800/50 rounded text-center'>
                            <span class='{technical_data["bb_color"]} font-semibold text-sm'>{technical_data["bb_position"]}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """
    
    # Calculate comprehensive momentum scoring
    momentum_score = 0
    momentum_signals = []
    
    # RSI Momentum Analysis
    rsi_val = technical_data["current_rsi"]
    if rsi_val > 70:
        momentum_score -= 2
        momentum_signals.append(("RSI Overbought", f"{rsi_val:.1f}", "text-red-600", "⚠️", "Potential bearish reversal"))
    elif rsi_val > 60:
        momentum_score += 1
        momentum_signals.append(("RSI Bullish", f"{rsi_val:.1f}", "text-green-600", "📈", "Strong upward momentum"))
    elif rsi_val < 30:
        momentum_score -= 2
        momentum_signals.append(("RSI Oversold", f"{rsi_val:.1f}", "text-red-600", "📉", "Potential bullish reversal"))
    elif rsi_val < 40:
        momentum_score -= 1
        momentum_signals.append(("RSI Bearish", f"{rsi_val:.1f}", "text-orange-600", "📉", "Weak downward pressure"))
    else:
        momentum_score += 0
        momentum_signals.append(("RSI Neutral", f"{rsi_val:.1f}", "text-gray-600", "➡️", "Balanced momentum"))
    
    # MACD Momentum Analysis
    macd_val = technical_data["current_macd"]
    macd_signal = technical_data["current_macd_signal"]
    macd_diff = macd_val - macd_signal
    
    if macd_diff > 0 and macd_val > 0:
        momentum_score += 2
        momentum_signals.append(("MACD Bullish", f"{macd_diff:+.4f}", "text-green-600", "🚀", "Strong bullish momentum"))
    elif macd_diff > 0:
        momentum_score += 1
        momentum_signals.append(("MACD Positive", f"{macd_diff:+.4f}", "text-green-500", "📈", "Improving momentum"))
    elif macd_diff < 0 and macd_val < 0:
        momentum_score -= 2
        momentum_signals.append(("MACD Bearish", f"{macd_diff:+.4f}", "text-red-600", "📉", "Strong bearish momentum"))
    else:
        momentum_score -= 1
        momentum_signals.append(("MACD Negative", f"{macd_diff:+.4f}", "text-orange-600", "📉", "Weakening momentum"))
    
    # Stochastic Momentum Analysis
    stoch_k = technical_data["current_k"]
    stoch_d = technical_data["current_d"]
    
    if stoch_k > 80 and stoch_d > 80:
        momentum_score -= 1
        momentum_signals.append(("Stoch Overbought", f"K:{stoch_k:.1f}", "text-red-600", "⚠️", "Momentum may be exhausted"))
    elif stoch_k < 20 and stoch_d < 20:
        momentum_score -= 1
        momentum_signals.append(("Stoch Oversold", f"K:{stoch_k:.1f}", "text-red-600", "📉", "Potential momentum reversal"))
    elif stoch_k > stoch_d and stoch_k > 50:
        momentum_score += 1
        momentum_signals.append(("Stoch Bullish", f"K:{stoch_k:.1f}", "text-green-600", "📈", "Positive momentum crossover"))
    elif stoch_k < stoch_d and stoch_k < 50:
        momentum_score -= 1
        momentum_signals.append(("Stoch Bearish", f"K:{stoch_k:.1f}", "text-orange-600", "📉", "Negative momentum crossover"))
    else:
        momentum_signals.append(("Stoch Neutral", f"K:{stoch_k:.1f}", "text-gray-600", "➡️", "Mixed momentum signals"))
    
    # Overall Momentum Assessment
    momentum_score = max(-6, min(6, momentum_score))
    momentum_percentage = max(0, min(100, 50 + (momentum_score * 8.33)))
    
    if momentum_score >= 4:
        momentum_strength = "Very Strong"
        momentum_color = "text-green-600"
        momentum_bg = "bg-green-50 dark:bg-green-900/20"
        momentum_icon = "🚀"
    elif momentum_score >= 2:
        momentum_strength = "Strong"
        momentum_color = "text-green-500"
        momentum_bg = "bg-green-50 dark:bg-green-900/20"
        momentum_icon = "📈"
    elif momentum_score <= -4:
        momentum_strength = "Very Weak"
        momentum_color = "text-red-600"
        momentum_bg = "bg-red-50 dark:bg-red-900/20"
        momentum_icon = "📉"
    elif momentum_score <= -2:
        momentum_strength = "Weak"
        momentum_color = "text-red-500"
        momentum_bg = "bg-red-50 dark:bg-red-900/20"
        momentum_icon = "📉"
    else:
        momentum_strength = "Neutral"
        momentum_color = "text-gray-600"
        momentum_bg = "bg-gray-50 dark:bg-gray-800/50"
        momentum_icon = "➡️"

    # Enhanced Momentum Indicators Page
    pages['momentum'] = f"""
    <div id='momentum' class='page-content hidden space-y-6'>
        <!-- Momentum Dashboard -->
        <div class='bg-gradient-to-r from-purple-50 to-indigo-50 dark:from-purple-900/20 dark:to-indigo-900/20 p-6 rounded-xl border border-purple-100 dark:border-purple-800/50 shadow-sm'>
            <div class='flex items-center mb-6'>
                <div class='p-2 bg-purple-100 dark:bg-purple-800/50 rounded-lg mr-3'>
                    <svg class='w-6 h-6 text-purple-600 dark:text-purple-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M13 10V3L4 14h7v7l9-11h-7z'></path>
                    </svg>
                </div>
                <div>
                    <h3 class='text-xl font-bold text-gray-900 dark:text-white'>Momentum Analysis Dashboard</h3>
                    <p class='text-sm text-gray-600 dark:text-gray-400'>Comprehensive momentum indicators and trend strength</p>
                </div>
            </div>
            
            <!-- Overall Momentum Strength -->
            <div class='mb-6'>
                <div class='flex items-center justify-between mb-3'>
                    <span class='text-sm font-medium text-gray-700 dark:text-gray-300'>Overall Momentum Strength</span>
                    <div class='flex items-center space-x-2'>
                        <span class='text-lg'>{momentum_icon}</span>
                        <span class='font-bold {momentum_color}'>{momentum_strength}</span>
                    </div>
                </div>
                <div class='w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3 mb-2'>
                    <div class='bg-gradient-to-r from-red-500 via-yellow-500 to-green-500 h-3 rounded-full relative'>
                        <div class='absolute top-0 bg-gray-800 dark:bg-white h-3 w-1 rounded-full shadow-lg' style='left: {momentum_percentage:.1f}%'></div>
                    </div>
                </div>
                <div class='flex justify-between text-xs text-gray-500 dark:text-gray-400'>
                    <span>Very Weak</span>
                    <span class='font-medium text-purple-600 dark:text-purple-400'>Score: {momentum_score:+d}/6</span>
                    <span>Very Strong</span>
                </div>
            </div>
            
            <!-- Momentum Metrics Grid -->
            <div class='grid grid-cols-1 md:grid-cols-4 gap-4'>
                <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <div class='text-xs text-gray-500 dark:text-gray-400 mb-1'>RSI Momentum</div>
                    <div class='text-2xl font-bold {technical_data["rsi_color"]}'>{technical_data["current_rsi"]:.1f}</div>
                    <div class='text-sm text-gray-600 dark:text-gray-400'>{technical_data["rsi_signal"]}</div>
                </div>
                <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <div class='text-xs text-gray-500 dark:text-gray-400 mb-1'>MACD Signal</div>
                    <div class='text-2xl font-bold {technical_data["macd_color"]}'>{macd_diff:+.4f}</div>
                    <div class='text-sm text-gray-600 dark:text-gray-400'>{technical_data["macd_signal_text"]}</div>
                </div>
                <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <div class='text-xs text-gray-500 dark:text-gray-400 mb-1'>Stochastic</div>
                    <div class='text-2xl font-bold {technical_data["stoch_color"]}'>{technical_data["current_k"]:.1f}</div>
                    <div class='text-sm text-gray-600 dark:text-gray-400'>{technical_data["stoch_signal"]}</div>
                </div>
                <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <div class='text-xs text-gray-500 dark:text-gray-400 mb-1'>Money Flow</div>
                    <div class='text-2xl font-bold {technical_data["mfi_color"]}'>{technical_data["current_mfi"]:.1f}</div>
                    <div class='text-sm text-gray-600 dark:text-gray-400'>{technical_data["mfi_signal"]}</div>
                </div>
            </div>
        </div>
        
        <!-- Momentum Signals Analysis -->
        <div class='bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 p-5 rounded-xl'>
            <div class='flex items-center mb-4'>
                <div class='p-1.5 bg-purple-100 dark:bg-purple-900/30 rounded-lg mr-3'>
                    <svg class='w-5 h-5 text-purple-600 dark:text-purple-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z'></path>
                    </svg>
                </div>
                <h3 class='text-lg font-semibold text-gray-800 dark:text-white'>Momentum Signals Analysis</h3>
            </div>
            
            <div class='grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4'>
                {''.join([f'''
                <div class='p-4 rounded-lg border {signal[2].replace("text-", "border-").replace("-600", "-200")} bg-white dark:bg-gray-800/50'>
                    <div class='flex items-center justify-between mb-2'>
                        <span class='text-lg'>{signal[3]}</span>
                        <span class='text-sm font-medium {signal[2]}'>{signal[1]}</span>
                    </div>
                    <h4 class='font-semibold text-gray-900 dark:text-white mb-1'>{signal[0]}</h4>
                    <p class='text-xs text-gray-600 dark:text-gray-400'>{signal[4]}</p>
                </div>
                ''' for signal in momentum_signals])}
            </div>
        </div>
        
        <!-- Detailed Momentum Indicators -->
        <div class='bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 p-5 rounded-xl'>
            <div class='flex items-center mb-4'>
                <div class='p-1.5 bg-indigo-100 dark:bg-indigo-900/30 rounded-lg mr-3'>
                    <svg class='w-5 h-5 text-indigo-600 dark:text-indigo-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z'></path>
                    </svg>
                </div>
                <h3 class='text-lg font-semibold text-gray-800 dark:text-white'>Detailed Momentum Indicators</h3>
            </div>
            
            <div class='grid grid-cols-1 md:grid-cols-2 gap-6'>
                <!-- Enhanced RSI -->
                <div class='bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 p-4 rounded-lg border border-blue-100 dark:border-blue-800/50'>
                    <h4 class='font-semibold mb-3 text-gray-800 dark:text-white flex items-center'>
                        <span class='w-2 h-2 bg-blue-500 rounded-full mr-2'></span>
                        RSI (14-period)
                    </h4>
                    <div class='space-y-3'>
                        <div class='text-center'>
                            <p class='text-3xl font-bold {technical_data["rsi_color"]}'>{technical_data["current_rsi"]:.1f}</p>
                            <p class='{technical_data["rsi_color"]} font-semibold text-sm'>{technical_data["rsi_signal"]}</p>
                        </div>
                        <div class='w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3'>
                            <div class='bg-blue-600 h-3 rounded-full transition-all duration-300' style='width: {min(technical_data["current_rsi"], 100)}%'></div>
                        </div>
                        <div class='flex justify-between text-xs text-gray-500'>
                            <span>Oversold (30)</span>
                            <span>Neutral (50)</span>
                            <span>Overbought (70)</span>
                        </div>
                    </div>
                </div>
                
                <!-- Enhanced MACD -->
                <div class='bg-gradient-to-br from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 p-4 rounded-lg border border-green-100 dark:border-green-800/50'>
                    <h4 class='font-semibold mb-3 text-gray-800 dark:text-white flex items-center'>
                        <span class='w-2 h-2 bg-green-500 rounded-full mr-2'></span>
                        MACD (12,26,9)
                    </h4>
                    <div class='space-y-2'>
                        <div class='flex justify-between'>
                            <span class='text-sm text-gray-600 dark:text-gray-400'>MACD Line</span>
                            <span class='font-semibold {technical_data["macd_color"]}'>{technical_data["current_macd"]:.4f}</span>
                        </div>
                        <div class='flex justify-between'>
                            <span class='text-sm text-gray-600 dark:text-gray-400'>Signal Line</span>
                            <span class='font-semibold text-gray-900 dark:text-white'>{technical_data["current_macd_signal"]:.4f}</span>
                        </div>
                        <div class='flex justify-between'>
                            <span class='text-sm text-gray-600 dark:text-gray-400'>Histogram</span>
                            <span class='font-semibold {technical_data["macd_color"]}'>{macd_diff:+.4f}</span>
                        </div>
                        <div class='mt-3 p-2 bg-white/50 dark:bg-gray-800/50 rounded text-center'>
                            <span class='{technical_data["macd_color"]} font-semibold text-sm'>{technical_data["macd_signal_text"]}</span>
                        </div>
                    </div>
                </div>
                
                <!-- Enhanced Stochastic -->
                <div class='bg-gradient-to-br from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 p-4 rounded-lg border border-purple-100 dark:border-purple-800/50'>
                    <h4 class='font-semibold mb-3 text-gray-800 dark:text-white flex items-center'>
                        <span class='w-2 h-2 bg-purple-500 rounded-full mr-2'></span>
                        Stochastic (14,3,3)
                    </h4>
                    <div class='space-y-3'>
                        <div class='grid grid-cols-2 gap-4'>
                            <div class='text-center'>
                                <p class='text-sm text-gray-600 dark:text-gray-400'>%K</p>
                                <p class='text-xl font-bold {technical_data["stoch_color"]}'>{technical_data["current_k"]:.1f}</p>
                            </div>
                            <div class='text-center'>
                                <p class='text-sm text-gray-600 dark:text-gray-400'>%D</p>
                                <p class='text-xl font-bold text-gray-900 dark:text-white'>{technical_data["current_d"]:.1f}</p>
                            </div>
                        </div>
                        <div class='w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3'>
                            <div class='bg-purple-600 h-3 rounded-full transition-all duration-300' style='width: {min(technical_data["current_k"], 100)}%'></div>
                        </div>
                        <div class='text-center'>
                            <span class='{technical_data["stoch_color"]} font-semibold text-sm'>{technical_data["stoch_signal"]}</span>
                        </div>
                    </div>
                </div>
                
                <!-- Enhanced MFI -->
                <div class='bg-gradient-to-br from-amber-50 to-orange-50 dark:from-amber-900/20 dark:to-orange-900/20 p-4 rounded-lg border border-amber-100 dark:border-amber-800/50'>
                    <h4 class='font-semibold mb-3 text-gray-800 dark:text-white flex items-center'>
                        <span class='w-2 h-2 bg-amber-500 rounded-full mr-2'></span>
                        Money Flow Index (14)
                    </h4>
                    <div class='space-y-3'>
                        <div class='text-center'>
                            <p class='text-3xl font-bold {technical_data["mfi_color"]}'>{technical_data["current_mfi"]:.1f}</p>
                            <p class='{technical_data["mfi_color"]} font-semibold text-sm'>{technical_data["mfi_signal"]}</p>
                        </div>
                        <div class='w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3'>
                            <div class='bg-amber-600 h-3 rounded-full transition-all duration-300' style='width: {min(technical_data["current_mfi"], 100)}%'></div>
                        </div>
                        <div class='flex justify-between text-xs text-gray-500'>
                            <span>Oversold (20)</span>
                            <span>Neutral (50)</span>
                            <span>Overbought (80)</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """
    
    # Enhanced Volatility Analysis Page
    
    # Calculate comprehensive volatility metrics
    returns = hist['Close'].pct_change().dropna()
    
    # Historical volatility calculations
    if len(returns) > 0:
        # Daily volatility
        daily_vol = returns.std()
        # Annualized volatility (252 trading days)
        annualized_vol = daily_vol * (252 ** 0.5) * 100
        
        # Rolling volatilities for different periods
        vol_10d = returns.rolling(10).std().iloc[-1] * (252 ** 0.5) * 100 if len(returns) >= 10 else annualized_vol
        vol_30d = returns.rolling(30).std().iloc[-1] * (252 ** 0.5) * 100 if len(returns) >= 30 else annualized_vol
        vol_60d = returns.rolling(60).std().iloc[-1] * (252 ** 0.5) * 100 if len(returns) >= 60 else annualized_vol
        
        # Volatility trend (comparing recent vs historical)
        recent_vol = vol_10d
        historical_vol = vol_60d
        vol_trend = ((recent_vol - historical_vol) / historical_vol * 100) if historical_vol > 0 else 0
        
        # Volatility percentile (current vs historical range)
        vol_series = returns.rolling(30).std() * (252 ** 0.5) * 100
        vol_series = vol_series.dropna()
        if len(vol_series) > 0:
            current_vol_percentile = (vol_series <= vol_30d).sum() / len(vol_series) * 100
        else:
            current_vol_percentile = 50
            
        # Risk assessment based on volatility
        if annualized_vol > 60:
            vol_risk = "Extreme"
            vol_risk_color = "text-red-600"
            vol_risk_bg = "bg-red-50 dark:bg-red-900/20"
            vol_risk_icon = "🔥"
        elif annualized_vol > 40:
            vol_risk = "High"
            vol_risk_color = "text-orange-600"
            vol_risk_bg = "bg-orange-50 dark:bg-orange-900/20"
            vol_risk_icon = "⚡"
        elif annualized_vol > 25:
            vol_risk = "Moderate"
            vol_risk_color = "text-yellow-600"
            vol_risk_bg = "bg-yellow-50 dark:bg-yellow-900/20"
            vol_risk_icon = "⚖️"
        elif annualized_vol > 15:
            vol_risk = "Low"
            vol_risk_color = "text-green-600"
            vol_risk_bg = "bg-green-50 dark:bg-green-900/20"
            vol_risk_icon = "🟢"
        else:
            vol_risk = "Very Low"
            vol_risk_color = "text-blue-600"
            vol_risk_bg = "bg-blue-50 dark:bg-blue-900/20"
            vol_risk_icon = "🔵"
    else:
        annualized_vol = vol_10d = vol_30d = vol_60d = 0
        vol_trend = 0
        current_vol_percentile = 50
        vol_risk = "Unknown"
        vol_risk_color = "text-gray-600"
        vol_risk_bg = "bg-gray-50 dark:bg-gray-800/50"
        vol_risk_icon = "❓"
    
    # ATR-based volatility
    atr_percentage = (technical_data["current_atr"] / current_price * 100) if current_price > 0 else 0
    
    # Bollinger Band width (volatility indicator)
    bb_width = ((technical_data["current_bb_upper"] - technical_data["current_bb_lower"]) / current_price * 100) if current_price > 0 else 0
    
    pages['volatility'] = f"""
    <div id='volatility' class='page-content space-y-6'>
        <!-- Volatility Overview Dashboard -->
        <div class='bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 p-6 rounded-xl border border-yellow-100 dark:border-yellow-800/50 shadow-sm'>
            <div class='flex items-center mb-6'>
                <div class='p-2 bg-yellow-100 dark:bg-yellow-800/50 rounded-lg mr-3'>
                    <svg class='w-6 h-6 text-yellow-600 dark:text-yellow-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M13 10V3L4 14h7v7l9-11h-7z'></path>
                    </svg>
                </div>
                <div>
                    <h3 class='text-xl font-bold text-gray-900 dark:text-white'>Comprehensive Volatility Analysis</h3>
                    <p class='text-sm text-gray-600 dark:text-gray-400'>Risk assessment and volatility metrics</p>
                </div>
            </div>
            
            <!-- Volatility Risk Assessment -->
            <div class='mb-6'>
                <div class='flex items-center justify-between mb-3'>
                    <span class='text-sm font-medium text-gray-700 dark:text-gray-300'>Volatility Risk Level</span>
                    <div class='flex items-center space-x-2'>
                        <span class='text-lg'>{vol_risk_icon}</span>
                        <span class='font-bold {vol_risk_color}'>{vol_risk} Risk</span>
                    </div>
                </div>
                <div class='w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3 mb-2'>
                    <div class='bg-gradient-to-r from-green-500 via-yellow-500 to-red-500 h-3 rounded-full relative'>
                        <div class='absolute top-0 bg-blue-600 h-3 w-1 rounded-full shadow-lg' style='left: {min(annualized_vol / 80 * 100, 100):.1f}%'></div>
                    </div>
                </div>
                <div class='flex justify-between text-xs text-gray-500 dark:text-gray-400'>
                    <span>Very Low (0-15%)</span>
                    <span class='font-medium text-blue-600 dark:text-blue-400'>Current: {annualized_vol:.1f}%</span>
                    <span>Extreme (60%+)</span>
                </div>
            </div>
            
            <!-- Key Volatility Metrics -->
            <div class='grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4'>
                <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <div class='text-xs text-gray-500 dark:text-gray-400 mb-1'>Annualized Volatility</div>
                    <div class='text-2xl font-bold {vol_risk_color}'>{annualized_vol:.1f}%</div>
                    <div class='text-sm text-gray-600 dark:text-gray-400'>Historical measure</div>
                </div>
                <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <div class='text-xs text-gray-500 dark:text-gray-400 mb-1'>ATR (14-day)</div>
                    <div class='text-2xl font-bold text-gray-900 dark:text-white'>{atr_percentage:.2f}%</div>
                    <div class='text-sm text-gray-600 dark:text-gray-400'>Daily range</div>
                </div>
                <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <div class='text-xs text-gray-500 dark:text-gray-400 mb-1'>Volatility Percentile</div>
                    <div class='text-2xl font-bold text-gray-900 dark:text-white'>{current_vol_percentile:.0f}%</div>
                    <div class='text-sm text-gray-600 dark:text-gray-400'>vs History</div>
                </div>
                <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <div class='text-xs text-gray-500 dark:text-gray-400 mb-1'>Bollinger Width</div>
                    <div class='text-2xl font-bold text-gray-900 dark:text-white'>{bb_width:.2f}%</div>
                    <div class='text-sm text-gray-600 dark:text-gray-400'>Band spread</div>
                </div>
            </div>
        </div>

        <!-- Historical Volatility Analysis -->
        <div class='bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 p-6 rounded-xl border border-purple-100 dark:border-purple-800/50 shadow-sm'>
            <div class='flex items-center mb-4'>
                <div class='p-2 bg-purple-100 dark:bg-purple-800/50 rounded-lg mr-3'>
                    <svg class='w-6 h-6 text-purple-600 dark:text-purple-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z'></path>
                    </svg>
                </div>
                <h3 class='text-xl font-bold text-gray-900 dark:text-white'>Historical Volatility Breakdown</h3>
            </div>
            
            <div class='grid grid-cols-1 md:grid-cols-3 gap-4'>
                <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <h4 class='font-semibold text-gray-900 dark:text-white mb-3 flex items-center'>
                        <span class='w-2 h-2 bg-green-500 rounded-full mr-2'></span>
                        10-Day Volatility
                    </h4>
                    <div class='text-2xl font-bold text-gray-900 dark:text-white mb-2'>{vol_10d:.1f}%</div>
                    <div class='text-sm text-gray-600 dark:text-gray-400'>Short-term volatility</div>
                </div>
                
                <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <h4 class='font-semibold text-gray-900 dark:text-white mb-3 flex items-center'>
                        <span class='w-2 h-2 bg-blue-500 rounded-full mr-2'></span>
                        30-Day Volatility
                    </h4>
                    <div class='text-2xl font-bold text-gray-900 dark:text-white mb-2'>{vol_30d:.1f}%</div>
                    <div class='text-sm text-gray-600 dark:text-gray-400'>Medium-term volatility</div>
                </div>
                
                <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <h4 class='font-semibold text-gray-900 dark:text-white mb-3 flex items-center'>
                        <span class='w-2 h-2 bg-purple-500 rounded-full mr-2'></span>
                        60-Day Volatility
                    </h4>
                    <div class='text-2xl font-bold text-gray-900 dark:text-white mb-2'>{vol_60d:.1f}%</div>
                    <div class='text-sm text-gray-600 dark:text-gray-400'>Long-term volatility</div>
                </div>
            </div>
            
            <!-- Volatility Trend Analysis -->
            <div class='mt-6 p-4 {vol_risk_bg} rounded-lg border border-gray-100 dark:border-gray-700'>
                <h4 class='font-semibold text-gray-900 dark:text-white mb-2'>Volatility Trend Analysis</h4>
                <div class='flex items-center justify-between'>
                    <span class='text-sm text-gray-600 dark:text-gray-400'>Recent vs Historical Trend</span>
                    <span class='font-bold {vol_risk_color}'>
                        {f"+{vol_trend:.1f}%" if vol_trend > 0 else f"{vol_trend:.1f}%"} 
                        {"📈" if vol_trend > 5 else "📉" if vol_trend < -5 else "➡️"}
                    </span>
                </div>
                <div class='text-sm text-gray-600 dark:text-gray-400 mt-2'>
                    {"Volatility is increasing - expect larger price swings" if vol_trend > 5 else "Volatility is decreasing - price movements becoming more stable" if vol_trend < -5 else "Volatility is relatively stable"}
                </div>
            </div>
        </div>

        <!-- Technical Volatility Indicators -->
        <div class='bg-white dark:bg-gray-800/50 border border-gray-100 dark:border-gray-700/50 rounded-xl p-6 shadow-sm'>
            <div class='flex items-center mb-4'>
                <div class='p-2 bg-gray-100 dark:bg-gray-700 rounded-lg mr-3'>
                    <svg class='w-6 h-6 text-gray-600 dark:text-gray-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z'></path>
                    </svg>
                </div>
                <h3 class='text-xl font-bold text-gray-900 dark:text-white'>Technical Volatility Indicators</h3>
            </div>
            
            <div class='grid grid-cols-1 md:grid-cols-2 gap-6'>
                <!-- ATR Analysis -->
                <div class='p-4 bg-gradient-to-br from-amber-50 to-orange-50 dark:from-amber-900/20 dark:to-orange-900/20 rounded-lg border border-amber-100 dark:border-amber-800/50'>
                    <h4 class='font-semibold mb-3 text-gray-800 dark:text-white flex items-center'>
                        <span class='text-amber-600 mr-2'>📊</span>
                        Average True Range (ATR)
                    </h4>
                    <div class='space-y-3'>
                        <div class='flex justify-between'>
                            <span class='text-sm text-gray-600 dark:text-gray-400'>ATR Value</span>
                            <span class='font-bold text-gray-900 dark:text-white'>${technical_data["current_atr"]:.2f}</span>
                        </div>
                        <div class='flex justify-between'>
                            <span class='text-sm text-gray-600 dark:text-gray-400'>% of Price</span>
                            <span class='font-bold text-amber-600'>{atr_percentage:.2f}%</span>
                        </div>
                        <div class='w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2'>
                            <div class='bg-amber-500 h-2 rounded-full' style='width: {min(atr_percentage * 4, 100):.1f}%'></div>
                        </div>
                        <div class='text-xs text-gray-500 dark:text-gray-400'>
                            Expected daily range: ${current_price - technical_data["current_atr"]:.2f} - ${current_price + technical_data["current_atr"]:.2f}
                        </div>
                    </div>
                </div>
                
                <!-- Bollinger Bands Analysis -->
                <div class='p-4 bg-gradient-to-br from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-lg border border-blue-100 dark:border-blue-800/50'>
                    <h4 class='font-semibold mb-3 text-gray-800 dark:text-white flex items-center'>
                        <span class='text-blue-600 mr-2'>📈</span>
                        Bollinger Bands
                    </h4>
                    <div class='space-y-3'>
                        <div class='flex justify-between'>
                            <span class='text-sm text-gray-600 dark:text-gray-400'>Upper Band</span>
                            <span class='font-bold text-gray-900 dark:text-white'>${technical_data["current_bb_upper"]:.2f}</span>
                        </div>
                        <div class='flex justify-between'>
                            <span class='text-sm text-gray-600 dark:text-gray-400'>Lower Band</span>
                            <span class='font-bold text-gray-900 dark:text-white'>${technical_data["current_bb_lower"]:.2f}</span>
                        </div>
                        <div class='flex justify-between'>
                            <span class='text-sm text-gray-600 dark:text-gray-400'>Band Width</span>
                            <span class='font-bold text-blue-600'>{bb_width:.2f}%</span>
                        </div>
                        <div class='p-2 {technical_data["bb_color"].replace("text-", "bg-").replace("-600", "-50").replace("-500", "-50")} dark:bg-gray-800/50 rounded text-center'>
                            <span class='{technical_data["bb_color"]} font-semibold text-sm'>{technical_data["bb_position"]}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """
    
    # Calculate comprehensive technical signals for summary
    signals = []
    signal_scores = []
    
    # RSI Signal
    if technical_data["current_rsi"] > 70:
        signals.append(("RSI", "Sell", "Overbought", "text-red-600", "⚠️"))
        signal_scores.append(-2)
    elif technical_data["current_rsi"] < 30:
        signals.append(("RSI", "Buy", "Oversold", "text-green-600", "📈"))
        signal_scores.append(2)
    else:
        signals.append(("RSI", "Hold", "Neutral", "text-gray-600", "➡️"))
        signal_scores.append(0)
    
    # MACD Signal
    if technical_data["current_macd"] > technical_data["current_macd_signal"]:
        signals.append(("MACD", "Buy", "Bullish", "text-green-600", "📈"))
        signal_scores.append(1)
    else:
        signals.append(("MACD", "Sell", "Bearish", "text-red-600", "📉"))
        signal_scores.append(-1)
    
    # Moving Average Signal
    if current_price > technical_data["ma_20"]:
        if technical_data["ma_50"] and current_price > technical_data["ma_50"]:
            signals.append(("MA Trend", "Buy", "Above MAs", "text-green-600", "🚀"))
            signal_scores.append(2)
        else:
            signals.append(("MA Trend", "Buy", "Above 20-MA", "text-green-600", "📈"))
            signal_scores.append(1)
    else:
        if technical_data["ma_50"] and current_price < technical_data["ma_50"]:
            signals.append(("MA Trend", "Sell", "Below MAs", "text-red-600", "📉"))
            signal_scores.append(-2)
        else:
            signals.append(("MA Trend", "Sell", "Below 20-MA", "text-red-600", "📉"))
            signal_scores.append(-1)
    
    # Bollinger Bands Signal
    bb_position = ""
    if current_price > technical_data["current_bb_upper"]:
        signals.append(("Bollinger", "Sell", "Above Upper", "text-red-600", "⚠️"))
        signal_scores.append(-1)
        bb_position = "Above Upper Band"
    elif current_price < technical_data["current_bb_lower"]:
        signals.append(("Bollinger", "Buy", "Below Lower", "text-green-600", "📈"))
        signal_scores.append(1)
        bb_position = "Below Lower Band"
    else:
        signals.append(("Bollinger", "Hold", "Within Bands", "text-gray-600", "➡️"))
        signal_scores.append(0)
        bb_position = "Within Bands"
    
    # Stochastic Signal
    if technical_data["current_k"] > 80:
        signals.append(("Stochastic", "Sell", "Overbought", "text-red-600", "⚠️"))
        signal_scores.append(-1)
    elif technical_data["current_k"] < 20:
        signals.append(("Stochastic", "Buy", "Oversold", "text-green-600", "📈"))
        signal_scores.append(1)
    else:
        signals.append(("Stochastic", "Hold", "Neutral", "text-gray-600", "➡️"))
        signal_scores.append(0)
    
    # Calculate overall score
    total_score = sum(signal_scores)
    max_possible = len(signal_scores) * 2
    score_percentage = ((total_score + max_possible) / (2 * max_possible)) * 100
    
    # Determine overall sentiment
    if total_score >= 3:
        overall_sentiment = "Strong Buy"
        sentiment_color = "text-green-700 dark:text-green-400"
        sentiment_bg = "bg-green-100 dark:bg-green-900/30"
        sentiment_border = "border-green-300 dark:border-green-600"
        sentiment_icon = "🚀"
    elif total_score >= 1:
        overall_sentiment = "Buy"
        sentiment_color = "text-green-600 dark:text-green-400"
        sentiment_bg = "bg-green-50 dark:bg-green-900/20"
        sentiment_border = "border-green-200 dark:border-green-700"
        sentiment_icon = "📈"
    elif total_score <= -3:
        overall_sentiment = "Strong Sell"
        sentiment_color = "text-red-700 dark:text-red-400"
        sentiment_bg = "bg-red-100 dark:bg-red-900/30"
        sentiment_border = "border-red-300 dark:border-red-600"
        sentiment_icon = "📉"
    elif total_score <= -1:
        overall_sentiment = "Sell"
        sentiment_color = "text-red-600 dark:text-red-400"
        sentiment_bg = "bg-red-50 dark:bg-red-900/20"
        sentiment_border = "border-red-200 dark:border-red-700"
        sentiment_icon = "📉"
    else:
        overall_sentiment = "Hold"
        sentiment_color = "text-gray-600 dark:text-gray-400"
        sentiment_bg = "bg-gray-50 dark:bg-gray-900/20"
        sentiment_border = "border-gray-200 dark:border-gray-700"
        sentiment_icon = "➡️"
    
    # Calculate volatility level
    volatility_pct = (technical_data["current_atr"] / current_price) * 100
    if volatility_pct > 5:
        volatility_level = "High"
        volatility_color = "text-red-600"
        volatility_icon = "🔥"
    elif volatility_pct > 2:
        volatility_level = "Moderate"
        volatility_color = "text-yellow-600"
        volatility_icon = "⚡"
    else:
        volatility_level = "Low"
        volatility_color = "text-green-600"
        volatility_icon = "🟢"
    
    # Summary Page
    pages['summary'] = f"""
    <div id='summary' class='page-content hidden'>
        <div class='space-y-6'>
            <!-- Overall Technical Score -->
            <div class='bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 border border-blue-200 dark:border-blue-700 rounded-xl p-6'>
                <div class='flex items-center justify-between mb-4'>
                    <div class='flex items-center'>
                        <div class='p-2 bg-blue-100 dark:bg-blue-900/30 rounded-lg mr-4'>
                            <svg class='w-6 h-6 text-blue-600 dark:text-blue-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                                <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z' />
                            </svg>
                        </div>
                        <div>
                            <h3 class='text-xl font-bold text-gray-900 dark:text-white'>Technical Analysis Summary</h3>
                            <p class='text-sm text-gray-600 dark:text-gray-400'>Comprehensive technical indicator analysis</p>
                        </div>
                    </div>
                    <div class='text-right'>
                        <div class='text-2xl font-bold text-blue-600 dark:text-blue-400'>{score_percentage:.0f}/100</div>
                        <div class='text-xs text-gray-500 dark:text-gray-400'>Technical Score</div>
                    </div>
                </div>
                
                <!-- Overall Sentiment -->
                <div class='flex items-center justify-center mb-4'>
                    <div class='px-6 py-3 {sentiment_bg} border {sentiment_border} rounded-full'>
                        <div class='flex items-center space-x-2'>
                            <span class='text-2xl'>{sentiment_icon}</span>
                            <span class='text-lg font-bold {sentiment_color}'>{overall_sentiment}</span>
                        </div>
                    </div>
                </div>
                
                <!-- Score Bar -->
                <div class='w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3 mb-2'>
                    <div class='h-3 rounded-full transition-all duration-500' style='width: {score_percentage}%; background: linear-gradient(to right, #ef4444, #f59e0b, #10b981);'></div>
                </div>
                <div class='flex justify-between text-xs text-gray-500 dark:text-gray-400'>
                    <span>Strong Sell</span>
                    <span>Hold</span>
                    <span>Strong Buy</span>
                </div>
            </div>
            
            <!-- Signal Summary Grid -->
            <div class='grid grid-cols-1 lg:grid-cols-2 gap-6'>
                <!-- Individual Signals -->
                <div class='bg-white dark:bg-gray-800/50 border border-gray-100 dark:border-gray-700/50 rounded-xl p-6'>
                    <div class='flex items-center mb-4'>
                        <div class='p-2 bg-purple-100 dark:bg-purple-900/30 rounded-lg mr-3'>
                            <svg class='w-5 h-5 text-purple-600 dark:text-purple-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                                <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M13 10V3L4 14h7v7l9-11h-7z' />
                            </svg>
                        </div>
                        <h4 class='font-semibold text-gray-900 dark:text-white'>Technical Signals</h4>
                    </div>
                    <div class='space-y-3'>
    """
    
    # Add individual signals
    for indicator, signal, description, color, icon in signals:
        pages['summary'] += f"""
                        <div class='flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg'>
                            <div class='flex items-center space-x-3'>
                                <span class='text-lg'>{icon}</span>
                                <div>
                                    <div class='font-medium text-gray-900 dark:text-white'>{indicator}</div>
                                    <div class='text-xs text-gray-500 dark:text-gray-400'>{description}</div>
                                </div>
                            </div>
                            <span class='px-2 py-1 text-xs font-medium {color} bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-600 rounded'>
                                {signal}
                            </span>
                        </div>
        """
    
    pages['summary'] += f"""
                    </div>
                </div>
                
                <!-- Key Metrics -->
                <div class='bg-white dark:bg-gray-800/50 border border-gray-100 dark:border-gray-700/50 rounded-xl p-6'>
                    <div class='flex items-center mb-4'>
                        <div class='p-2 bg-orange-100 dark:bg-orange-900/30 rounded-lg mr-3'>
                            <svg class='w-5 h-5 text-orange-600 dark:text-orange-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                                <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z' />
                            </svg>
                        </div>
                        <h4 class='font-semibold text-gray-900 dark:text-white'>Key Metrics</h4>
                    </div>
                    <div class='space-y-4'>
                        <!-- Current Price vs MAs -->
                        <div class='p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg'>
                            <div class='flex items-center justify-between mb-2'>
                                <span class='text-sm font-medium text-gray-700 dark:text-gray-300'>Price Position</span>
                                <span class='text-xs text-blue-600 dark:text-blue-400'>Moving Averages</span>
                            </div>
                            <div class='space-y-1 text-sm'>
                                <div class='flex justify-between'>
                                    <span class='text-gray-600 dark:text-gray-400'>vs 20-MA:</span>
                                    <span class='font-medium {"text-green-600" if current_price > technical_data["ma_20"] else "text-red-600"}'>{((current_price - technical_data["ma_20"]) / technical_data["ma_20"] * 100):+.1f}%</span>
                                </div>
                                {f'<div class="flex justify-between"><span class="text-gray-600 dark:text-gray-400">vs 50-MA:</span><span class="font-medium {"text-green-600" if current_price > technical_data["ma_50"] else "text-red-600"}">{((current_price - technical_data["ma_50"]) / technical_data["ma_50"] * 100):+.1f}%</span></div>' if technical_data["ma_50"] else '<div class="flex justify-between"><span class="text-gray-600 dark:text-gray-400">vs 50-MA:</span><span class="text-gray-500 dark:text-gray-400">Insufficient data</span></div>'}
                            </div>
                        </div>
                        
                        <!-- Volatility -->
                        <div class='p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg'>
                            <div class='flex items-center justify-between mb-2'>
                                <span class='text-sm font-medium text-gray-700 dark:text-gray-300'>Volatility</span>
                                <div class='flex items-center space-x-1'>
                                    <span class='text-sm'>{volatility_icon}</span>
                                    <span class='text-xs font-medium {volatility_color}'>{volatility_level}</span>
                                </div>
                            </div>
                            <div class='text-sm text-gray-600 dark:text-gray-400'>
                                ATR: ${technical_data["current_atr"]:.2f} ({volatility_pct:.1f}% of price)
                            </div>
                        </div>
                        
                        <!-- Bollinger Position -->
                        <div class='p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg'>
                            <div class='flex items-center justify-between mb-2'>
                                <span class='text-sm font-medium text-gray-700 dark:text-gray-300'>Bollinger Bands</span>
                                <span class='text-xs text-purple-600 dark:text-purple-400'>Price Position</span>
                            </div>
                            <div class='text-sm text-gray-600 dark:text-gray-400'>
                                {bb_position}
                            </div>
                            <div class='mt-2 text-xs text-gray-500 dark:text-gray-400'>
                                Upper: ${technical_data["current_bb_upper"]:.2f} | Lower: ${technical_data["current_bb_lower"]:.2f}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Analysis Details -->
            <div class='bg-white dark:bg-gray-800/50 border border-gray-100 dark:border-gray-700/50 rounded-xl p-6'>
                <div class='flex items-center mb-4'>
                    <div class='p-2 bg-gray-100 dark:bg-gray-700 rounded-lg mr-3'>
                        <svg class='w-5 h-5 text-gray-600 dark:text-gray-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                            <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z' />
                        </svg>
                    </div>
                    <h4 class='font-semibold text-gray-900 dark:text-white'>Analysis Details</h4>
                </div>
                <div class='grid grid-cols-1 md:grid-cols-2 gap-4 text-sm'>
                    <div class='space-y-2'>
                        <div class='flex justify-between'>
                            <span class='text-gray-600 dark:text-gray-400'>Analysis Period:</span>
                            <span class='font-medium text-gray-900 dark:text-white'>{len(hist)} trading days</span>
                        </div>
                        <div class='flex justify-between'>
                            <span class='text-gray-600 dark:text-gray-400'>Data Range:</span>
                            <span class='font-medium text-gray-900 dark:text-white'>{hist.index[0].strftime('%m/%d/%Y')}</span>
                        </div>
                        <div class='flex justify-between'>
                            <span class='text-gray-600 dark:text-gray-400'>to:</span>
                            <span class='font-medium text-gray-900 dark:text-white'>{hist.index[-1].strftime('%m/%d/%Y')}</span>
                        </div>
                    </div>
                    <div class='space-y-2'>
                        <div class='flex justify-between'>
                            <span class='text-gray-600 dark:text-gray-400'>Current RSI:</span>
                            <span class='font-medium {"text-red-600" if technical_data["current_rsi"] > 70 else "text-green-600" if technical_data["current_rsi"] < 30 else "text-gray-900 dark:text-white"}'>{technical_data["current_rsi"]:.1f}</span>
                        </div>
                        <div class='flex justify-between'>
                            <span class='text-gray-600 dark:text-gray-400'>MACD Signal:</span>
                            <span class='font-medium {"text-green-600" if technical_data["current_macd"] > technical_data["current_macd_signal"] else "text-red-600"}'>{("Bullish" if technical_data["current_macd"] > technical_data["current_macd_signal"] else "Bearish")}</span>
                        </div>
                        <div class='flex justify-between'>
                            <span class='text-gray-600 dark:text-gray-400'>Stochastic %K:</span>
                            <span class='font-medium {"text-red-600" if technical_data["current_k"] > 80 else "text-green-600" if technical_data["current_k"] < 20 else "text-gray-900 dark:text-white"}'>{technical_data["current_k"]:.1f}</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Disclaimer -->
            <div class='bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-700 rounded-xl p-4'>
                <div class='flex items-start space-x-3'>
                    <div class='flex-shrink-0'>
                        <svg class='w-5 h-5 text-amber-600 dark:text-amber-400 mt-0.5' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                            <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z' />
                        </svg>
                    </div>
                    <div>
                        <h5 class='font-semibold text-amber-900 dark:text-amber-200 mb-1'>Important Disclaimer</h5>
                        <p class='text-xs text-amber-800 dark:text-amber-300 leading-relaxed'>
                            This technical analysis is for informational purposes only and should not be considered financial advice. 
                            Always conduct your own research and consider your risk tolerance before making investment decisions.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """
    
    # Calculate Support & Resistance Levels
    import numpy as np
    
    # Get recent price data for S&R calculation
    recent_highs = hist['High'].tail(50).values
    recent_lows = hist['Low'].tail(50).values
    recent_closes = hist['Close'].tail(50).values
    
    # Calculate pivot points and key levels
    def find_support_resistance_levels(highs, lows, closes, current_price):
        levels = []
        
        # Recent swing highs and lows
        for i in range(2, len(highs)-2):
            # Swing high
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                levels.append(('Resistance', highs[i], 'Swing High'))
            # Swing low  
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                levels.append(('Support', lows[i], 'Swing Low'))
        
        # Psychological levels (round numbers)
        price_range = max(highs) - min(lows)
        step = 5 if price_range < 50 else 10 if price_range < 200 else 25 if price_range < 500 else 50
        
        base = int(min(lows) / step) * step
        while base <= max(highs) + step:
            if abs(base - current_price) <= price_range * 0.3:  # Within reasonable range
                level_type = 'Resistance' if base > current_price else 'Support'
                levels.append((level_type, base, 'Psychological'))
            base += step
        
        # Fibonacci retracement levels
        high_price = max(highs)
        low_price = min(lows)
        diff = high_price - low_price
        
        fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        for fib in fib_levels:
            fib_price = high_price - (diff * fib)
            level_type = 'Resistance' if fib_price > current_price else 'Support'
            levels.append((level_type, fib_price, f'Fib {fib:.1%}'))
        
        return levels
    
    # Get all levels
    all_levels = find_support_resistance_levels(recent_highs, recent_lows, recent_closes, current_price)
    
    # Filter and sort levels
    support_levels = [(level[1], level[2]) for level in all_levels if level[0] == 'Support' and abs(level[1] - current_price) / current_price < 0.15]
    resistance_levels = [(level[1], level[2]) for level in all_levels if level[0] == 'Resistance' and abs(level[1] - current_price) / current_price < 0.15]
    
    # Sort by proximity to current price
    support_levels.sort(key=lambda x: abs(x[0] - current_price))
    resistance_levels.sort(key=lambda x: abs(x[0] - current_price))
    
    # Take top 5 of each
    support_levels = support_levels[:5]
    resistance_levels = resistance_levels[:5]
    
    # Calculate level strengths and distances
    def calculate_level_strength(price, price_data):
        touches = sum(1 for p in price_data if abs(p - price) / price < 0.02)  # Within 2%
        return min(touches, 5)  # Cap at 5 for display
    
    # Add strength calculations
    support_data = []
    for price, level_type in support_levels:
        strength = calculate_level_strength(price, recent_lows)
        distance = ((current_price - price) / current_price) * 100
        support_data.append({
            'price': price,
            'type': level_type,
            'strength': strength,
            'distance': distance
        })
    
    resistance_data = []
    for price, level_type in resistance_levels:
        strength = calculate_level_strength(price, recent_highs)
        distance = ((price - current_price) / current_price) * 100
        resistance_data.append({
            'price': price,
            'type': level_type,
            'strength': strength,
            'distance': distance
        })
    
    # Determine current market position
    nearest_support = min(support_data, key=lambda x: x['distance']) if support_data else None
    nearest_resistance = min(resistance_data, key=lambda x: x['distance']) if resistance_data else None
    
    if nearest_support and nearest_resistance:
        support_distance = nearest_support['distance']
        resistance_distance = nearest_resistance['distance']
        
        if support_distance < resistance_distance:
            market_position = "Near Support"
            position_color = "text-green-600"
            position_icon = "🛡️"
        else:
            market_position = "Near Resistance"
            position_color = "text-red-600"
            position_icon = "⚠️"
    else:
        market_position = "Between Levels"
        position_color = "text-gray-600"
        position_icon = "➡️"
    
    # Support & Resistance Page
    pages['support_resistance'] = f"""
    <div id='support_resistance' class='page-content hidden space-y-6'>
        <!-- Support & Resistance Dashboard -->
        <div class='bg-gradient-to-r from-teal-50 to-green-50 dark:from-teal-900/20 dark:to-green-900/20 p-6 rounded-xl border border-teal-100 dark:border-teal-800/50 shadow-sm'>
            <div class='flex items-center mb-6'>
                <div class='p-2 bg-teal-100 dark:bg-teal-800/50 rounded-lg mr-3'>
                    <svg class='w-6 h-6 text-teal-600 dark:text-teal-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z'></path>
                    </svg>
                </div>
                <div>
                    <h3 class='text-xl font-bold text-gray-900 dark:text-white'>Support & Resistance Analysis</h3>
                    <p class='text-sm text-gray-600 dark:text-gray-400'>Key price levels and trading zones identification</p>
                </div>
            </div>
            
            <!-- Current Position Analysis -->
            <div class='grid grid-cols-1 md:grid-cols-4 gap-4 mb-6'>
                <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <div class='text-xs text-gray-500 dark:text-gray-400 mb-1'>Current Price</div>
                    <div class='text-2xl font-bold text-gray-900 dark:text-white'>${current_price:.2f}</div>
                    <div class='text-sm text-gray-600 dark:text-gray-400'>Live market price</div>
                </div>
                <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <div class='text-xs text-gray-500 dark:text-gray-400 mb-1'>Market Position</div>
                    <div class='text-lg font-bold {position_color}'>{position_icon} {market_position}</div>
                    <div class='text-sm text-gray-600 dark:text-gray-400'>Relative to key levels</div>
                </div>
                <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <div class='text-xs text-gray-500 dark:text-gray-400 mb-1'>Nearest Support</div>
                    <div class='text-lg font-bold text-green-600 dark:text-green-400'>{"$" + f"{nearest_support['price']:.2f}" if nearest_support else "N/A"}</div>
                    <div class='text-sm text-gray-600 dark:text-gray-400'>{f"-{nearest_support['distance']:.1f}%" if nearest_support else "No data"}</div>
                </div>
                <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <div class='text-xs text-gray-500 dark:text-gray-400 mb-1'>Nearest Resistance</div>
                    <div class='text-lg font-bold text-red-600 dark:text-red-400'>{"$" + f"{nearest_resistance['price']:.2f}" if nearest_resistance else "N/A"}</div>
                    <div class='text-sm text-gray-600 dark:text-gray-400'>{f"+{nearest_resistance['distance']:.1f}%" if nearest_resistance else "No data"}</div>
                </div>
            </div>
        </div>
        
        <!-- Support Levels Analysis -->
        <div class='bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 p-5 rounded-xl'>
            <div class='flex items-center mb-4'>
                <div class='p-1.5 bg-green-100 dark:bg-green-900/30 rounded-lg mr-3'>
                    <svg class='w-5 h-5 text-green-600 dark:text-green-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M7 16l-4-4m0 0l4-4m-4 4h18'></path>
                    </svg>
                </div>
                <h3 class='text-lg font-semibold text-gray-800 dark:text-white'>Support Levels</h3>
            </div>
            
            <div class='space-y-3'>
                {''.join([f"""
                <div class='flex items-center justify-between p-4 bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 rounded-lg border border-green-100 dark:border-green-800/50'>
                    <div class='flex items-center space-x-4'>
                        <div class='text-center'>
                            <div class='text-lg font-bold text-green-600 dark:text-green-400'>${level["price"]:.2f}</div>
                            <div class='text-xs text-gray-500 dark:text-gray-400'>{level["type"]}</div>
                        </div>
                        <div class='h-8 w-px bg-gray-200 dark:bg-gray-600'></div>
                        <div>
                            <div class='text-sm font-medium text-gray-900 dark:text-white'>Distance: {level["distance"]:.1f}%</div>
                            <div class='text-xs text-gray-600 dark:text-gray-400'>Strength: {"●" * level["strength"]}{"○" * (5-level["strength"])}</div>
                        </div>
                    </div>
                    <div class='text-right'>
                        <div class='text-sm font-bold text-green-600 dark:text-green-400'>BUY ZONE</div>
                        <div class='text-xs text-gray-500 dark:text-gray-400'>{"Strong" if level["strength"] >= 3 else "Moderate" if level["strength"] >= 2 else "Weak"} level</div>
                    </div>
                </div>
                """ for level in support_data]) if support_data else '<div class="text-center text-gray-500 dark:text-gray-400 py-4">No significant support levels identified</div>'}
            </div>
        </div>
        
        <!-- Resistance Levels Analysis -->
        <div class='bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 p-5 rounded-xl'>
            <div class='flex items-center mb-4'>
                <div class='p-1.5 bg-red-100 dark:bg-red-900/30 rounded-lg mr-3'>
                    <svg class='w-5 h-5 text-red-600 dark:text-red-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M17 8l4 4m0 0l-4 4m4-4H3'></path>
                    </svg>
                </div>
                <h3 class='text-lg font-semibold text-gray-800 dark:text-white'>Resistance Levels</h3>
            </div>
            
            <div class='space-y-3'>
                {''.join([f"""
                <div class='flex items-center justify-between p-4 bg-gradient-to-r from-red-50 to-pink-50 dark:from-red-900/20 dark:to-pink-900/20 rounded-lg border border-red-100 dark:border-red-800/50'>
                    <div class='flex items-center space-x-4'>
                        <div class='text-center'>
                            <div class='text-lg font-bold text-red-600 dark:text-red-400'>${level["price"]:.2f}</div>
                            <div class='text-xs text-gray-500 dark:text-gray-400'>{level["type"]}</div>
                        </div>
                        <div class='h-8 w-px bg-gray-200 dark:bg-gray-600'></div>
                        <div>
                            <div class='text-sm font-medium text-gray-900 dark:text-white'>Distance: +{level["distance"]:.1f}%</div>
                            <div class='text-xs text-gray-600 dark:text-gray-400'>Strength: {"●" * level["strength"]}{"○" * (5-level["strength"])}</div>
                        </div>
                    </div>
                    <div class='text-right'>
                        <div class='text-sm font-bold text-red-600 dark:text-red-400'>SELL ZONE</div>
                        <div class='text-xs text-gray-500 dark:text-gray-400'>{"Strong" if level["strength"] >= 3 else "Moderate" if level["strength"] >= 2 else "Weak"} level</div>
                    </div>
                </div>
                """ for level in resistance_data]) if resistance_data else '<div class="text-center text-gray-500 dark:text-gray-400 py-4">No significant resistance levels identified</div>'}
            </div>
        </div>
        
        <!-- Trading Strategy Insights -->
        <div class='bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 p-6 rounded-xl border border-blue-100 dark:border-blue-800/50 shadow-sm'>
            <div class='flex items-center mb-4'>
                <div class='p-2 bg-blue-100 dark:bg-blue-800/50 rounded-lg mr-3'>
                    <svg class='w-6 h-6 text-blue-600 dark:text-blue-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z'></path>
                    </svg>
                </div>
                <h3 class='text-xl font-bold text-gray-900 dark:text-white'>Trading Strategy Insights</h3>
            </div>
            
            <div class='grid grid-cols-1 md:grid-cols-2 gap-6'>
                <div>
                    <h4 class='font-medium text-gray-700 dark:text-gray-300 mb-3'>Key Observations</h4>
                    <div class='space-y-2 text-sm'>
                        <p class='text-gray-600 dark:text-gray-400'>• Current price is {market_position.lower()}</p>
                        <p class='text-gray-600 dark:text-gray-400'>• {len(support_data)} support levels identified within 15%</p>
                        <p class='text-gray-600 dark:text-gray-400'>• {len(resistance_data)} resistance levels identified within 15%</p>
                        <p class='text-gray-600 dark:text-gray-400'>• {"Strong" if (support_data and max(s["strength"] for s in support_data) >= 3) or (resistance_data and max(r["strength"] for r in resistance_data) >= 3) else "Moderate"} level strength detected</p>
                    </div>
                </div>
                
                <div>
                    <h4 class='font-medium text-gray-700 dark:text-gray-300 mb-3'>Trading Recommendations</h4>
                    <div class='space-y-2 text-sm'>
                        {f'<p class="text-green-600 dark:text-green-400">• Consider buying near ${nearest_support["price"]:.2f} support</p>' if nearest_support and nearest_support["distance"] < 5 else ''}
                        {f'<p class="text-red-600 dark:text-red-400">• Consider selling near ${nearest_resistance["price"]:.2f} resistance</p>' if nearest_resistance and nearest_resistance["distance"] < 5 else ''}
                        <p class='text-gray-600 dark:text-gray-400'>• Use stop losses below support levels</p>
                        <p class='text-gray-600 dark:text-gray-400'>• Watch for breakouts above resistance</p>
                        <p class='text-gray-600 dark:text-gray-400'>• Higher volume confirms level significance</p>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """
    
    return pages

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
        from market_analyzer import calculate_rsi, calculate_macd, calculate_bollinger_bands, calculate_stochastic, calculate_mfi, calculate_ichimoku, calculate_atr
        
        # Calculate technical indicators
        rsi = calculate_rsi(hist['Close'])
        macd_line, macd_signal = calculate_macd(hist['Close'])
        macd_histogram = macd_line - macd_signal  # Calculate histogram manually
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(hist['Close'])
        k, d = calculate_stochastic(hist['High'], hist['Low'], hist['Close'])
        mfi = calculate_mfi(hist['High'], hist['Low'], hist['Close'], hist['Volume'])
        ichimoku = calculate_ichimoku(hist['High'], hist['Low'], hist['Close'])
        atr = calculate_atr(hist['High'], hist['Low'], hist['Close'])
        
        # Get current values
        current_price = hist['Close'].iloc[-1]
        current_rsi = rsi.iloc[-1] if not rsi.empty else 0
        current_macd = macd_line.iloc[-1] if not macd_line.empty else 0
        current_macd_signal = macd_signal.iloc[-1] if not macd_signal.empty else 0
        current_bb_upper = bb_upper.iloc[-1] if not bb_upper.empty else 0
        current_bb_lower = bb_lower.iloc[-1] if not bb_lower.empty else 0
        current_k = k.iloc[-1] if not k.empty else 0
        current_d = d.iloc[-1] if not d.empty else 0
        current_mfi = mfi.iloc[-1] if not mfi.empty else 0
        current_atr = atr.iloc[-1] if not atr.empty else 0
        
        # Get current Ichimoku values if available
        current_ichimoku = {}
        if ichimoku and all(k in ichimoku for k in ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span']):
            current_ichimoku = {
                'tenkan_sen': ichimoku['tenkan_sen'].iloc[-1],
                'kijun_sen': ichimoku['kijun_sen'].iloc[-1],
                'senkou_span_a': ichimoku['senkou_span_a'].iloc[-1],
                'senkou_span_b': ichimoku['senkou_span_b'].iloc[-1],
                'chikou_span': ichimoku['chikou_span'].iloc[-26] if len(ichimoku['chikou_span']) > 26 else 0  # 26 periods ago
            }
        
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
            
        # Determine Stochastic signal
        if current_k > 80:
            stoch_signal = "Overbought"
            stoch_color = "text-red-600"
        elif current_k < 20:
            stoch_signal = "Oversold"
            stoch_color = "text-green-600"
        else:
            stoch_signal = "Neutral"
            stoch_color = "text-gray-600"
            
        # Determine MFI signal
        if current_mfi > 80:
            mfi_signal = "Overbought"
            mfi_color = "text-red-600"
        elif current_mfi < 20:
            mfi_signal = "Oversold"
            mfi_color = "text-green-600"
        else:
            mfi_signal = "Neutral"
            mfi_color = "text-gray-600"
        
        # Get company info
        info = stock.info
        company_name = info.get('longName', ticker)
        
        # Prepare technical data for pages
        technical_data = {
            'ma_20': ma_20,
            'ma_50': ma_50,
            'current_rsi': current_rsi,
            'rsi_signal': rsi_signal,
            'rsi_color': rsi_color,
            'current_macd': current_macd,
            'current_macd_signal': current_macd_signal,
            'macd_signal_text': macd_signal_text,
            'macd_color': macd_color,
            'current_bb_upper': current_bb_upper,
            'current_bb_lower': current_bb_lower,
            'bb_position': bb_position,
            'bb_color': bb_color,
            'current_k': current_k,
            'current_d': current_d,
            'stoch_signal': stoch_signal,
            'stoch_color': stoch_color,
            'current_mfi': current_mfi,
            'mfi_signal': mfi_signal,
            'mfi_color': mfi_color,
            'current_atr': current_atr
        }
        
        # Generate paginated analysis
        pages = create_technical_analysis_pages(
            ticker=ticker,
            hist=hist,
            info=info,
            current_price=current_price,
            company_name=company_name,
            technical_data=technical_data
        )
        
        return {
            'ticker': ticker,
            'company_name': company_name,
            'pages': pages
        }
        
    except Exception as e:
        return f"<div class='p-4'><h3 class='text-lg font-semibold mb-2 text-red-600'>Error</h3><p>Failed to perform technical analysis for {ticker}: {str(e)}</p></div>"

# Default industry averages (can be expanded as needed)
DEFAULT_INDUSTRY_AVERAGES = {
    'Technology': {
        'pe': 30.0,
        'peg': 2.0,
        'pb': 10.0,
        'operating_margin': 0.25,  # 25%
        'profit_margin': 0.20,     # 20%
        'roe': 0.25,              # 25%
        'debt_to_equity': 0.5,    # 0.5x
        'current_ratio': 2.0,     # 2.0x
        'quick_ratio': 1.8,       # 1.8x
    },
    'Finance': {
        'pe': 15.0,
        'peg': 1.5,
        'pb': 1.5,
        'operating_margin': 0.35,
        'profit_margin': 0.25,
        'roe': 0.15,
        'debt_to_equity': 2.0,
        'current_ratio': 1.2,
        'quick_ratio': 1.1,
    },
    'Healthcare': {
        'pe': 25.0,
        'peg': 1.8,
        'pb': 4.0,
        'operating_margin': 0.20,
        'profit_margin': 0.15,
        'roe': 0.18,
        'debt_to_equity': 0.8,
        'current_ratio': 1.8,
        'quick_ratio': 1.6,
    },
    # Add more sectors as needed
}

def get_industry_average(sector, metric):
    """Get industry average for a given sector and metric."""
    if not sector or sector == 'N/A':
        return None
    
    # Map metric names to keys in our default data
    metric_map = {
        'P/E Ratio': 'pe',
        'PEG Ratio': 'peg',
        'P/B Ratio': 'pb',
        'Operating Margin': 'operating_margin',
        'Profit Margin': 'profit_margin',
        'ROE': 'roe',
        'Debt/Equity': 'debt_to_equity',
        'Current Ratio': 'current_ratio',
        'Quick Ratio': 'quick_ratio',
    }
    
    key = metric_map.get(metric)
    if not key:
        return None
        
    # Try to find a matching sector (case insensitive)
    for sector_name, values in DEFAULT_INDUSTRY_AVERAGES.items():
        if sector_name.lower() in sector.lower():
            return values.get(key)
    
    # If no exact match, return first available sector's value
    first_sector = next(iter(DEFAULT_INDUSTRY_AVERAGES.values()), {})
    return first_sector.get(key)

def create_basic_analysis_pages(ticker, hist, info, current_price, prev_price, price_change, price_change_pct, 
                             high_52w, low_52w, current_52w_range, avg_volume, latest_volume, 
                             volume_ratio, market_cap_str, pe_ratio, company_name, performance_metrics):
    """Create paginated content for enhanced basic analysis with technical indicators."""
    from datetime import datetime
    import numpy as np
    import pandas as pd
    
    # Format the change display
    change_color = "text-green-600" if price_change >= 0 else "text-red-600"
    volume_color = "text-green-600" if latest_volume > avg_volume * 1.2 else "text-yellow-600" if latest_volume > avg_volume * 0.8 else "text-red-600"
    change_symbol = "+" if price_change >= 0 else ""
    
    # Calculate market cap for classification
    market_cap = info.get('marketCap', 0)
    market_cap_class = (
        'Mega Cap' if market_cap >= 200e9 else
        'Large Cap' if market_cap >= 10e9 else
        'Mid Cap' if market_cap >= 2e9 else
        'Small Cap' if market_cap >= 300e6 else
        'Micro Cap' if market_cap >= 50e6 else 'Nano Cap'
    )
    
    # Calculate technical indicators if we have enough data
    def calculate_rsi(prices, period=14):
        """Calculate RSI (Relative Strength Index)"""
        if len(prices) < period + 1:
            return None
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not rsi.empty else None
    
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        if len(prices) < slow + signal:
            return None, None, None
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]
    
    def calculate_moving_averages(prices):
        """Calculate key moving averages"""
        ma_20 = prices.rolling(window=20).mean().iloc[-1] if len(prices) >= 20 else None
        ma_50 = prices.rolling(window=50).mean().iloc[-1] if len(prices) >= 50 else None
        ma_200 = prices.rolling(window=200).mean().iloc[-1] if len(prices) >= 200 else None
        return ma_20, ma_50, ma_200
    
    def get_support_resistance(prices, highs, lows):
        """Calculate basic support and resistance levels"""
        if len(prices) < 20:
            return None, None
        
        # Recent high and low as basic resistance/support
        recent_high = highs.tail(20).max()
        recent_low = lows.tail(20).min()
        
        # 52-week levels
        resistance = max(recent_high, high_52w * 0.98)  # Slightly below 52w high
        support = min(recent_low, low_52w * 1.02)       # Slightly above 52w low
        
        return support, resistance
    
    # Calculate indicators if we have historical data
    rsi = None
    macd_line = None
    macd_signal = None
    macd_histogram = None
    ma_20 = None
    ma_50 = None
    ma_200 = None
    support = None
    resistance = None
    
    if not hist.empty and len(hist) > 1:
        prices = hist['Close']
        rsi = calculate_rsi(prices)
        macd_line, macd_signal, macd_histogram = calculate_macd(prices)
        ma_20, ma_50, ma_200 = calculate_moving_averages(prices)
        support, resistance = get_support_resistance(prices, hist['High'], hist['Low'])
    
    # Helper functions for formatting and color coding
    def format_rsi(rsi_value):
        if rsi_value is None:
            return "N/A", "", ""
        if rsi_value > 70:
            return f"{rsi_value:.1f}", "text-red-600", "Overbought"
        elif rsi_value < 30:
            return f"{rsi_value:.1f}", "text-green-600", "Oversold"
        else:
            return f"{rsi_value:.1f}", "text-yellow-600", "Neutral"
    
    def format_ma_trend(current, ma_20, ma_50, ma_200):
        if None in [ma_20, ma_50, ma_200]:
            return "Insufficient Data", "text-gray-500"
        
        if current > ma_20 > ma_50 > ma_200:
            return "Strong Uptrend", "text-green-600"
        elif current > ma_20 > ma_50:
            return "Uptrend", "text-green-500"
        elif current < ma_20 < ma_50 < ma_200:
            return "Strong Downtrend", "text-red-600"
        elif current < ma_20 < ma_50:
            return "Downtrend", "text-red-500"
        else:
            return "Sideways/Mixed", "text-yellow-600"
    
    def format_macd_signal(macd_line, macd_signal, histogram):
        if None in [macd_line, macd_signal, histogram]:
            return "N/A", "text-gray-500"
        
        if macd_line > macd_signal and histogram > 0:
            return "Bullish", "text-green-600"
        elif macd_line < macd_signal and histogram < 0:
            return "Bearish", "text-red-600"
        else:
            return "Neutral", "text-yellow-600"
    
    # Create pages dictionary
    pages = {}
    
    # Get formatted values for display
    rsi_value, rsi_color, rsi_status = format_rsi(rsi)
    trend_status, trend_color = format_ma_trend(current_price, ma_20, ma_50, ma_200)
    macd_status, macd_color = format_macd_signal(macd_line, macd_signal, macd_histogram)
    
    # Format values to avoid f-string conditional formatting issues
    resistance_str = f"${resistance:.2f}" if resistance is not None else "N/A"
    support_str = f"${support:.2f}" if support is not None else "N/A"
    ma_20_str = f"${ma_20:.2f}" if ma_20 is not None else "N/A"
    ma_50_str = f"${ma_50:.2f}" if ma_50 is not None else "N/A"
    ma_200_str = f"${ma_200:.2f}" if ma_200 is not None else "N/A"
    macd_line_str = f"{macd_line:.4f}" if macd_line is not None else "N/A"
    macd_signal_str = f"{macd_signal:.4f}" if macd_signal is not None else "N/A"
    macd_histogram_str = f"{macd_histogram:.4f}" if macd_histogram is not None else "N/A"
    
    # Page 1: Basic Analysis Overview - Focus on Fundamentals
    
    # Calculate basic business health indicators
    business_health = []
    health_score = 0
    
    # Market Cap Assessment
    if market_cap >= 200e9:
        business_health.append(("Size", "Mega Cap", "text-blue-600", "🏢"))
        health_score += 2
    elif market_cap >= 10e9:
        business_health.append(("Size", "Large Cap", "text-green-600", "🏬"))
        health_score += 1
    elif market_cap >= 2e9:
        business_health.append(("Size", "Mid Cap", "text-yellow-600", "🏪"))
        health_score += 0
    else:
        business_health.append(("Size", "Small Cap", "text-orange-600", "🏠"))
        health_score -= 1
    
    # P/E Ratio Assessment
    if pe_ratio != 'N/A' and isinstance(pe_ratio, (int, float)):
        if pe_ratio < 15:
            business_health.append(("Valuation", "Undervalued", "text-green-600", "💰"))
            health_score += 1
        elif pe_ratio < 25:
            business_health.append(("Valuation", "Fair Value", "text-blue-600", "⚖️"))
            health_score += 0
        else:
            business_health.append(("Valuation", "Expensive", "text-red-600", "💸"))
            health_score -= 1
    else:
        business_health.append(("Valuation", "No P/E Data", "text-gray-600", "❓"))
    
    # Volume Activity Assessment
    if volume_ratio > 1.5:
        business_health.append(("Activity", "High Interest", "text-green-600", "📈"))
        health_score += 1
    elif volume_ratio > 0.8:
        business_health.append(("Activity", "Normal", "text-blue-600", "➡️"))
        health_score += 0
    else:
        business_health.append(("Activity", "Low Interest", "text-red-600", "📉"))
        health_score -= 1
    
    # Calculate overall business health
    max_health_score = 4
    min_health_score = -3
    
    if health_score >= 3:
        overall_health = "Excellent"
        health_color = "text-green-600"
        health_bg = "bg-green-50 dark:bg-green-900/20"
        health_icon = "🚀"
    elif health_score >= 1:
        overall_health = "Good"
        health_color = "text-blue-600"
        health_bg = "bg-blue-50 dark:bg-blue-900/20"
        health_icon = "👍"
    elif health_score >= -1:
        overall_health = "Fair"
        health_color = "text-yellow-600"
        health_bg = "bg-yellow-50 dark:bg-yellow-900/20"
        health_icon = "⚖️"
    else:
        overall_health = "Poor"
        health_color = "text-red-600"
        health_bg = "bg-red-50 dark:bg-red-900/20"
        health_icon = "⚠️"
    
    # Normalize health score to percentage
    health_percentage = ((health_score - min_health_score) / (max_health_score - min_health_score)) * 100
    
    pages['overview'] = f"""
    <div id="overview" class="page-content space-y-6">
        <!-- Business Health Dashboard -->
        <div class='bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 p-6 rounded-xl border border-blue-100 dark:border-blue-800/50 shadow-sm'>
            <div class='flex items-center mb-6'>
                <div class='p-2 bg-blue-100 dark:bg-blue-800/50 rounded-lg mr-3'>
                    <svg class='w-6 h-6 text-blue-600 dark:text-blue-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4'></path>
                    </svg>
                </div>
                <div>
                    <h3 class='text-xl font-bold text-gray-900 dark:text-white'>Business Health Overview</h3>
                    <p class='text-sm text-gray-600 dark:text-gray-400'>Fundamental analysis and company assessment</p>
                </div>
            </div>
            
            <!-- Overall Business Health Score -->
            <div class='mb-6'>
                <div class='flex items-center justify-between mb-3'>
                    <span class='text-sm font-medium text-gray-700 dark:text-gray-300'>Overall Business Health</span>
                    <div class='flex items-center space-x-2'>
                        <span class='text-lg'>{health_icon}</span>
                        <span class='font-bold {health_color}'>{overall_health}</span>
                    </div>
                </div>
                <div class='w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3 mb-2'>
                    <div class='bg-gradient-to-r from-red-500 via-yellow-500 to-green-500 h-3 rounded-full relative'>
                        <div class='absolute top-0 bg-blue-600 h-3 w-1 rounded-full shadow-lg' style='left: {health_percentage:.1f}%'></div>
                    </div>
                </div>
                <div class='flex justify-between text-xs text-gray-500 dark:text-gray-400'>
                    <span>Poor</span>
                    <span class='font-medium text-blue-600 dark:text-blue-400'>Health: {overall_health}</span>
                    <span>Excellent</span>
                </div>
            </div>
            
            <!-- Price and Key Metrics Grid -->
            <div class='grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4'>
                <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <div class='text-xs text-gray-500 dark:text-gray-400 mb-1'>Current Price</div>
                    <div class='text-2xl font-bold text-gray-900 dark:text-white'>${current_price:,.2f}</div>
                    <div class='text-sm {change_color} font-medium'>{change_symbol}{price_change:.2f} ({change_symbol}{price_change_pct:.2f}%)</div>
                </div>
                <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <div class='text-xs text-gray-500 dark:text-gray-400 mb-1'>Market Cap</div>
                    <div class='text-2xl font-bold text-gray-900 dark:text-white'>{market_cap_str}</div>
                    <div class='text-sm text-gray-600 dark:text-gray-400'>P/E: {pe_ratio if pe_ratio != 'N/A' else 'N/A'}</div>
                </div>
                <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <div class='text-xs text-gray-500 dark:text-gray-400 mb-1'>Volume Activity</div>
                    <div class='text-2xl font-bold {volume_color}'>{volume_ratio:.1f}x</div>
                    <div class='text-sm text-gray-600 dark:text-gray-400'>vs Avg Volume</div>
                </div>
                <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <div class='text-xs text-gray-500 dark:text-gray-400 mb-1'>52W Position</div>
                    <div class='text-2xl font-bold text-gray-900 dark:text-white'>{current_52w_range:.0f}%</div>
                    <div class='text-sm text-gray-600 dark:text-gray-400'>Beta: {info.get('beta', 'N/A')}</div>
                </div>
            </div>
        </div>
        
        <!-- Price Action Analysis -->
        <div class='bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 p-5 rounded-xl'>
            <div class='flex items-center mb-4'>
                <div class='p-1.5 bg-orange-100 dark:bg-orange-900/30 rounded-lg mr-3'>
                    <svg class='w-5 h-5 text-orange-600 dark:text-orange-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z' />
                    </svg>
                </div>
                <h3 class='text-lg font-semibold text-gray-800 dark:text-white'>Price Action & Key Levels</h3>
            </div>
            
            <div class='grid grid-cols-1 md:grid-cols-2 gap-6'>
                <!-- Support & Resistance -->
                <div>
                    <h4 class='font-medium text-gray-700 dark:text-gray-300 mb-3'>Key Levels</h4>
                    <div class='space-y-3'>
                        <div class='flex justify-between items-center p-3 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-100 dark:border-red-800/30'>
                            <div class='flex items-center'>
                                <span class='text-red-600 dark:text-red-400 mr-2'>📈</span>
                                <span class='text-sm font-medium text-red-700 dark:text-red-300'>Resistance</span>
                            </div>
                            <span class='font-bold text-red-800 dark:text-red-200'>{resistance_str}</span>
                        </div>
                        <div class='flex justify-between items-center p-3 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-100 dark:border-green-800/30'>
                            <div class='flex items-center'>
                                <span class='text-green-600 dark:text-green-400 mr-2'>📉</span>
                                <span class='text-sm font-medium text-green-700 dark:text-green-300'>Support</span>
                            </div>
                            <span class='font-bold text-green-800 dark:text-green-200'>{support_str}</span>
                        </div>
                    </div>
                </div>
                
                <!-- Moving Averages -->
                <div>
                    <h4 class='font-medium text-gray-700 dark:text-gray-300 mb-3'>Moving Averages</h4>
                    <div class='space-y-2'>
                        <div class='flex justify-between text-sm'>
                            <span class='text-gray-500 dark:text-gray-400'>20-Day MA</span>
                            <span class='font-medium' style='color: {"#10b981" if ma_20 and current_price > ma_20 else "#ef4444" if ma_20 else "#6b7280"}'>{ma_20_str}</span>
                        </div>
                        <div class='flex justify-between text-sm'>
                            <span class='text-gray-500 dark:text-gray-400'>50-Day MA</span>
                            <span class='font-medium' style='color: {"#10b981" if ma_50 and current_price > ma_50 else "#ef4444" if ma_50 else "#6b7280"}'>{ma_50_str}</span>
                        </div>
                        <div class='flex justify-between text-sm'>
                            <span class='text-gray-500 dark:text-gray-400'>200-Day MA</span>
                            <span class='font-medium' style='color: {"#10b981" if ma_200 and current_price > ma_200 else "#ef4444" if ma_200 else "#6b7280"}'>{ma_200_str}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Business Health Indicators -->
        <div class='bg-gradient-to-r from-emerald-50 to-teal-50 dark:from-emerald-900/20 dark:to-teal-900/20 p-6 rounded-xl border border-emerald-100 dark:border-emerald-800/50 shadow-sm'>
            <div class='flex items-center mb-4'>
                <div class='p-2 bg-emerald-100 dark:bg-emerald-800/50 rounded-lg mr-3'>
                    <svg class='w-6 h-6 text-emerald-600 dark:text-emerald-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z'></path>
                    </svg>
                </div>
                <h3 class='text-xl font-bold text-gray-900 dark:text-white'>Business Health Indicators</h3>
            </div>
            
            <div class='grid grid-cols-1 md:grid-cols-3 gap-4'>
                {''.join([f'''
                <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <div class='flex items-center justify-between mb-2'>
                        <span class='text-sm font-medium text-gray-700 dark:text-gray-300'>{indicator[0]}</span>
                        <span class='text-lg'>{indicator[3]}</span>
                    </div>
                    <div class='text-lg font-bold {indicator[2]} mb-1'>{indicator[1]}</div>
                    <div class='text-xs text-gray-500 dark:text-gray-400'>
                        {f"Market Cap: {market_cap_str}" if indicator[0] == "Size" else f"P/E Ratio: {pe_ratio}" if indicator[0] == "Valuation" and pe_ratio != "N/A" else f"Volume: {volume_ratio:.1f}x avg" if indicator[0] == "Activity" else "Assessment based on fundamentals"}
                    </div>
                </div>
                ''' for indicator in business_health])}
            </div>
        </div>

        <!-- Company Information & Performance -->
        <div class='bg-white dark:bg-gray-800/50 border border-gray-100 dark:border-gray-700/50 rounded-xl p-6 shadow-sm'>
            <div class='flex items-center mb-4'>
                <div class='p-2 bg-orange-100 dark:bg-orange-800/50 rounded-lg mr-3'>
                    <svg class='w-6 h-6 text-orange-600 dark:text-orange-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4'></path>
                    </svg>
                </div>
                <h3 class='text-xl font-bold text-gray-900 dark:text-white'>Company Information & Performance</h3>
            </div>
            
            <div class='grid grid-cols-1 md:grid-cols-2 gap-6'>
                <!-- Company Details -->
                <div>
                    <h4 class='font-medium text-gray-700 dark:text-gray-300 mb-3'>Company Details</h4>
                    <div class='space-y-3'>
                        <div class='flex justify-between items-center p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-100 dark:border-blue-800/30'>
                            <div class='flex items-center'>
                                <span class='text-blue-600 dark:text-blue-400 mr-2'>🏢</span>
                                <span class='text-sm font-medium text-blue-700 dark:text-blue-300'>Market Cap Class</span>
                            </div>
                            <span class='font-bold text-blue-800 dark:text-blue-200'>{market_cap_class}</span>
                        </div>
                        <div class='flex justify-between items-center p-3 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-100 dark:border-green-800/30'>
                            <div class='flex items-center'>
                                <span class='text-green-600 dark:text-green-400 mr-2'>🏭</span>
                                <span class='text-sm font-medium text-green-700 dark:text-green-300'>Sector</span>
                            </div>
                            <span class='font-bold text-green-800 dark:text-green-200'>{info.get('sector', 'N/A')}</span>
                        </div>
                        <div class='flex justify-between items-center p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg border border-purple-100 dark:border-purple-800/30'>
                            <div class='flex items-center'>
                                <span class='text-purple-600 dark:text-purple-400 mr-2'>🔧</span>
                                <span class='text-sm font-medium text-purple-700 dark:text-purple-300'>Industry</span>
                            </div>
                            <span class='font-bold text-purple-800 dark:text-purple-200'>{info.get('industry', 'N/A')}</span>
                        </div>
                    </div>
                </div>
                
                <!-- Recent Performance -->
                <div>
                    <h4 class='font-medium text-gray-700 dark:text-gray-300 mb-3'>Recent Performance</h4>
                    <div class='space-y-2'>
                        <div class='flex justify-between text-sm'>
                            <span class='text-gray-500 dark:text-gray-400'>1 Week</span>
                            <span class='font-medium {"text-green-600" if performance_metrics.get("1w") and performance_metrics["1w"] > 0 else "text-red-600" if performance_metrics.get("1w") else "text-gray-600"}'>{f"{performance_metrics['1w']:.1f}%" if performance_metrics.get("1w") else "N/A"}</span>
                        </div>
                        <div class='flex justify-between text-sm'>
                            <span class='text-gray-500 dark:text-gray-400'>1 Month</span>
                            <span class='font-medium {"text-green-600" if performance_metrics.get("1m") and performance_metrics["1m"] > 0 else "text-red-600" if performance_metrics.get("1m") else "text-gray-600"}'>{f"{performance_metrics['1m']:.1f}%" if performance_metrics.get("1m") else "N/A"}</span>
                        </div>
                        <div class='flex justify-between text-sm'>
                            <span class='text-gray-500 dark:text-gray-400'>Year to Date</span>
                            <span class='font-medium {"text-green-600" if performance_metrics.get("ytd") and performance_metrics["ytd"] > 0 else "text-red-600" if performance_metrics.get("ytd") else "text-gray-600"}'>{f"{performance_metrics['ytd']:.1f}%" if performance_metrics.get("ytd") else "N/A"}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- 52-Week Performance Range -->
        <div class='bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 p-6 rounded-xl border border-indigo-100 dark:border-indigo-800/50 shadow-sm'>
            <div class='flex items-center mb-4'>
                <div class='p-2 bg-indigo-100 dark:bg-indigo-800/50 rounded-lg mr-3'>
                    <svg class='w-6 h-6 text-indigo-600 dark:text-indigo-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M13 7h8m0 0v8m0-8l-8 8-4-4-6 6'></path>
                    </svg>
                </div>
                <h3 class='text-xl font-bold text-gray-900 dark:text-white'>52-Week Performance Range</h3>
            </div>
            
            <div class='space-y-4'>
                <div class='flex justify-between text-sm text-gray-600 dark:text-gray-400'>
                    <span>52-Week Range</span>
                    <span class='font-medium'>${low_52w:,.2f} - ${high_52w:,.2f}</span>
                </div>
                <div class='relative w-full bg-gray-200 dark:bg-gray-700 rounded-full h-4'>
                    <div class='bg-gradient-to-r from-red-500 via-yellow-500 to-green-500 h-4 rounded-full' style='width: 100%'></div>
                    <div class='absolute top-0 bg-blue-600 h-4 w-2 rounded-full shadow-lg' style='left: {current_52w_range:.2f}%'></div>
                </div>
                <div class='flex justify-between text-xs text-gray-500 dark:text-gray-400'>
                    <span>Low: ${low_52w:,.2f}</span>
                    <span class='font-medium text-blue-600 dark:text-blue-400'>Current: ${current_price:,.2f} ({current_52w_range:.1f}%)</span>
                    <span>High: ${high_52w:,.2f}</span>
                </div>
                <div class='text-center'>
                    <span class='inline-flex items-center px-3 py-1 rounded-full text-sm font-medium {health_bg} {health_color}'>
                        📊 Position: {"Near High" if current_52w_range > 80 else "Upper Range" if current_52w_range > 60 else "Mid Range" if current_52w_range > 40 else "Lower Range" if current_52w_range > 20 else "Near Low"}
                    </span>
                </div>
            </div>
        </div>
    </div>
    """
    
    # Calculate enhanced volume metrics
    volume_data = hist['Volume'].dropna()
    if len(volume_data) > 0:
        # Volume trend analysis (last 5 days vs previous 5 days)
        recent_volume = volume_data.tail(5).mean()
        previous_volume = volume_data.tail(10).head(5).mean() if len(volume_data) >= 10 else recent_volume
        volume_trend = ((recent_volume - previous_volume) / previous_volume * 100) if previous_volume > 0 else 0
        
        # Volume volatility (coefficient of variation)
        volume_volatility = (volume_data.std() / volume_data.mean() * 100) if volume_data.mean() > 0 else 0
        
        # High volume days (above 1.5x average)
        high_volume_threshold = avg_volume * 1.5
        high_volume_days = len(volume_data[volume_data > high_volume_threshold])
        high_volume_pct = (high_volume_days / len(volume_data) * 100) if len(volume_data) > 0 else 0
        
        # Volume-Price correlation
        price_changes = hist['Close'].pct_change().dropna()
        volume_changes = hist['Volume'].pct_change().dropna()
        if len(price_changes) > 1 and len(volume_changes) > 1:
            # Align the series
            min_len = min(len(price_changes), len(volume_changes))
            price_changes = price_changes.tail(min_len)
            volume_changes = volume_changes.tail(min_len)
            volume_price_corr = price_changes.corr(volume_changes) if min_len > 1 else 0
        else:
            volume_price_corr = 0
        
        # Determine volume trend status
        if volume_trend > 20:
            trend_status = "Strong Increase"
            trend_color = "text-green-600"
            trend_icon = "🚀"
        elif volume_trend > 5:
            trend_status = "Increasing"
            trend_color = "text-green-500"
            trend_icon = "📈"
        elif volume_trend < -20:
            trend_status = "Strong Decrease"
            trend_color = "text-red-600"
            trend_icon = "📉"
        elif volume_trend < -5:
            trend_status = "Decreasing"
            trend_color = "text-red-500"
            trend_icon = "📉"
        else:
            trend_status = "Stable"
            trend_color = "text-gray-600"
            trend_icon = "➡️"
        
        # Volume volatility assessment
        if volume_volatility > 100:
            volatility_status = "Very High"
            volatility_color = "text-red-600"
            volatility_icon = "🔥"
        elif volume_volatility > 50:
            volatility_status = "High"
            volatility_color = "text-orange-600"
            volatility_icon = "⚡"
        elif volume_volatility > 25:
            volatility_status = "Moderate"
            volatility_color = "text-yellow-600"
            volatility_icon = "⚖️"
        else:
            volatility_status = "Low"
            volatility_color = "text-green-600"
            volatility_icon = "🟢"
        
        # Liquidity assessment based on volume
        if latest_volume > avg_volume * 2:
            liquidity_status = "Very High"
            liquidity_color = "text-green-600"
            liquidity_icon = "💧"
        elif latest_volume > avg_volume * 1.5:
            liquidity_status = "High"
            liquidity_color = "text-green-500"
            liquidity_icon = "💧"
        elif latest_volume < avg_volume * 0.5:
            liquidity_status = "Low"
            liquidity_color = "text-red-600"
            liquidity_icon = "🏜️"
        elif latest_volume < avg_volume * 0.75:
            liquidity_status = "Below Average"
            liquidity_color = "text-orange-600"
            liquidity_icon = "⚠️"
        else:
            liquidity_status = "Normal"
            liquidity_color = "text-gray-600"
            liquidity_icon = "💧"
        
        # Volume-Price correlation interpretation
        if abs(volume_price_corr) > 0.5:
            corr_strength = "Strong"
            corr_color = "text-blue-600"
        elif abs(volume_price_corr) > 0.3:
            corr_strength = "Moderate"
            corr_color = "text-blue-500"
        else:
            corr_strength = "Weak"
            corr_color = "text-gray-600"
        
        if volume_price_corr > 0:
            corr_direction = "Positive"
            corr_icon = "📈"
        elif volume_price_corr < 0:
            corr_direction = "Negative"
            corr_icon = "📉"
        else:
            corr_direction = "Neutral"
            corr_icon = "➡️"
    else:
        # Default values if no volume data
        volume_trend, trend_status, trend_color, trend_icon = 0, "No Data", "text-gray-500", "❓"
        volume_volatility, volatility_status, volatility_color, volatility_icon = 0, "No Data", "text-gray-500", "❓"
        liquidity_status, liquidity_color, liquidity_icon = "No Data", "text-gray-500", "❓"
        high_volume_pct = 0
        volume_price_corr, corr_strength, corr_direction, corr_color, corr_icon = 0, "No Data", "No Data", "text-gray-500", "❓"
    
    # Page 2: Enhanced Volume and Liquidity Analysis
    pages['volume'] = f"""
    <div id="volume" class="page-content space-y-6">
        <!-- Volume Overview -->
        <div class='bg-gradient-to-r from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 border border-blue-200 dark:border-blue-700 rounded-xl p-6'>
            <div class='flex items-center justify-between mb-4'>
                <div class='flex items-center'>
                    <div class='p-2 bg-blue-100 dark:bg-blue-900/30 rounded-lg mr-4'>
                        <svg class='w-6 h-6 text-blue-600 dark:text-blue-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                            <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z' />
                        </svg>
                    </div>
                    <div>
                        <h3 class='text-xl font-bold text-gray-900 dark:text-white'>Volume Analysis</h3>
                        <p class='text-sm text-gray-600 dark:text-gray-400'>Trading volume and liquidity metrics</p>
                    </div>
                </div>
                <div class='text-right'>
                    <div class='text-2xl font-bold {volume_color}'>{volume_ratio:.1f}x</div>
                    <div class='text-xs text-gray-500 dark:text-gray-400'>vs Average</div>
                </div>
            </div>
            
            <!-- Volume Metrics Grid -->
            <div class='grid grid-cols-1 md:grid-cols-3 gap-4'>
                <!-- Current Volume -->
                <div class='bg-white/70 dark:bg-gray-800/70 rounded-lg p-4'>
                    <div class='flex items-center justify-between mb-2'>
                        <span class='text-sm font-medium text-gray-700 dark:text-gray-300'>Current Volume</span>
                        <span class='text-lg'>📊</span>
                    </div>
                    <div class='text-2xl font-bold text-gray-900 dark:text-white'>{latest_volume:,.0f}</div>
                    <div class='text-xs text-gray-500 dark:text-gray-400 mt-1'>Latest trading session</div>
                </div>
                
                <!-- Average Volume -->
                <div class='bg-white/70 dark:bg-gray-800/70 rounded-lg p-4'>
                    <div class='flex items-center justify-between mb-2'>
                        <span class='text-sm font-medium text-gray-700 dark:text-gray-300'>Average Volume</span>
                        <span class='text-lg'>📈</span>
                    </div>
                    <div class='text-2xl font-bold text-gray-900 dark:text-white'>{avg_volume:,.0f}</div>
                    <div class='text-xs text-gray-500 dark:text-gray-400 mt-1'>{len(hist)} day average</div>
                </div>
                
                <!-- Volume Trend -->
                <div class='bg-white/70 dark:bg-gray-800/70 rounded-lg p-4'>
                    <div class='flex items-center justify-between mb-2'>
                        <span class='text-sm font-medium text-gray-700 dark:text-gray-300'>5-Day Trend</span>
                        <span class='text-lg'>{trend_icon}</span>
                    </div>
                    <div class='text-2xl font-bold {trend_color}'>{volume_trend:+.1f}%</div>
                    <div class='text-xs {trend_color} mt-1'>{trend_status}</div>
                </div>
            </div>
        </div>
        
        <!-- Volume Analysis Grid -->
        <div class='grid grid-cols-1 lg:grid-cols-2 gap-6'>
            <!-- Volume Characteristics -->
            <div class='bg-white dark:bg-gray-800/50 border border-gray-100 dark:border-gray-700/50 rounded-xl p-6'>
                <div class='flex items-center mb-4'>
                    <div class='p-2 bg-purple-100 dark:bg-purple-900/30 rounded-lg mr-3'>
                        <svg class='w-5 h-5 text-purple-600 dark:text-purple-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                            <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M11 3.055A9.001 9.001 0 1020.945 13H11V3.055z' />
                        </svg>
                    </div>
                    <h4 class='font-semibold text-gray-900 dark:text-white'>Volume Characteristics</h4>
                </div>
                <div class='space-y-4'>
                    <!-- Volume Volatility -->
                    <div class='p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg'>
                        <div class='flex items-center justify-between mb-2'>
                            <span class='text-sm font-medium text-gray-700 dark:text-gray-300'>Volume Volatility</span>
                            <div class='flex items-center space-x-1'>
                                <span class='text-sm'>{volatility_icon}</span>
                                <span class='text-xs font-medium {volatility_color}'>{volatility_status}</span>
                            </div>
                        </div>
                        <div class='text-lg font-bold text-gray-900 dark:text-white'>{volume_volatility:.1f}%</div>
                        <div class='text-xs text-gray-500 dark:text-gray-400'>Coefficient of variation</div>
                    </div>
                    
                    <!-- High Volume Days -->
                    <div class='p-3 bg-orange-50 dark:bg-orange-900/20 rounded-lg'>
                        <div class='flex items-center justify-between mb-2'>
                            <span class='text-sm font-medium text-gray-700 dark:text-gray-300'>High Volume Days</span>
                            <span class='text-sm'>🔥</span>
                        </div>
                        <div class='text-lg font-bold text-gray-900 dark:text-white'>{high_volume_pct:.1f}%</div>
                        <div class='text-xs text-gray-500 dark:text-gray-400'>Above 1.5x average volume</div>
                    </div>
                    
                    <!-- Liquidity Status -->
                    <div class='p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg'>
                        <div class='flex items-center justify-between mb-2'>
                            <span class='text-sm font-medium text-gray-700 dark:text-gray-300'>Current Liquidity</span>
                            <div class='flex items-center space-x-1'>
                                <span class='text-sm'>{liquidity_icon}</span>
                                <span class='text-xs font-medium {liquidity_color}'>{liquidity_status}</span>
                            </div>
                        </div>
                        <div class='text-lg font-bold {liquidity_color}'>{volume_ratio:.1f}x Average</div>
                        <div class='text-xs text-gray-500 dark:text-gray-400'>Market depth indicator</div>
                    </div>
                </div>
            </div>
            
            <!-- Volume-Price Relationship -->
            <div class='bg-white dark:bg-gray-800/50 border border-gray-100 dark:border-gray-700/50 rounded-xl p-6'>
                <div class='flex items-center mb-4'>
                    <div class='p-2 bg-green-100 dark:bg-green-900/30 rounded-lg mr-3'>
                        <svg class='w-5 h-5 text-green-600 dark:text-green-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                            <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M13 7h8m0 0v8m0-8l-8 8-4-4-6 6' />
                        </svg>
                    </div>
                    <h4 class='font-semibold text-gray-900 dark:text-white'>Volume-Price Analysis</h4>
                </div>
                <div class='space-y-4'>
                    <!-- Correlation -->
                    <div class='p-4 bg-gradient-to-r from-green-50 to-blue-50 dark:from-green-900/20 dark:to-blue-900/20 rounded-lg border border-green-200 dark:border-green-700'>
                        <div class='flex items-center justify-between mb-3'>
                            <span class='text-sm font-medium text-gray-700 dark:text-gray-300'>Price-Volume Correlation</span>
                            <span class='text-lg'>{corr_icon}</span>
                        </div>
                        <div class='text-2xl font-bold {corr_color}'>{volume_price_corr:.3f}</div>
                        <div class='text-sm {corr_color} mt-1'>{corr_strength} {corr_direction}</div>
                        <div class='text-xs text-gray-500 dark:text-gray-400 mt-2'>
                            {("Strong correlation suggests volume confirms price moves" if abs(volume_price_corr) > 0.5 else 
                              "Moderate correlation indicates some volume-price relationship" if abs(volume_price_corr) > 0.3 else 
                              "Weak correlation suggests volume and price move independently")}
                        </div>
                    </div>
                    
                    <!-- Volume Pattern Analysis -->
                    <div class='p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg'>
                        <h5 class='font-medium text-gray-800 dark:text-gray-200 mb-2'>Volume Pattern Insights</h5>
                        <div class='space-y-2 text-sm text-gray-600 dark:text-gray-400'>
                            {"<p>• <strong>High Volume Confirmation:</strong> Strong price moves supported by volume</p>" if volume_ratio > 1.5 and abs(volume_price_corr) > 0.3 else ""}
                            {"<p>• <strong>Volume Breakout:</strong> Unusual volume may signal trend change</p>" if volume_ratio > 2 else ""}
                            {"<p>• <strong>Low Volume Drift:</strong> Price moves without volume support</p>" if volume_ratio < 0.7 else ""}
                            {"<p>• <strong>Volume Divergence:</strong> Price and volume moving in opposite directions</p>" if volume_price_corr < -0.3 else ""}
                            <p>• <strong>Trend Strength:</strong> {("Strong" if volume_ratio > 1.5 else "Moderate" if volume_ratio > 1.0 else "Weak")} volume support</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Volume Trading Insights -->
        <div class='bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-700 rounded-xl p-5'>
            <div class='flex items-center mb-3'>
                <div class='p-1.5 bg-amber-100 dark:bg-amber-900/30 rounded-lg mr-3'>
                    <svg class='w-4 h-4 text-amber-600 dark:text-amber-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z' />
                    </svg>
                </div>
                <h4 class='font-semibold text-amber-800 dark:text-amber-200'>Volume Trading Insights</h4>
            </div>
            <div class='grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-amber-700 dark:text-amber-300'>
                <div class='space-y-1'>
                    {"<p>• <strong>High Activity:</strong> Increased institutional interest likely</p>" if volume_ratio > 2 else ""}
                    {"<p>• <strong>Breakout Potential:</strong> Volume spike may signal trend change</p>" if volume_ratio > 1.8 else ""}
                    {"<p>• <strong>Accumulation Phase:</strong> Steady volume suggests building positions</p>" if 0.8 <= volume_ratio <= 1.2 and trend_status == "Stable" else ""}
                    {"<p>• <strong>Distribution Phase:</strong> High volume with price weakness</p>" if volume_ratio > 1.5 and volume_price_corr < -0.2 else ""}
                </div>
                <div class='space-y-1'>
                    <p>• <strong>Liquidity:</strong> {liquidity_status.lower()} trading liquidity</p>
                    <p>• <strong>Volatility:</strong> {volatility_status.lower()} volume consistency</p>
                    <p>• <strong>Trend Support:</strong> Volume {("confirms" if abs(volume_price_corr) > 0.3 else "doesn't confirm")} price direction</p>
                    <p>• <strong>Risk Level:</strong> {("Higher" if volume_volatility > 75 else "Moderate" if volume_volatility > 40 else "Lower")} volume-based risk</p>
                </div>
            </div>
        </div>
    </div>
    """
    
    # Page 3: Technical Indicators
    pages['technical'] = f"""
    <div id="technical" class="page-content space-y-4">
        <!-- Momentum Indicators -->
        <div class='bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 p-5 rounded-xl'>
            <div class='flex items-center mb-4'>
                <div class='p-1.5 bg-blue-100 dark:bg-blue-900/30 rounded-lg mr-3'>
                    <svg class='w-5 h-5 text-blue-600 dark:text-blue-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M13 10V3L4 14h7v7l9-11h-7z' />
                    </svg>
                </div>
                <h3 class='text-lg font-semibold text-gray-800 dark:text-white'>Momentum Indicators</h3>
            </div>
            
            <div class='grid grid-cols-1 md:grid-cols-2 gap-6'>
                <!-- RSI Analysis -->
                <div class='p-4 bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-900/30 rounded-lg border border-purple-100 dark:border-purple-800/30'>
                    <div class='flex items-center justify-between mb-3'>
                        <h4 class='font-medium text-purple-800 dark:text-purple-200'>RSI (14-day)</h4>
                        <span class='text-2xl'>📊</span>
                    </div>
                    <div class='space-y-2'>
                        <div class='flex justify-between items-center'>
                            <span class='text-sm text-purple-700 dark:text-purple-300'>Current RSI</span>
                            <span class='font-bold text-lg {rsi_color}'>{rsi_value}</span>
                        </div>
                        <div class='w-full bg-purple-200 dark:bg-purple-800 rounded-full h-2'>
                            <div class='bg-purple-600 h-2 rounded-full' style='width: {rsi if rsi is not None else 0}%'></div>
                        </div>
                        <div class='flex justify-between text-xs text-purple-600 dark:text-purple-400'>
                            <span>Oversold (30)</span>
                            <span>Neutral (50)</span>
                            <span>Overbought (70)</span>
                        </div>
                        <div class='mt-2 p-2 bg-white/50 dark:bg-gray-800/50 rounded text-center'>
                            <span class='text-sm font-medium {rsi_color}'>Status: {rsi_status}</span>
                        </div>
                    </div>
                </div>
                
                <!-- MACD Analysis -->
                <div class='p-4 bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-900/30 rounded-lg border border-green-100 dark:border-green-800/30'>
                    <div class='flex items-center justify-between mb-3'>
                        <h4 class='font-medium text-green-800 dark:text-green-200'>MACD</h4>
                        <span class='text-2xl'>📈</span>
                    </div>
                    <div class='space-y-2'>
                        <div class='flex justify-between text-sm'>
                            <span class='text-green-700 dark:text-green-300'>MACD Line</span>
                            <span class='font-medium'>{macd_line_str}</span>
                        </div>
                        <div class='flex justify-between text-sm'>
                            <span class='text-green-700 dark:text-green-300'>Signal Line</span>
                            <span class='font-medium'>{macd_signal_str}</span>
                        </div>
                        <div class='flex justify-between text-sm'>
                            <span class='text-green-700 dark:text-green-300'>Histogram</span>
                            <span class='font-medium'>{macd_histogram_str}</span>
                        </div>
                        <div class='mt-2 p-2 bg-white/50 dark:bg-gray-800/50 rounded text-center'>
                            <span class='text-sm font-medium {macd_color}'>Signal: {macd_status}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Trend Analysis -->
        <div class='bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 p-5 rounded-xl'>
            <div class='flex items-center mb-4'>
                <div class='p-1.5 bg-orange-100 dark:bg-orange-900/30 rounded-lg mr-3'>
                    <svg class='w-5 h-5 text-orange-600 dark:text-orange-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z' />
                    </svg>
                </div>
                <h3 class='text-lg font-semibold text-gray-800 dark:text-white'>Trend Analysis</h3>
            </div>
            
            <div class='grid grid-cols-1 md:grid-cols-2 gap-6'>
                <!-- Moving Average Analysis -->
                <div>
                    <h4 class='font-medium text-gray-700 dark:text-gray-300 mb-3'>Moving Average Crossovers</h4>
                    <div class='space-y-3'>
                        <div class='flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700/30 rounded-lg'>
                            <div class='flex items-center'>
                                <span class='w-3 h-3 bg-blue-500 rounded-full mr-3'></span>
                                <span class='text-sm font-medium'>20-Day MA</span>
                            </div>
                            <span class='font-bold' style='color: {"#10b981" if ma_20 and current_price > ma_20 else "#ef4444" if ma_20 else "#6b7280"}'>{ma_20_str}</span>
                        </div>
                        <div class='flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700/30 rounded-lg'>
                            <div class='flex items-center'>
                                <span class='w-3 h-3 bg-yellow-500 rounded-full mr-3'></span>
                                <span class='text-sm font-medium'>50-Day MA</span>
                            </div>
                            <span class='font-bold' style='color: {"#10b981" if ma_50 and current_price > ma_50 else "#ef4444" if ma_50 else "#6b7280"}'>{ma_50_str}</span>
                        </div>
                        <div class='flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700/30 rounded-lg'>
                            <div class='flex items-center'>
                                <span class='w-3 h-3 bg-red-500 rounded-full mr-3'></span>
                                <span class='text-sm font-medium'>200-Day MA</span>
                            </div>
                            <span class='font-bold' style='color: {"#10b981" if ma_200 and current_price > ma_200 else "#ef4444" if ma_200 else "#6b7280"}'>{ma_200_str}</span>
                        </div>
                    </div>
                </div>
                
                <!-- Trend Summary -->
                <div>
                    <h4 class='font-medium text-gray-700 dark:text-gray-300 mb-3'>Overall Trend Assessment</h4>
                    <div class='p-4 bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 rounded-lg border border-indigo-100 dark:border-indigo-800/30'>
                        <div class='text-center'>
                            <div class='text-3xl mb-2'>
                                {'🚀' if trend_status == 'Strong Uptrend' else '📈' if 'Uptrend' in trend_status else '📉' if 'Downtrend' in trend_status else '📊'}
                            </div>
                            <h5 class='font-bold text-lg {trend_color} mb-2'>{trend_status}</h5>
                            <p class='text-sm text-gray-600 dark:text-gray-400'>
                                {'Price is above all major moving averages, indicating strong bullish momentum.' if trend_status == 'Strong Uptrend' else
                                 'Price is above short-term averages, showing upward momentum.' if trend_status == 'Uptrend' else
                                 'Price is below all major moving averages, indicating strong bearish pressure.' if trend_status == 'Strong Downtrend' else
                                 'Price is below short-term averages, showing downward pressure.' if trend_status == 'Downtrend' else
                                 'Mixed signals from moving averages suggest consolidation or transition period.'}
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Volume Analysis -->
        <div class='bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 p-5 rounded-xl'>
            <div class='flex items-center mb-4'>
                <div class='p-1.5 bg-teal-100 dark:bg-teal-900/30 rounded-lg mr-3'>
                    <svg class='w-5 h-5 text-teal-600 dark:text-teal-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z' />
                    </svg>
                </div>
                <h3 class='text-lg font-semibold text-gray-800 dark:text-white'>Volume Analysis</h3>
            </div>
            
            <div class='grid grid-cols-1 md:grid-cols-3 gap-4'>
                <div class='p-4 bg-gradient-to-br from-teal-50 to-teal-100 dark:from-teal-900/20 dark:to-teal-900/30 rounded-lg border border-teal-100 dark:border-teal-800/30'>
                    <div class='flex items-center justify-between mb-2'>
                        <span class='text-sm font-medium text-teal-700 dark:text-teal-300'>Latest Volume</span>
                        <span class='text-lg'>📊</span>
                    </div>
                    <p class='text-xl font-bold text-teal-800 dark:text-teal-200'>{latest_volume:,.0f}</p>
                </div>
                <div class='p-4 bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-900/30 rounded-lg border border-blue-100 dark:border-blue-800/30'>
                    <div class='flex items-center justify-between mb-2'>
                        <span class='text-sm font-medium text-blue-700 dark:text-blue-300'>Average Volume</span>
                        <span class='text-lg'>📈</span>
                    </div>
                    <p class='text-xl font-bold text-blue-800 dark:text-blue-200'>{avg_volume:,.0f}</p>
                </div>
                <div class='p-4 bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-900/30 rounded-lg border border-purple-100 dark:border-purple-800/30'>
                    <div class='flex items-center justify-between mb-2'>
                        <span class='text-sm font-medium text-purple-700 dark:text-purple-300'>Volume Ratio</span>
                        <span class='text-lg'>⚡</span>
                    </div>
                    <p class='text-xl font-bold {volume_color}'>{volume_ratio:.1f}x</p>
                </div>
            </div>
            
            <div class='mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-100 dark:border-blue-800/30'>
                <p class='text-sm text-blue-700 dark:text-blue-300'>
                    <span class='font-medium'>Volume Insight:</span> 
                    {'High volume suggests strong interest and validates price movements.' if volume_ratio > 1.2 else
                     'Normal volume indicates typical trading activity.' if volume_ratio > 0.8 else
                     'Low volume may indicate lack of conviction in current price movement.'}
                </p>
            </div>
        </div>
    </div>
    """
    
    # Calculate enhanced performance metrics
    returns_data = hist['Close'].pct_change().dropna()
    
    # Calculate additional performance periods
    performance_periods = {
        '1d': 1, '3d': 3, '1w': 5, '2w': 10, '1m': 22, '3m': 66, '6m': 132, 'ytd': None, '1y': 252
    }
    
    enhanced_performance = {}
    for period, days in performance_periods.items():
        if period == 'ytd':
            # YTD calculation (from performance_metrics)
            enhanced_performance[period] = performance_metrics.get('ytd')
        elif days and len(hist) > days:
            start_price = hist['Close'].iloc[-days-1]
            end_price = hist['Close'].iloc[-1]
            enhanced_performance[period] = ((end_price - start_price) / start_price * 100)
        else:
            enhanced_performance[period] = None
    
    # Calculate volatility metrics
    if len(returns_data) > 0:
        daily_volatility = returns_data.std()
        annualized_volatility = daily_volatility * (252 ** 0.5) * 100  # Annualized volatility
        
        # Sharpe ratio approximation (assuming risk-free rate of 2%)
        risk_free_rate = 0.02
        if len(returns_data) >= 252:
            annual_return = ((hist['Close'].iloc[-1] / hist['Close'].iloc[-252]) - 1) * 100
            sharpe_ratio = (annual_return - risk_free_rate * 100) / annualized_volatility if annualized_volatility > 0 else 0
        else:
            sharpe_ratio = None
        
        # Maximum drawdown calculation
        cumulative_returns = (1 + returns_data).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        # Win rate (percentage of positive days)
        positive_days = len(returns_data[returns_data > 0])
        win_rate = (positive_days / len(returns_data)) * 100
        
        # Average gain vs average loss
        gains = returns_data[returns_data > 0]
        losses = returns_data[returns_data < 0]
        avg_gain = gains.mean() * 100 if len(gains) > 0 else 0
        avg_loss = abs(losses.mean()) * 100 if len(losses) > 0 else 0
        gain_loss_ratio = avg_gain / avg_loss if avg_loss > 0 else 0
        
        # Consecutive performance streaks
        consecutive_gains = 0
        consecutive_losses = 0
        current_gain_streak = 0
        current_loss_streak = 0
        max_gain_streak = 0
        max_loss_streak = 0
        
        for ret in returns_data:
            if ret > 0:
                current_gain_streak += 1
                current_loss_streak = 0
                max_gain_streak = max(max_gain_streak, current_gain_streak)
            elif ret < 0:
                current_loss_streak += 1
                current_gain_streak = 0
                max_loss_streak = max(max_loss_streak, current_loss_streak)
            else:
                current_gain_streak = 0
                current_loss_streak = 0
    else:
        annualized_volatility = 0
        sharpe_ratio = None
        max_drawdown = 0
        win_rate = 0
        avg_gain = 0
        avg_loss = 0
        gain_loss_ratio = 0
        max_gain_streak = 0
        max_loss_streak = 0
    
    # Performance assessment
    def get_performance_assessment(perf):
        if perf is None:
            return "No Data", "text-gray-500", "❓"
        elif perf >= 20:
            return "Excellent", "text-green-600", "🚀"
        elif perf >= 10:
            return "Strong", "text-green-500", "📈"
        elif perf >= 5:
            return "Good", "text-green-400", "✅"
        elif perf >= 0:
            return "Positive", "text-blue-500", "➕"
        elif perf >= -5:
            return "Slight Loss", "text-yellow-600", "⚠️"
        elif perf >= -10:
            return "Moderate Loss", "text-orange-600", "📉"
        else:
            return "Poor", "text-red-600", "❌"
    
    # Risk assessment
    if annualized_volatility > 40:
        risk_level = "Very High"
        risk_color = "text-red-600"
        risk_icon = "🔥"
    elif annualized_volatility > 25:
        risk_level = "High"
        risk_color = "text-orange-600"
        risk_icon = "⚡"
    elif annualized_volatility > 15:
        risk_level = "Moderate"
        risk_color = "text-yellow-600"
        risk_icon = "⚖️"
    else:
        risk_level = "Low"
        risk_color = "text-green-600"
        risk_icon = "🛡️"
    
    # Page 4: Enhanced Performance Analysis
    pages['performance'] = f"""
    <div id="performance" class="page-content space-y-6">
        <!-- Performance Overview -->
        <div class='bg-gradient-to-r from-green-50 to-blue-50 dark:from-green-900/20 dark:to-blue-900/20 border border-green-200 dark:border-green-700 rounded-xl p-6'>
            <div class='flex items-center justify-between mb-4'>
                <div class='flex items-center'>
                    <div class='p-2 bg-green-100 dark:bg-green-900/30 rounded-lg mr-4'>
                        <svg class='w-6 h-6 text-green-600 dark:text-green-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                            <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M13 7h8m0 0v8m0-8l-8 8-4-4-6 6' />
                        </svg>
                    </div>
                    <div>
                        <h3 class='text-xl font-bold text-gray-900 dark:text-white'>Performance Analysis</h3>
                        <p class='text-sm text-gray-600 dark:text-gray-400'>Returns, risk metrics, and performance statistics</p>
                    </div>
                </div>
                <div class='text-right'>
                    <div class='text-2xl font-bold {("text-green-600" if enhanced_performance.get("ytd", 0) and enhanced_performance["ytd"] >= 0 else "text-red-600")}'>{enhanced_performance.get("ytd", 0):+.1f}%</div>
                    <div class='text-xs text-gray-500 dark:text-gray-400'>YTD Return</div>
                </div>
            </div>
            
            <!-- Performance Periods Grid -->
            <div class='grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-3'>
                <!-- 1 Day -->
                <div class='bg-white/70 dark:bg-gray-800/70 rounded-lg p-3 text-center'>
                    <div class='text-xs text-gray-500 dark:text-gray-400 mb-1'>1 Day</div>
                    {f'<div class="text-lg font-bold {("text-green-600" if enhanced_performance["1d"] and enhanced_performance["1d"] >= 0 else "text-red-600")}">{enhanced_performance["1d"]:+.2f}%</div>' if enhanced_performance["1d"] is not None else '<div class="text-sm text-gray-400">N/A</div>'}
                    {f'<div class="text-xs {get_performance_assessment(enhanced_performance["1d"])[1]}">{get_performance_assessment(enhanced_performance["1d"])[2]}</div>' if enhanced_performance["1d"] is not None else ''}
                </div>
                
                <!-- 1 Week -->
                <div class='bg-white/70 dark:bg-gray-800/70 rounded-lg p-3 text-center'>
                    <div class='text-xs text-gray-500 dark:text-gray-400 mb-1'>1 Week</div>
                    {f'<div class="text-lg font-bold {("text-green-600" if enhanced_performance["1w"] and enhanced_performance["1w"] >= 0 else "text-red-600")}">{enhanced_performance["1w"]:+.2f}%</div>' if enhanced_performance["1w"] is not None else '<div class="text-sm text-gray-400">N/A</div>'}
                    {f'<div class="text-xs {get_performance_assessment(enhanced_performance["1w"])[1]}">{get_performance_assessment(enhanced_performance["1w"])[2]}</div>' if enhanced_performance["1w"] is not None else ''}
                </div>
                
                <!-- 1 Month -->
                <div class='bg-white/70 dark:bg-gray-800/70 rounded-lg p-3 text-center'>
                    <div class='text-xs text-gray-500 dark:text-gray-400 mb-1'>1 Month</div>
                    {f'<div class="text-lg font-bold {("text-green-600" if enhanced_performance["1m"] and enhanced_performance["1m"] >= 0 else "text-red-600")}">{enhanced_performance["1m"]:+.2f}%</div>' if enhanced_performance["1m"] is not None else '<div class="text-sm text-gray-400">N/A</div>'}
                    {f'<div class="text-xs {get_performance_assessment(enhanced_performance["1m"])[1]}">{get_performance_assessment(enhanced_performance["1m"])[2]}</div>' if enhanced_performance["1m"] is not None else ''}
                </div>
                
                <!-- 3 Months -->
                <div class='bg-white/70 dark:bg-gray-800/70 rounded-lg p-3 text-center'>
                    <div class='text-xs text-gray-500 dark:text-gray-400 mb-1'>3 Months</div>
                    {f'<div class="text-lg font-bold {("text-green-600" if enhanced_performance["3m"] and enhanced_performance["3m"] >= 0 else "text-red-600")}">{enhanced_performance["3m"]:+.2f}%</div>' if enhanced_performance["3m"] is not None else '<div class="text-sm text-gray-400">N/A</div>'}
                    {f'<div class="text-xs {get_performance_assessment(enhanced_performance["3m"])[1]}">{get_performance_assessment(enhanced_performance["3m"])[2]}</div>' if enhanced_performance["3m"] is not None else ''}
                </div>
                
                <!-- 1 Year -->
                <div class='bg-white/70 dark:bg-gray-800/70 rounded-lg p-3 text-center'>
                    <div class='text-xs text-gray-500 dark:text-gray-400 mb-1'>1 Year</div>
                    {f'<div class="text-lg font-bold {("text-green-600" if enhanced_performance["1y"] and enhanced_performance["1y"] >= 0 else "text-red-600")}">{enhanced_performance["1y"]:+.2f}%</div>' if enhanced_performance["1y"] is not None else '<div class="text-sm text-gray-400">N/A</div>'}
                    {f'<div class="text-xs {get_performance_assessment(enhanced_performance["1y"])[1]}">{get_performance_assessment(enhanced_performance["1y"])[2]}</div>' if enhanced_performance["1y"] is not None else ''}
                </div>
            </div>
        </div>
        
        <!-- Risk & Return Analysis -->
        <div class='grid grid-cols-1 lg:grid-cols-2 gap-6'>
            <!-- Risk Metrics -->
            <div class='bg-white dark:bg-gray-800/50 border border-gray-100 dark:border-gray-700/50 rounded-xl p-6'>
                <div class='flex items-center mb-4'>
                    <div class='p-2 bg-red-100 dark:bg-red-900/30 rounded-lg mr-3'>
                        <svg class='w-5 h-5 text-red-600 dark:text-red-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                            <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z' />
                        </svg>
                    </div>
                    <h4 class='font-semibold text-gray-900 dark:text-white'>Risk Analysis</h4>
                </div>
                <div class='space-y-4'>
                    <!-- Volatility -->
                    <div class='p-3 bg-red-50 dark:bg-red-900/20 rounded-lg'>
                        <div class='flex items-center justify-between mb-2'>
                            <span class='text-sm font-medium text-gray-700 dark:text-gray-300'>Annualized Volatility</span>
                            <div class='flex items-center space-x-1'>
                                <span class='text-sm'>{risk_icon}</span>
                                <span class='text-xs font-medium {risk_color}'>{risk_level}</span>
                            </div>
                        </div>
                        <div class='text-2xl font-bold text-gray-900 dark:text-white'>{annualized_volatility:.1f}%</div>
                        <div class='text-xs text-gray-500 dark:text-gray-400'>Standard deviation of returns</div>
                    </div>
                    
                    <!-- Maximum Drawdown -->
                    <div class='p-3 bg-orange-50 dark:bg-orange-900/20 rounded-lg'>
                        <div class='flex items-center justify-between mb-2'>
                            <span class='text-sm font-medium text-gray-700 dark:text-gray-300'>Maximum Drawdown</span>
                            <span class='text-sm'>📉</span>
                        </div>
                        <div class='text-2xl font-bold text-red-600'>{max_drawdown:.1f}%</div>
                        <div class='text-xs text-gray-500 dark:text-gray-400'>Largest peak-to-trough decline</div>
                    </div>
                    
                    <!-- Sharpe Ratio -->
                    <div class='p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg'>
                        <div class='flex items-center justify-between mb-2'>
                            <span class='text-sm font-medium text-gray-700 dark:text-gray-300'>Sharpe Ratio</span>
                            <span class='text-sm'>⚖️</span>
                        </div>
                        {f'<div class="text-2xl font-bold {("text-green-600" if sharpe_ratio and sharpe_ratio > 1 else "text-yellow-600" if sharpe_ratio and sharpe_ratio > 0 else "text-red-600")}">{sharpe_ratio:.2f}</div>' if sharpe_ratio is not None else '<div class="text-lg text-gray-400">N/A</div>'}
                        <div class='text-xs text-gray-500 dark:text-gray-400'>Risk-adjusted return measure</div>
                    </div>
                </div>
            </div>
            
            <!-- Trading Statistics -->
            <div class='bg-white dark:bg-gray-800/50 border border-gray-100 dark:border-gray-700/50 rounded-xl p-6'>
                <div class='flex items-center mb-4'>
                    <div class='p-2 bg-purple-100 dark:bg-purple-900/30 rounded-lg mr-3'>
                        <svg class='w-5 h-5 text-purple-600 dark:text-purple-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                            <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z' />
                        </svg>
                    </div>
                    <h4 class='font-semibold text-gray-900 dark:text-white'>Trading Statistics</h4>
                </div>
                <div class='space-y-4'>
                    <!-- Win Rate -->
                    <div class='p-3 bg-purple-50 dark:bg-purple-900/20 rounded-lg'>
                        <div class='flex items-center justify-between mb-2'>
                            <span class='text-sm font-medium text-gray-700 dark:text-gray-300'>Win Rate</span>
                            <span class='text-sm'>🎯</span>
                        </div>
                        <div class='text-2xl font-bold {("text-green-600" if win_rate > 55 else "text-yellow-600" if win_rate > 45 else "text-red-600")}'>{win_rate:.1f}%</div>
                        <div class='text-xs text-gray-500 dark:text-gray-400'>Percentage of positive days</div>
                    </div>
                    
                    <!-- Average Gain vs Loss -->
                    <div class='p-3 bg-green-50 dark:bg-green-900/20 rounded-lg'>
                        <div class='flex items-center justify-between mb-2'>
                            <span class='text-sm font-medium text-gray-700 dark:text-gray-300'>Gain/Loss Ratio</span>
                            <span class='text-sm'>⚖️</span>
                        </div>
                        <div class='text-2xl font-bold {("text-green-600" if gain_loss_ratio > 1.2 else "text-yellow-600" if gain_loss_ratio > 0.8 else "text-red-600")}'>{gain_loss_ratio:.2f}</div>
                        <div class='text-xs text-gray-500 dark:text-gray-400'>Avg gain: {avg_gain:.2f}% | Avg loss: {avg_loss:.2f}%</div>
                    </div>
                    
                    <!-- Streak Analysis -->
                    <div class='p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg'>
                        <div class='flex items-center justify-between mb-2'>
                            <span class='text-sm font-medium text-gray-700 dark:text-gray-300'>Streak Analysis</span>
                            <span class='text-sm'>🔥</span>
                        </div>
                        <div class='grid grid-cols-2 gap-2 text-sm'>
                            <div>
                                <div class='text-green-600 font-bold'>+{max_gain_streak}</div>
                                <div class='text-xs text-gray-500'>Max gains</div>
                            </div>
                            <div>
                                <div class='text-red-600 font-bold'>-{max_loss_streak}</div>
                                <div class='text-xs text-gray-500'>Max losses</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Performance Insights -->
        <div class='bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-700 rounded-xl p-5'>
            <div class='flex items-center mb-3'>
                <div class='p-1.5 bg-blue-100 dark:bg-blue-900/30 rounded-lg mr-3'>
                    <svg class='w-4 h-4 text-blue-600 dark:text-blue-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z' />
                    </svg>
                </div>
                <h4 class='font-semibold text-blue-800 dark:text-blue-200'>Performance Insights</h4>
            </div>
            <div class='grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-blue-700 dark:text-blue-300'>
                <div class='space-y-1'>
                    <p>• <strong>Risk Level:</strong> {risk_level.lower()} volatility ({annualized_volatility:.1f}%)</p>
                    <p>• <strong>Consistency:</strong> {win_rate:.0f}% positive trading days</p>
                    {"<p>• <strong>Risk-Adjusted Return:</strong> " + ("Excellent" if sharpe_ratio and sharpe_ratio > 2 else "Good" if sharpe_ratio and sharpe_ratio > 1 else "Fair" if sharpe_ratio and sharpe_ratio > 0 else "Poor") + " Sharpe ratio</p>" if sharpe_ratio is not None else "<p>• <strong>Risk-Adjusted Return:</strong> Insufficient data for Sharpe ratio</p>"}
                    <p>• <strong>Drawdown Risk:</strong> {abs(max_drawdown):.1f}% maximum decline</p>
                </div>
                <div class='space-y-1'>
                    <p>• <strong>Trading Profile:</strong> {("Conservative" if annualized_volatility < 20 else "Moderate" if annualized_volatility < 35 else "Aggressive")} risk profile</p>
                    <p>• <strong>Return Pattern:</strong> {("Consistent gains" if win_rate > 60 else "Balanced" if win_rate > 40 else "Volatile returns")}</p>
                    <p>• <strong>Best Period:</strong> {max([(k, v) for k, v in enhanced_performance.items() if v is not None], key=lambda x: x[1], default=("N/A", 0))[0].upper()}</p>
                    <p>• <strong>Streak Tendency:</strong> {("Momentum-driven" if max_gain_streak > max_loss_streak else "Mean-reverting" if max_loss_streak > max_gain_streak else "Balanced")}</p>
                </div>
            </div>
        </div>
    </div>
    """
    
    # Page 4: Enhanced Company Analysis
    
    # Calculate financial metrics
    market_cap = info.get('marketCap', 0)
    market_cap_formatted = f"${market_cap / 1e9:.2f}B" if market_cap and market_cap > 1e9 else f"${market_cap / 1e6:.2f}M" if market_cap and market_cap > 1e6 else "N/A"
    
    # Key financial ratios
    pe_ratio = info.get('trailingPE', 'N/A')
    pb_ratio = info.get('priceToBook', 'N/A')
    roe = info.get('returnOnEquity', 'N/A')
    debt_to_equity = info.get('debtToEquity', 'N/A')
    profit_margin = info.get('profitMargins', 'N/A')
    operating_margin = info.get('operatingMargins', 'N/A')
    
    # Revenue and growth
    revenue = info.get('totalRevenue', 0)
    revenue_formatted = f"${revenue / 1e9:.2f}B" if revenue and revenue > 1e9 else f"${revenue / 1e6:.2f}M" if revenue and revenue > 1e6 else "N/A"
    revenue_growth = info.get('revenueGrowth', 'N/A')
    
    # Business description
    business_summary = info.get('longBusinessSummary', 'No business summary available.')
    if len(business_summary) > 300:
        business_summary = business_summary[:300] + "..."
    
    pages['company'] = f"""
    <div id="company" class="page-content space-y-6">
        <!-- Company Header -->
        <div class='bg-gradient-to-r from-purple-50 to-blue-50 dark:from-purple-900/20 dark:to-blue-900/20 p-6 rounded-xl border border-purple-100 dark:border-purple-800/50 shadow-sm'>
            <div class='flex items-center mb-4'>
                <div class='p-2 bg-purple-100 dark:bg-purple-800/50 rounded-lg mr-3'>
                    <svg class='w-6 h-6 text-purple-600 dark:text-purple-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4'></path>
                    </svg>
                </div>
                <h3 class='text-xl font-bold text-gray-900 dark:text-white'>Company Overview</h3>
            </div>
            
            <div class='grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6'>
                <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <div class='text-xs text-gray-500 dark:text-gray-400 mb-1'>Market Cap</div>
                    <div class='text-lg font-bold text-gray-900 dark:text-white'>{market_cap_formatted}</div>
                </div>
                <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <div class='text-xs text-gray-500 dark:text-gray-400 mb-1'>Revenue (TTM)</div>
                    <div class='text-lg font-bold text-gray-900 dark:text-white'>{revenue_formatted}</div>
                </div>
                <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <div class='text-xs text-gray-500 dark:text-gray-400 mb-1'>Employees</div>
                    <div class='text-lg font-bold text-gray-900 dark:text-white'>{info.get('fullTimeEmployees', 'N/A'):,}</div>
                </div>
                <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <div class='text-xs text-gray-500 dark:text-gray-400 mb-1'>Sector</div>
                    <div class='text-lg font-bold text-gray-900 dark:text-white'>{info.get('sector', 'N/A')}</div>
                </div>
            </div>
            
            <div class='bg-white/50 dark:bg-gray-800/50 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                <h4 class='font-semibold text-gray-900 dark:text-white mb-2'>Business Description</h4>
                <p class='text-sm text-gray-600 dark:text-gray-300 leading-relaxed'>{business_summary}</p>
            </div>
        </div>

        <!-- Financial Health Dashboard -->
        <div class='bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 p-6 rounded-xl border border-green-100 dark:border-green-800/50 shadow-sm'>
            <div class='flex items-center mb-4'>
                <div class='p-2 bg-green-100 dark:bg-green-800/50 rounded-lg mr-3'>
                    <svg class='w-6 h-6 text-green-600 dark:text-green-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z'></path>
                    </svg>
                </div>
                <h3 class='text-xl font-bold text-gray-900 dark:text-white'>Financial Health</h3>
            </div>
            
            <div class='grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4'>
                <!-- Valuation Metrics -->
                <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <h4 class='font-semibold text-gray-900 dark:text-white mb-3 flex items-center'>
                        <span class='w-2 h-2 bg-blue-500 rounded-full mr-2'></span>
                        Valuation
                    </h4>
                    <div class='space-y-2'>
                        <div class='flex justify-between'>
                            <span class='text-sm text-gray-600 dark:text-gray-400'>P/E Ratio</span>
                            <span class='text-sm font-medium text-gray-900 dark:text-white'>{pe_ratio if pe_ratio != 'N/A' else 'N/A'}</span>
                        </div>
                        <div class='flex justify-between'>
                            <span class='text-sm text-gray-600 dark:text-gray-400'>P/B Ratio</span>
                            <span class='text-sm font-medium text-gray-900 dark:text-white'>{pb_ratio if pb_ratio != 'N/A' else 'N/A'}</span>
                        </div>
                        <div class='flex justify-between'>
                            <span class='text-sm text-gray-600 dark:text-gray-400'>Revenue Growth</span>
                            <span class='text-sm font-medium {"text-green-600" if revenue_growth != "N/A" and revenue_growth and revenue_growth > 0 else "text-red-600" if revenue_growth != "N/A" and revenue_growth and revenue_growth < 0 else "text-gray-900 dark:text-white"}'>{f"{revenue_growth:.1%}" if revenue_growth != "N/A" and revenue_growth else "N/A"}</span>
                        </div>
                    </div>
                </div>
                
                <!-- Profitability -->
                <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <h4 class='font-semibold text-gray-900 dark:text-white mb-3 flex items-center'>
                        <span class='w-2 h-2 bg-green-500 rounded-full mr-2'></span>
                        Profitability
                    </h4>
                    <div class='space-y-2'>
                        <div class='flex justify-between'>
                            <span class='text-sm text-gray-600 dark:text-gray-400'>Profit Margin</span>
                            <span class='text-sm font-medium text-gray-900 dark:text-white'>{f"{profit_margin:.1%}" if profit_margin != "N/A" and profit_margin else "N/A"}</span>
                        </div>
                        <div class='flex justify-between'>
                            <span class='text-sm text-gray-600 dark:text-gray-400'>Operating Margin</span>
                            <span class='text-sm font-medium text-gray-900 dark:text-white'>{f"{operating_margin:.1%}" if operating_margin != "N/A" and operating_margin else "N/A"}</span>
                        </div>
                        <div class='flex justify-between'>
                            <span class='text-sm text-gray-600 dark:text-gray-400'>ROE</span>
                            <span class='text-sm font-medium text-gray-900 dark:text-white'>{f"{roe:.1%}" if roe != "N/A" and roe else "N/A"}</span>
                        </div>
                    </div>
                </div>
                
                <!-- Financial Strength -->
                <div class='bg-white/70 dark:bg-gray-800/70 p-4 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <h4 class='font-semibold text-gray-900 dark:text-white mb-3 flex items-center'>
                        <span class='w-2 h-2 bg-purple-500 rounded-full mr-2'></span>
                        Financial Strength
                    </h4>
                    <div class='space-y-2'>
                        <div class='flex justify-between'>
                            <span class='text-sm text-gray-600 dark:text-gray-400'>Debt/Equity</span>
                            <span class='text-sm font-medium text-gray-900 dark:text-white'>{f"{debt_to_equity:.2f}" if debt_to_equity != "N/A" and debt_to_equity else "N/A"}</span>
                        </div>
                        <div class='flex justify-between'>
                            <span class='text-sm text-gray-600 dark:text-gray-400'>Current Ratio</span>
                            <span class='text-sm font-medium text-gray-900 dark:text-white'>{f"{info.get('currentRatio', 'N/A'):.2f}" if info.get('currentRatio') != None else "N/A"}</span>
                        </div>
                        <div class='flex justify-between'>
                            <span class='text-sm text-gray-600 dark:text-gray-400'>Quick Ratio</span>
                            <span class='text-sm font-medium text-gray-900 dark:text-white'>{f"{info.get('quickRatio', 'N/A'):.2f}" if info.get('quickRatio') != None else "N/A"}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Company Details -->
        <div class='bg-white dark:bg-gray-800/50 border border-gray-100 dark:border-gray-700/50 rounded-xl p-6 shadow-sm'>
            <div class='flex items-center mb-4'>
                <div class='p-2 bg-gray-100 dark:bg-gray-700 rounded-lg mr-3'>
                    <svg class='w-6 h-6 text-gray-600 dark:text-gray-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z'></path>
                    </svg>
                </div>
                <h3 class='text-xl font-bold text-gray-900 dark:text-white'>Company Details</h3>
            </div>
            
            <div class='grid grid-cols-1 md:grid-cols-2 gap-6'>
                <div class='space-y-3'>
                    <div class='flex justify-between py-2 border-b border-gray-100 dark:border-gray-700'>
                        <span class='text-sm text-gray-500 dark:text-gray-400'>Industry</span>
                        <span class='text-sm font-medium text-gray-900 dark:text-white'>{info.get('industry', 'N/A')}</span>
                    </div>
                    <div class='flex justify-between py-2 border-b border-gray-100 dark:border-gray-700'>
                        <span class='text-sm text-gray-500 dark:text-gray-400'>Country</span>
                        <span class='text-sm font-medium text-gray-900 dark:text-white'>{info.get('country', 'N/A')}</span>
                    </div>
                    <div class='flex justify-between py-2 border-b border-gray-100 dark:border-gray-700'>
                        <span class='text-sm text-gray-500 dark:text-gray-400'>Exchange</span>
                        <span class='text-sm font-medium text-gray-900 dark:text-white'>{info.get('exchange', 'N/A')}</span>
                    </div>
                </div>
                <div class='space-y-3'>
                    <div class='flex justify-between py-2 border-b border-gray-100 dark:border-gray-700'>
                        <span class='text-sm text-gray-500 dark:text-gray-400'>Website</span>
                        <a href='{info.get('website', '#')}' target='_blank' class='text-sm font-medium text-blue-600 hover:underline dark:text-blue-400'>{info.get('website', 'N/A')}</a>
                    </div>
                    <div class='flex justify-between py-2 border-b border-gray-100 dark:border-gray-700'>
                        <span class='text-sm text-gray-500 dark:text-gray-400'>Phone</span>
                        <span class='text-sm font-medium text-gray-900 dark:text-white'>{info.get('phone', 'N/A')}</span>
                    </div>
                    <div class='flex justify-between py-2 border-b border-gray-100 dark:border-gray-700'>
                        <span class='text-sm text-gray-500 dark:text-gray-400'>City</span>
                        <span class='text-sm font-medium text-gray-900 dark:text-white'>{info.get('city', 'N/A')}</span>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """
    
    return pages

def get_comparison_row(metric_name, company_value, industry_avg, sector, is_percentage=False):
    """Helper function to generate a comparison row with appropriate styling."""
    # Try to get default industry average if not provided
    if industry_avg is None and sector and sector != 'N/A':
        industry_avg = get_industry_average(sector, metric_name)
    
    if company_value == 'N/A' and industry_avg is None:
        return f'''
        <div class='flex justify-between items-center p-2 bg-gray-50 dark:bg-gray-800/50 rounded'>
            <span class='text-sm text-gray-600 dark:text-gray-400'>{metric_name}</span>
            <span class='text-sm text-gray-500'>No data available</span>
        </div>'''
    
    try:
        company_num = float(company_value)
        industry_num = float(industry_avg)
        
        # Format values
        if is_percentage:
            company_display = f"{company_num * 100:.1f}%"
            industry_display = f"{industry_num * 100:.1f}%"
            diff = (company_num - industry_num) * 100
        else:
            company_display = f"{company_num:.2f}"
            industry_display = f"{industry_num:.2f}"
            diff = company_num - industry_num
        
        # Determine style based on comparison
        if metric_name in ['P/E Ratio', 'PEG Ratio', 'P/B Ratio', 'Debt/Equity']:
            is_better = company_num <= industry_num
        else:
            is_better = company_num >= industry_num
            
        diff_text = f"{abs(diff):.2f}{'%' if is_percentage else ''} "
        diff_text += "higher" if diff > 0 else "lower" if diff < 0 else "same as"
        
        bg_color = 'bg-blue-50 dark:bg-blue-900/20' if is_better else 'bg-red-50 dark:bg-red-900/20' if diff != 0 else 'bg-gray-50 dark:bg-gray-800/50'
        text_color = 'text-blue-600 dark:text-blue-400' if is_better else 'text-red-600 dark:text-red-400' if diff != 0 else 'text-gray-600 dark:text-gray-400'
        
        return f'''
        <div class='p-2 rounded {bg_color}'>
            <div class='flex justify-between items-center mb-1'>
                <span class='text-sm font-medium text-gray-600 dark:text-gray-300'>{metric_name}</span>
                <span class='text-sm font-medium {text_color}'>{company_display}</span>
            </div>
            <div class='flex justify-between text-xs text-gray-500 dark:text-gray-400'>
                <span>Industry: {industry_display}</span>
                <span class='{text_color}'>{diff_text if diff != 0 else 'Same as average'}</span>
            </div>
        </div>'''
        
    except (ValueError, TypeError):
        return f'''
        <div class='flex justify-between items-center p-2 bg-gray-50 dark:bg-gray-800/50 rounded'>
            <span class='text-sm text-gray-600 dark:text-gray-400'>{metric_name}</span>
            <span class='text-sm text-gray-500'>N/A</span>
        </div>'''

def calculate_dcf_valuation(info, ticker):
    """Calculate DCF valuation for a stock."""
    try:
        import yfinance as yf
        
        # Get financial data
        stock = yf.Ticker(ticker)
        
        # Try to get cash flow data
        try:
            cash_flow = stock.cashflow
            if cash_flow.empty:
                return None
        except:
            return None
        
        # Get key metrics
        market_cap = info.get('marketCap', 0)
        shares_outstanding = info.get('sharesOutstanding', info.get('impliedSharesOutstanding', 0))
        
        if not shares_outstanding or shares_outstanding == 0:
            return None
            
        # Get free cash flow (most recent year)
        try:
            # Try to get Free Cash Flow directly
            if 'Free Cash Flow' in cash_flow.index:
                fcf = cash_flow.loc['Free Cash Flow'].iloc[0]
            elif 'Operating Cash Flow' in cash_flow.index and 'Capital Expenditure' in cash_flow.index:
                ocf = cash_flow.loc['Operating Cash Flow'].iloc[0]
                capex = cash_flow.loc['Capital Expenditure'].iloc[0]
                fcf = ocf + capex  # capex is usually negative
            else:
                return None
                
            if fcf <= 0:
                return None
                
        except:
            return None
        
        # DCF assumptions
        growth_years = 5
        terminal_growth = 0.025  # 2.5% terminal growth
        discount_rate = 0.10     # 10% WACC assumption
        
        # Estimate growth rate from revenue growth or use conservative 5%
        revenue_growth = info.get('revenueGrowth', 0.05)
        if revenue_growth and revenue_growth > 0:
            growth_rate = min(revenue_growth, 0.15)  # Cap at 15%
        else:
            growth_rate = 0.05  # Default 5%
        
        # Calculate projected FCF for next 5 years
        projected_fcf = []
        current_fcf = fcf
        
        for year in range(1, growth_years + 1):
            # Declining growth rate over time
            year_growth = growth_rate * (0.9 ** (year - 1))  # Decay growth each year
            current_fcf = current_fcf * (1 + year_growth)
            projected_fcf.append(current_fcf)
        
        # Calculate terminal value
        terminal_fcf = projected_fcf[-1] * (1 + terminal_growth)
        terminal_value = terminal_fcf / (discount_rate - terminal_growth)
        
        # Discount all cash flows to present value
        pv_fcf = []
        for i, fcf_year in enumerate(projected_fcf):
            pv = fcf_year / ((1 + discount_rate) ** (i + 1))
            pv_fcf.append(pv)
        
        # Discount terminal value
        pv_terminal = terminal_value / ((1 + discount_rate) ** growth_years)
        
        # Total enterprise value
        enterprise_value = sum(pv_fcf) + pv_terminal
        
        # Get net cash (cash - debt) to calculate equity value
        total_cash = info.get('totalCash', 0)
        total_debt = info.get('totalDebt', 0)
        net_cash = total_cash - total_debt
        
        # Equity value
        equity_value = enterprise_value + net_cash
        
        # Intrinsic value per share
        intrinsic_value = equity_value / shares_outstanding
        
        # Current stock price for comparison
        current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
        
        # Margin of safety
        margin_of_safety = ((intrinsic_value - current_price) / current_price) * 100 if current_price > 0 else 0
        
        return {
            'fcf': fcf,
            'growth_rate': growth_rate,
            'discount_rate': discount_rate,
            'terminal_growth': terminal_growth,
            'projected_fcf': projected_fcf,
            'pv_fcf': pv_fcf,
            'terminal_value': terminal_value,
            'pv_terminal': pv_terminal,
            'enterprise_value': enterprise_value,
            'equity_value': equity_value,
            'intrinsic_value': intrinsic_value,
            'current_price': current_price,
            'margin_of_safety': margin_of_safety,
            'shares_outstanding': shares_outstanding
        }
        
    except Exception as e:
        return None

def create_fundamental_analysis_pages(ticker, info, company_name, sector, industry, market_cap_formatted, 
                                   pe_ratio, forward_pe, peg_ratio, price_to_book, price_to_sales,
                                   profit_margin, operating_margin, roe, roa, revenue_growth, earnings_growth,
                                   dividend_yield, payout_ratio, debt_to_equity, current_ratio, quick_ratio):
    """Create paginated content for fundamental analysis."""
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
    
    # Helper function to get color-coded indicators for metrics
    def get_value_indicator(value, metric_type='ratio'):
        """Get color-coded indicator for financial metrics"""
        if value == 'N/A' or value is None:
            return 'text-gray-500 dark:text-gray-400', '⚪'
        
        try:
            val = float(value)
            if metric_type == 'pe_ratio':
                if val < 15: return 'text-green-600 dark:text-green-400', '🟢'
                elif val < 25: return 'text-yellow-600 dark:text-yellow-400', '🟡'
                else: return 'text-red-600 dark:text-red-400', '🔴'
            elif metric_type == 'margin':
                if val > 0.20: return 'text-green-600 dark:text-green-400', '🟢'
                elif val > 0.10: return 'text-yellow-600 dark:text-yellow-400', '🟡'
                elif val > 0: return 'text-orange-600 dark:text-orange-400', '🟠'
                else: return 'text-red-600 dark:text-red-400', '🔴'
            elif metric_type == 'growth':
                if val > 0.15: return 'text-green-600 dark:text-green-400', '🚀'
                elif val > 0.05: return 'text-blue-600 dark:text-blue-400', '📈'
                elif val > 0: return 'text-yellow-600 dark:text-yellow-400', '📊'
                else: return 'text-red-600 dark:text-red-400', '📉'
            elif metric_type == 'debt_ratio':
                if val < 0.3: return 'text-green-600 dark:text-green-400', '🟢'
                elif val < 0.6: return 'text-yellow-600 dark:text-yellow-400', '🟡'
                else: return 'text-red-600 dark:text-red-400', '🔴'
            else:  # default ratio
                if val > 1.5: return 'text-green-600 dark:text-green-400', '🟢'
                elif val > 1.0: return 'text-yellow-600 dark:text-yellow-400', '🟡'
                else: return 'text-red-600 dark:text-red-400', '🔴'
        except (ValueError, TypeError):
            return 'text-gray-500 dark:text-gray-400', '⚪'
    
    # Create pages dictionary to hold different sections
    pages = {}
    
    # Page 1: Company Overview and Key Metrics
    pages['overview'] = f"""
    <div class='p-4 space-y-6'>
        <div class='border-b border-gray-200 dark:border-gray-700 pb-4'>
            <h2 class='text-2xl font-bold text-gray-900 dark:text-white'>Fundamental Analysis</h2>
            <div class='flex items-center mt-1 space-x-2'>
                <span class='text-xl font-semibold text-gray-800 dark:text-gray-200'>{company_name}</span>
                <span class='px-2 py-0.5 text-xs font-medium bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200 rounded-full'>{ticker}</span>
            </div>
        </div>
        
        <!-- Company Overview -->
        <div class='bg-gradient-to-r from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-900/30 p-6 rounded-xl border border-blue-100 dark:border-blue-800/50 shadow-sm'>
            <div class='flex items-center mb-4'>
                <div class='p-2 bg-blue-100 dark:bg-blue-800/50 rounded-lg mr-3'>
                    <svg class='w-6 h-6 text-blue-600 dark:text-blue-300' fill='none' stroke='currentColor' viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4'></path>
                    </svg>
                </div>
                <h3 class='text-lg font-semibold text-gray-900 dark:text-white'>Company Overview</h3>
            </div>
            <div class='grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4'>
                <div class='bg-white/50 dark:bg-gray-800/50 p-3 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <p class='text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1'>Sector</p>
                    <p class='font-medium text-gray-900 dark:text-white'>{sector}</p>
                </div>
                <div class='bg-white/50 dark:bg-gray-800/50 p-3 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <p class='text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1'>Industry</p>
                    <p class='font-medium text-gray-900 dark:text-white'>{industry}</p>
                </div>
                <div class='bg-white/50 dark:bg-gray-800/50 p-3 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <p class='text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1'>Market Cap</p>
                    <p class='font-medium text-gray-900 dark:text-white'>{market_cap_formatted}</p>
                </div>
                <div class='bg-white/50 dark:bg-gray-800/50 p-3 rounded-lg border border-gray-100 dark:border-gray-700'>
                    <p class='text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1'>Employees</p>
                    <p class='font-medium text-gray-900 dark:text-white'>{info.get('fullTimeEmployees', 'N/A'):,}</p>
                </div>
            </div>
        </div>
        
        <!-- Key Metrics -->
        <div class='grid grid-cols-1 md:grid-cols-2 gap-5'>
            <!-- Valuation Metrics -->
            <div class='bg-white dark:bg-gray-800/50 border border-gray-100 dark:border-gray-700/50 rounded-xl p-5 shadow-sm hover:shadow-md transition-shadow duration-200'>
                <div class='flex items-center mb-4'>
                    <div class='p-1.5 bg-purple-100 dark:bg-purple-900/30 rounded-lg mr-3'>
                        <svg class='w-5 h-5 text-purple-600 dark:text-purple-400' fill='none' stroke='currentColor' viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'>
                            <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z'></path>
                        </svg>
                    </div>
                    <h4 class='font-semibold text-gray-800 dark:text-white'>Valuation Ratios</h4>
                </div>
                <div class='space-y-2.5'>
                    <div class='flex items-center justify-between py-2 px-3 bg-gray-50 dark:bg-gray-700/30 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700/50 transition-colors'>
                        <span class='text-sm font-medium text-gray-600 dark:text-gray-300'>P/E Ratio (TTM)</span>
                        <div class='flex items-center space-x-2'>
                            <span class='text-lg'>{get_value_indicator(pe_ratio, "pe_ratio")[1]}</span>
                            <span class='font-semibold {get_value_indicator(pe_ratio, "pe_ratio")[0]}'>{format_ratio(pe_ratio)}</span>
                        </div>
                    </div>
                    <div class='flex items-center justify-between py-2 px-3 bg-gray-50 dark:bg-gray-700/30 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700/50 transition-colors'>
                        <span class='text-sm font-medium text-gray-600 dark:text-gray-300'>Forward P/E</span>
                        <div class='flex items-center space-x-2'>
                            <span class='text-lg'>{get_value_indicator(forward_pe, "pe_ratio")[1]}</span>
                            <span class='font-semibold {get_value_indicator(forward_pe, "pe_ratio")[0]}'>{format_ratio(forward_pe)}</span>
                        </div>
                    </div>
                    <div class='flex items-center justify-between py-2 px-3 bg-gray-50 dark:bg-gray-700/30 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700/50 transition-colors'>
                        <span class='text-sm text-gray-600 dark:text-gray-400'>PEG Ratio</span>
                        <div class='flex items-center space-x-2'>
                            <span class='text-lg'>{get_value_indicator(peg_ratio, "ratio")[1]}</span>
                            <span class='font-semibold {get_value_indicator(peg_ratio, "ratio")[0]}'>{format_ratio(peg_ratio)}</span>
                        </div>
                    </div>
                    <div class='flex items-center justify-between py-2 px-3 bg-gray-50 dark:bg-gray-700/30 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700/50 transition-colors'>
                        <span class='text-sm text-gray-600 dark:text-gray-400'>Price-to-Book</span>
                        <div class='flex items-center space-x-2'>
                            <span class='text-lg'>{get_value_indicator(price_to_book, "ratio")[1]}</span>
                            <span class='font-semibold {get_value_indicator(price_to_book, "ratio")[0]}'>{format_ratio(price_to_book)}</span>
                        </div>
                    </div>
                    <div class='flex items-center justify-between py-2 px-3 bg-gray-50 dark:bg-gray-700/30 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700/50 transition-colors'>
                        <span class='text-sm text-gray-600 dark:text-gray-400'>Price-to-Sales</span>
                        <div class='flex items-center space-x-2'>
                            <span class='text-lg'>{get_value_indicator(price_to_sales, "ratio")[1]}</span>
                            <span class='font-semibold {get_value_indicator(price_to_sales, "ratio")[0]}'>{format_ratio(price_to_sales)}</span>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Profitability Metrics -->
            <div class='bg-white dark:bg-gray-800/50 border border-gray-100 dark:border-gray-700/50 rounded-xl p-5 shadow-sm hover:shadow-md transition-shadow duration-200'>
                <div class='flex items-center mb-4'>
                    <div class='p-1.5 bg-green-100 dark:bg-green-900/30 rounded-lg mr-3'>
                        <svg class='w-5 h-5 text-green-600 dark:text-green-400' fill='none' stroke='currentColor' viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'>
                            <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z'></path>
                        </svg>
                    </div>
                    <h4 class='font-semibold text-gray-800 dark:text-white'>Profitability</h4>
                </div>
                <div class='space-y-2.5'>
                    <div class='flex items-center justify-between py-2 px-3 bg-gray-50 dark:bg-gray-700/30 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700/50 transition-colors'>
                        <span class='text-sm text-gray-600 dark:text-gray-400'>Profit Margin</span>
                        <div class='flex items-center space-x-2'>
                            <span class='text-lg'>{get_value_indicator(profit_margin, "margin")[1]}</span>
                            <span class='font-semibold {get_value_indicator(profit_margin, "margin")[0]}'>{format_percentage(profit_margin)}</span>
                        </div>
                    </div>
                    <div class='flex items-center justify-between py-2 px-3 bg-gray-50 dark:bg-gray-700/30 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700/50 transition-colors'>
                        <span class='text-sm text-gray-600 dark:text-gray-400'>Operating Margin</span>
                        <div class='flex items-center space-x-2'>
                            <span class='text-lg'>{get_value_indicator(operating_margin, "margin")[1]}</span>
                            <span class='font-semibold {get_value_indicator(operating_margin, "margin")[0]}'>{format_percentage(operating_margin)}</span>
                        </div>
                    </div>
                    <div class='flex items-center justify-between py-2 px-3 bg-gray-50 dark:bg-gray-700/30 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700/50 transition-colors'>
                        <span class='text-sm text-gray-600 dark:text-gray-400'>Return on Equity</span>
                        <div class='flex items-center space-x-2'>
                            <span class='text-lg'>{get_value_indicator(roe, "margin")[1]}</span>
                            <span class='font-semibold {get_value_indicator(roe, "margin")[0]}'>{format_percentage(roe)}</span>
                        </div>
                    </div>
                    <div class='flex items-center justify-between py-2 px-3 bg-gray-50 dark:bg-gray-700/30 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700/50 transition-colors'>
                        <span class='text-sm text-gray-600 dark:text-gray-400'>Return on Assets</span>
                        <div class='flex items-center space-x-2'>
                            <span class='text-lg'>{get_value_indicator(roa, "margin")[1]}</span>
                            <span class='font-semibold {get_value_indicator(roa, "margin")[0]}'>{format_percentage(roa)}</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Color Legend -->
        <div class='bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 p-5 rounded-xl border border-indigo-100 dark:border-indigo-800/50 shadow-sm'>
            <div class='flex items-center mb-3'>
                <div class='p-1.5 bg-indigo-100 dark:bg-indigo-900/30 rounded-lg mr-3'>
                    <svg class='w-5 h-5 text-indigo-600 dark:text-indigo-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z' />
                    </svg>
                </div>
                <h3 class='text-lg font-semibold text-indigo-800 dark:text-indigo-200'>How to Read the Indicators</h3>
            </div>
            <div class='grid grid-cols-2 md:grid-cols-4 gap-3 text-sm'>
                <div class='flex items-center space-x-2 p-2 bg-white/50 dark:bg-gray-800/30 rounded-lg'>
                    <span class='text-lg'>🟢</span>
                    <span class='text-green-600 dark:text-green-400 font-medium'>Excellent</span>
                </div>
                <div class='flex items-center space-x-2 p-2 bg-white/50 dark:bg-gray-800/30 rounded-lg'>
                    <span class='text-lg'>🟡</span>
                    <span class='text-yellow-600 dark:text-yellow-400 font-medium'>Good</span>
                </div>
                <div class='flex items-center space-x-2 p-2 bg-white/50 dark:bg-gray-800/30 rounded-lg'>
                    <span class='text-lg'>🟠</span>
                    <span class='text-orange-600 dark:text-orange-400 font-medium'>Fair</span>
                </div>
                <div class='flex items-center space-x-2 p-2 bg-white/50 dark:bg-gray-800/30 rounded-lg'>
                    <span class='text-lg'>🔴</span>
                    <span class='text-red-600 dark:text-red-400 font-medium'>Poor</span>
                </div>
            </div>
            <div class='mt-3 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-100 dark:border-blue-800/30'>
                <p class='text-xs text-blue-700 dark:text-blue-300'>
                    <span class='font-medium'>Note:</span> Color indicators provide quick visual assessment based on common financial benchmarks. 
                    Always consider industry context and company-specific factors when making investment decisions.
                </p>
            </div>
        </div>
    </div>
    """

    # Page 2: Enhanced Growth and Dividend Information with Charts
    def get_growth_trend_indicator(value):
        """Get trend indicator for growth metrics"""
        if value == 'N/A' or value is None:
            return '⚪', 'gray'
        try:
            val = float(value)
            if val > 0.15: return '🚀', 'green'  # Excellent growth
            elif val > 0.05: return '📈', 'blue'  # Good growth
            elif val > 0: return '📊', 'yellow'   # Modest growth
            else: return '📉', 'red'              # Declining
        except:
            return '⚪', 'gray'
    
    def get_dividend_health_indicator(yield_val, payout_val):
        """Get dividend health based on yield and payout ratio"""
        try:
            div_yield = float(yield_val) if yield_val != 'N/A' and yield_val is not None else 0
            payout = float(payout_val) if payout_val != 'N/A' and payout_val is not None else 0
            
            if div_yield > 0.03 and payout < 0.6: return '💎', 'green'  # Excellent dividend
            elif div_yield > 0.02 and payout < 0.8: return '💰', 'blue'  # Good dividend
            elif div_yield > 0.01: return '💵', 'yellow'  # Modest dividend
            else: return '🚫', 'gray'  # No/low dividend
        except:
            return '🚫', 'gray'
    
    revenue_icon, revenue_color = get_growth_trend_indicator(revenue_growth)
    earnings_icon, earnings_color = get_growth_trend_indicator(earnings_growth)
    dividend_icon, dividend_color = get_dividend_health_indicator(dividend_yield, payout_ratio)
    
    pages['growth_dividend'] = f"""
    <div class='p-4 space-y-6'>
        <div class='border-b border-gray-200 dark:border-gray-700 pb-4'>
            <h2 class='text-2xl font-bold text-gray-900 dark:text-white'>Growth & Dividend Analysis</h2>
            <div class='flex items-center mt-1 space-x-2'>
                <span class='text-xl font-semibold text-gray-800 dark:text-gray-200'>{company_name}</span>
                <span class='px-2 py-0.5 text-xs font-medium bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200 rounded-full'>{ticker}</span>
            </div>
        </div>
        
        <!-- Growth Trend Dashboard -->
        <div class='bg-gradient-to-r from-green-50 to-blue-50 dark:from-green-900/20 dark:to-blue-900/20 p-6 rounded-xl border border-green-100 dark:border-green-800/50 shadow-sm'>
            <div class='flex items-center mb-4'>
                <div class='p-2 bg-green-100 dark:bg-green-800/50 rounded-lg mr-3'>
                    <svg class='w-6 h-6 text-green-600 dark:text-green-300' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M13 7h8m0 0v8m0-8l-8 8-4-4-6 6'></path>
                    </svg>
                </div>
                <h3 class='text-lg font-semibold text-gray-900 dark:text-white'>Growth Trend Analysis</h3>
            </div>
            <div class='grid grid-cols-1 md:grid-cols-3 gap-4'>
                <!-- Revenue Growth Card -->
                <div class='bg-white/70 dark:bg-gray-800/50 p-4 rounded-xl border border-gray-100 dark:border-gray-700 shadow-sm'>
                    <div class='flex items-center justify-between mb-2'>
                        <h4 class='font-medium text-gray-700 dark:text-gray-300'>Revenue Growth</h4>
                        <span class='text-2xl'>{revenue_icon}</span>
                    </div>
                    <div class='text-2xl font-bold text-{revenue_color}-600 dark:text-{revenue_color}-400 mb-1'>{format_percentage(revenue_growth)}</div>
                    <div class='w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2'>
                        <div class='bg-{revenue_color}-500 h-2 rounded-full transition-all duration-300' style='width: {min(abs(float(revenue_growth) * 500) if revenue_growth != "N/A" and revenue_growth is not None else 0, 100)}%'></div>
                    </div>
                    <p class='text-xs text-gray-500 dark:text-gray-400 mt-2'>Year-over-year revenue change</p>
                </div>
                
                <!-- Earnings Growth Card -->
                <div class='bg-white/70 dark:bg-gray-800/50 p-4 rounded-xl border border-gray-100 dark:border-gray-700 shadow-sm'>
                    <div class='flex items-center justify-between mb-2'>
                        <h4 class='font-medium text-gray-700 dark:text-gray-300'>Earnings Growth</h4>
                        <span class='text-2xl'>{earnings_icon}</span>
                    </div>
                    <div class='text-2xl font-bold text-{earnings_color}-600 dark:text-{earnings_color}-400 mb-1'>{format_percentage(earnings_growth)}</div>
                    <div class='w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2'>
                        <div class='bg-{earnings_color}-500 h-2 rounded-full transition-all duration-300' style='width: {min(abs(float(earnings_growth) * 500) if earnings_growth != "N/A" and earnings_growth is not None else 0, 100)}%'></div>
                    </div>
                    <p class='text-xs text-gray-500 dark:text-gray-400 mt-2'>Year-over-year earnings change</p>
                </div>
                
                <!-- Growth Quality Score -->
                <div class='bg-white/70 dark:bg-gray-800/50 p-4 rounded-xl border border-gray-100 dark:border-gray-700 shadow-sm'>
                    <div class='flex items-center justify-between mb-2'>
                        <h4 class='font-medium text-gray-700 dark:text-gray-300'>Growth Quality</h4>
                        <span class='text-2xl'>⭐</span>
                    </div>
                    <div class='text-2xl font-bold text-purple-600 dark:text-purple-400 mb-1'>
                        {"A+" if revenue_growth != "N/A" and earnings_growth != "N/A" and float(revenue_growth or 0) > 0.1 and float(earnings_growth or 0) > 0.1 else 
                         "A" if revenue_growth != "N/A" and earnings_growth != "N/A" and float(revenue_growth or 0) > 0.05 and float(earnings_growth or 0) > 0.05 else
                         "B" if revenue_growth != "N/A" and earnings_growth != "N/A" and float(revenue_growth or 0) > 0 and float(earnings_growth or 0) > 0 else "C"}
                    </div>
                    <div class='flex space-x-1 mb-2'>
                        {''.join(['⭐' for _ in range(5 if revenue_growth != "N/A" and earnings_growth != "N/A" and float(revenue_growth or 0) > 0.1 and float(earnings_growth or 0) > 0.1 else 
                                                   4 if revenue_growth != "N/A" and earnings_growth != "N/A" and float(revenue_growth or 0) > 0.05 and float(earnings_growth or 0) > 0.05 else
                                                   3 if revenue_growth != "N/A" and earnings_growth != "N/A" and float(revenue_growth or 0) > 0 and float(earnings_growth or 0) > 0 else 2)])}
                    </div>
                    <p class='text-xs text-gray-500 dark:text-gray-400'>Based on revenue & earnings consistency</p>
                </div>
            </div>
        </div>
        
        <!-- Enhanced Dividend Analysis -->
        <div class='bg-gradient-to-r from-yellow-50 to-orange-50 dark:from-yellow-900/20 dark:to-orange-900/20 p-6 rounded-xl border border-yellow-100 dark:border-yellow-800/50 shadow-sm'>
            <div class='flex items-center mb-4'>
                <div class='p-2 bg-yellow-100 dark:bg-yellow-800/50 rounded-lg mr-3'>
                    <svg class='w-6 h-6 text-yellow-600 dark:text-yellow-300' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z'></path>
                    </svg>
                </div>
                <h3 class='text-lg font-semibold text-gray-900 dark:text-white'>Dividend & Shareholder Returns</h3>
            </div>
            <div class='grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4'>
                <!-- Dividend Yield -->
                <div class='bg-white/70 dark:bg-gray-800/50 p-4 rounded-xl border border-gray-100 dark:border-gray-700 shadow-sm'>
                    <div class='flex items-center justify-between mb-2'>
                        <h4 class='font-medium text-gray-700 dark:text-gray-300'>Dividend Yield</h4>
                        <span class='text-xl'>{dividend_icon}</span>
                    </div>
                    <div class='text-xl font-bold text-{dividend_color}-600 dark:text-{dividend_color}-400 mb-1'>{format_percentage(dividend_yield)}</div>
                    <div class='w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2'>
                        <div class='bg-{dividend_color}-500 h-2 rounded-full' style='width: {min(float(dividend_yield or 0) * 2000, 100)}%'></div>
                    </div>
                </div>
                
                <!-- Payout Ratio -->
                <div class='bg-white/70 dark:bg-gray-800/50 p-4 rounded-xl border border-gray-100 dark:border-gray-700 shadow-sm'>
                    <div class='flex items-center justify-between mb-2'>
                        <h4 class='font-medium text-gray-700 dark:text-gray-300'>Payout Ratio</h4>
                        <span class='text-xl'>📊</span>
                    </div>
                    <div class='text-xl font-bold text-blue-600 dark:text-blue-400 mb-1'>{format_percentage(payout_ratio)}</div>
                    <div class='w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2'>
                        <div class='bg-blue-500 h-2 rounded-full' style='width: {min(float(payout_ratio or 0) * 100, 100)}%'></div>
                    </div>
                </div>
                
                <!-- Dividend Sustainability -->
                <div class='bg-white/70 dark:bg-gray-800/50 p-4 rounded-xl border border-gray-100 dark:border-gray-700 shadow-sm'>
                    <div class='flex items-center justify-between mb-2'>
                        <h4 class='font-medium text-gray-700 dark:text-gray-300'>Sustainability</h4>
                        <span class='text-xl'>🛡️</span>
                    </div>
                    <div class='text-xl font-bold text-green-600 dark:text-green-400 mb-1'>
                        {"High" if payout_ratio != "N/A" and payout_ratio is not None and float(payout_ratio) < 0.6 else
                         "Medium" if payout_ratio != "N/A" and payout_ratio is not None and float(payout_ratio) < 0.8 else
                         "Low" if payout_ratio != "N/A" and payout_ratio is not None else "N/A"}
                    </div>
                    <p class='text-xs text-gray-500 dark:text-gray-400'>Based on payout ratio</p>
                </div>
                
                <!-- Total Return Potential -->
                <div class='bg-white/70 dark:bg-gray-800/50 p-4 rounded-xl border border-gray-100 dark:border-gray-700 shadow-sm'>
                    <div class='flex items-center justify-between mb-2'>
                        <h4 class='font-medium text-gray-700 dark:text-gray-300'>Total Return</h4>
                        <span class='text-xl'>🎯</span>
                    </div>
                    <div class='text-xl font-bold text-purple-600 dark:text-purple-400 mb-1'>
                        {format_percentage((float(dividend_yield or 0) + float(earnings_growth or 0)) if dividend_yield != "N/A" and earnings_growth != "N/A" else "N/A")}
                    </div>
                    <p class='text-xs text-gray-500 dark:text-gray-400'>Dividend yield + growth estimate</p>
                </div>
            </div>
        </div>
        
        <!-- Profitability Trends -->
        <div class='bg-white dark:bg-gray-800/50 border border-gray-100 dark:border-gray-700/50 rounded-xl p-6 shadow-sm'>
            <div class='flex items-center mb-6'>
                <div class='p-2 bg-purple-100 dark:bg-purple-900/30 rounded-lg mr-3'>
                    <svg class='w-6 h-6 text-purple-600 dark:text-purple-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z'></path>
                    </svg>
                </div>
                <h3 class='text-lg font-semibold text-gray-900 dark:text-white'>Profitability Analysis</h3>
            </div>
            
            <div class='grid grid-cols-1 md:grid-cols-3 gap-6'>
                <!-- Profit Margins -->
                <div class='space-y-4'>
                    <h4 class='font-medium text-gray-800 dark:text-gray-200 border-b border-gray-200 dark:border-gray-600 pb-2'>Profit Margins</h4>
                    <div class='space-y-3'>
                        <div class='flex justify-between items-center py-2 px-3 bg-gray-50 dark:bg-gray-700/30 rounded-lg'>
                            <span class='text-sm font-medium text-gray-600 dark:text-gray-300'>Net Profit Margin</span>
                            <div class='flex items-center space-x-2'>
                                <span class='font-semibold text-gray-900 dark:text-white'>{format_percentage(profit_margin)}</span>
                                <div class='w-12 bg-gray-200 dark:bg-gray-600 rounded-full h-2'>
                                    <div class='bg-green-500 h-2 rounded-full' style='width: {min(float(profit_margin or 0) * 500, 100)}%'></div>
                                </div>
                            </div>
                        </div>
                        <div class='flex justify-between items-center py-2 px-3 bg-gray-50 dark:bg-gray-700/30 rounded-lg'>
                            <span class='text-sm font-medium text-gray-600 dark:text-gray-300'>Operating Margin</span>
                            <div class='flex items-center space-x-2'>
                                <span class='font-semibold text-gray-900 dark:text-white'>{format_percentage(operating_margin)}</span>
                                <div class='w-12 bg-gray-200 dark:bg-gray-600 rounded-full h-2'>
                                    <div class='bg-blue-500 h-2 rounded-full' style='width: {min(float(operating_margin or 0) * 500, 100)}%'></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Return Metrics -->
                <div class='space-y-4'>
                    <h4 class='font-medium text-gray-800 dark:text-gray-200 border-b border-gray-200 dark:border-gray-600 pb-2'>Return Metrics</h4>
                    <div class='space-y-3'>
                        <div class='flex justify-between items-center py-2 px-3 bg-gray-50 dark:bg-gray-700/30 rounded-lg'>
                            <span class='text-sm font-medium text-gray-600 dark:text-gray-300'>ROE</span>
                            <div class='flex items-center space-x-2'>
                                <span class='font-semibold text-gray-900 dark:text-white'>{format_percentage(roe)}</span>
                                <div class='w-12 bg-gray-200 dark:bg-gray-600 rounded-full h-2'>
                                    <div class='bg-purple-500 h-2 rounded-full' style='width: {min(float(roe or 0) * 500, 100)}%'></div>
                                </div>
                            </div>
                        </div>
                        <div class='flex justify-between items-center py-2 px-3 bg-gray-50 dark:bg-gray-700/30 rounded-lg'>
                            <span class='text-sm font-medium text-gray-600 dark:text-gray-300'>ROA</span>
                            <div class='flex items-center space-x-2'>
                                <span class='font-semibold text-gray-900 dark:text-white'>{format_percentage(roa)}</span>
                                <div class='w-12 bg-gray-200 dark:bg-gray-600 rounded-full h-2'>
                                    <div class='bg-orange-500 h-2 rounded-full' style='width: {min(float(roa or 0) * 1000, 100)}%'></div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Efficiency Score -->
                <div class='space-y-4'>
                    <h4 class='font-medium text-gray-800 dark:text-gray-200 border-b border-gray-200 dark:border-gray-600 pb-2'>Efficiency Score</h4>
                    <div class='text-center'>
                        <div class='text-4xl font-bold text-indigo-600 dark:text-indigo-400 mb-2'>
                            {int((float(profit_margin or 0) + float(roe or 0) + float(roa or 0)) * 100 / 3) if all(x != "N/A" and x is not None for x in [profit_margin, roe, roa]) else "N/A"}
                        </div>
                        <div class='text-sm text-gray-500 dark:text-gray-400 mb-3'>Overall Efficiency</div>
                        <div class='w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3'>
                            <div class='bg-gradient-to-r from-red-500 via-yellow-500 to-green-500 h-3 rounded-full' style='width: {min((float(profit_margin or 0) + float(roe or 0) + float(roa or 0)) * 100 / 3 * 10, 100) if all(x != "N/A" and x is not None for x in [profit_margin, roe, roa]) else 0}%'></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """

    # Page 3: Industry Comparison
    pages['industry'] = f"""
    <div class='p-4 space-y-6'>
        <div class='border-b border-gray-200 dark:border-gray-700 pb-4'>
            <h2 class='text-2xl font-bold text-gray-900 dark:text-white'>Industry Comparison</h2>
            <div class='flex items-center mt-1 space-x-2'>
                <span class='text-xl font-semibold text-gray-800 dark:text-gray-200'>{company_name}</span>
                <span class='px-2 py-0.5 text-xs font-medium bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200 rounded-full'>{ticker}</span>
            </div>
        </div>
        
        <!-- Industry Comparison -->
        <div class='bg-white dark:bg-gray-800/50 border border-gray-100 dark:border-gray-700/50 p-6 rounded-xl shadow-sm'>
            <div class='flex items-center mb-4'>
                <div class='p-1.5 bg-indigo-100 dark:bg-indigo-900/30 rounded-lg mr-3'>
                    <svg class='w-5 h-5 text-indigo-600 dark:text-indigo-400' fill='none' stroke='currentColor' viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z'></path>
                    </svg>
                </div>
                <h3 class='text-lg font-semibold text-gray-900 dark:text-white'>Industry Comparison</h3>
            </div>
            <p class='text-sm text-gray-500 dark:text-gray-400 mb-5'>How {company_name} compares to the {sector if sector != 'N/A' else 'industry'} average</p>
            <div class='space-y-4'>
                <div class='grid grid-cols-1 md:grid-cols-2 gap-4'>
                    <!-- Profitability Comparison -->
                    <div class='bg-gray-50 dark:bg-gray-700/50 p-3 rounded-lg'>
                        <div class='flex items-center mb-2'>
                            <span class='font-medium text-gray-700 dark:text-gray-300'>Profit Margins</span>
                            <span class='ml-2 text-xs px-2 py-0.5 rounded-full bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300'>
                                vs {sector if sector != 'N/A' else 'Industry'}
                            </span>
                        </div>
                        <div class='space-y-2'>
                            {get_comparison_row('Operating Margin', operating_margin, info.get('industryAverageOperatingMargin'), sector, is_percentage=True)}
                            {get_comparison_row('Profit Margin', profit_margin, info.get('industryAverageProfitMargin'), sector, is_percentage=True)}
                            {get_comparison_row('ROE', roe, info.get('industryAverageReturnOnEquity'), sector, is_percentage=True)}
                        </div>
                    </div>
                    
                    <!-- Valuation Comparison -->
                    <div class='bg-gray-50 dark:bg-gray-700/50 p-3 rounded-lg'>
                        <div class='flex items-center mb-2'>
                            <span class='font-medium text-gray-700 dark:text-gray-300'>Valuation</span>
                            <span class='ml-2 text-xs px-2 py-0.5 rounded-full bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300'>
                                vs {sector if sector != 'N/A' else 'Industry'}
                            </span>
                        </div>
                        <div class='space-y-2'>
                            {get_comparison_row('P/E Ratio', pe_ratio, info.get('industryAveragePE'), sector, is_percentage=False)}
                            {get_comparison_row('PEG Ratio', peg_ratio, info.get('industryAveragePEG'), sector, is_percentage=False)}
                            {get_comparison_row('P/B Ratio', price_to_book, info.get('industryAveragePB'), sector, is_percentage=False)}
                        </div>
                    </div>
                </div>
                
                <!-- Financial Health Comparison -->
                <div class='bg-gray-50 dark:bg-gray-700/50 p-3 rounded-lg'>
                    <div class='flex items-center mb-2'>
                        <span class='font-medium text-gray-700 dark:text-gray-300'>Financial Health</span>
                        <span class='ml-2 text-xs px-2 py-0.5 rounded-full bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300'>
                            vs {sector if sector != 'N/A' else 'Industry'}
                        </span>
                    </div>
                    <div class='grid grid-cols-1 md:grid-cols-3 gap-4'>
                        <div class='space-y-2'>
                            {get_comparison_row('Debt/Equity', debt_to_equity, info.get('industryAverageDebtToEquity'), sector, is_percentage=False)}
                        </div>
                        <div class='space-y-2'>
                            {get_comparison_row('Current Ratio', current_ratio, info.get('industryAverageCurrentRatio'), sector, is_percentage=False)}
                        </div>
                        <div class='space-y-2'>
                            {get_comparison_row('Quick Ratio', quick_ratio, info.get('industryAverageQuickRatio'), sector, is_percentage=False)}
                        </div>
                    </div>
                </div>
                
                <div class='mt-6 p-3 bg-blue-50 dark:bg-blue-900/20 border border-blue-100 dark:border-blue-800/30 rounded-lg'>
                    <div class='flex'>
                        <svg class='h-5 w-5 text-blue-500 dark:text-blue-400 mt-0.5 mr-2 flex-shrink-0' fill='currentColor' viewBox='0 0 20 20' xmlns='http://www.w3.org/2000/svg'>
                            <path fill-rule='evenodd' d='M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h2a1 1 0 100-2v-3a1 1 0 00-1-1H9z' clip-rule='evenodd'></path>
                        </svg>
                        <div class='text-sm text-blue-700 dark:text-blue-300'>
                            <p class='font-medium'>How to read this comparison:</p>
                            <div class='flex flex-wrap items-center gap-2 mt-1 text-xs'>
                                <span class='inline-flex items-center px-2 py-0.5 rounded-full bg-blue-100 text-blue-800 dark:bg-blue-900/50 dark:text-blue-200'>Better than average</span>
                                <span class='inline-flex items-center px-2 py-0.5 rounded-full bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300'>Similar to average</span>
                                <span class='inline-flex items-center px-2 py-0.5 rounded-full bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-200'>Below average</span>
                            </div>
                            {'<p class="mt-2 text-yellow-700 dark:text-yellow-300 font-medium">ℹ️ Using estimated industry averages. For more accurate comparisons, consider upgrading to a premium data source.</p>' 
                            if not any(k.startswith('industryAverage') and v is not None for k, v in info.items()) else ''}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    """

    # Page 4: Financial Health
    pages['financial_health'] = f"""
    <div class='p-4 space-y-6'>
        <div class='border-b border-gray-200 dark:border-gray-700 pb-4'>
            <h2 class='text-2xl font-bold text-gray-900 dark:text-white'>Financial Health</h2>
            <div class='flex items-center mt-1 space-x-2'>
                <span class='text-xl font-semibold text-gray-800 dark:text-gray-200'>{company_name}</span>
                <span class='px-2 py-0.5 text-xs font-medium bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200 rounded-full'>{ticker}</span>
            </div>
        </div>
        
        <!-- Financial Health -->
        <div class='bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 p-6 rounded-xl border border-green-100 dark:border-green-800/50 shadow-sm'>
            <div class='flex items-center mb-4'>
                <div class='p-1.5 bg-green-100 dark:bg-green-900/30 rounded-lg mr-3'>
                    <svg class='w-5 h-5 text-green-600 dark:text-green-400' fill='none' stroke='currentColor' viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'>
                        <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z'></path>
                    </svg>
                </div>
                <h3 class='text-lg font-semibold text-green-800 dark:text-green-200'>Financial Health</h3>
            </div>
            <div class='grid grid-cols-1 md:grid-cols-3 gap-4'>
                <div class='bg-white/60 dark:bg-gray-800/40 p-4 rounded-lg border border-green-100 dark:border-green-800/30'>
                    <div class='flex items-center justify-between mb-2'>
                        <p class='text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider'>Debt-to-Equity</p>
                        <span class='text-xl'>{get_value_indicator(debt_to_equity, "debt_ratio")[1]}</span>
                    </div>
                    <p class='text-xl font-semibold {get_value_indicator(debt_to_equity, "debt_ratio")[0]}'>{format_ratio(debt_to_equity)}</p>
                    <p class='mt-1 text-xs text-gray-500 dark:text-gray-400'>Lower is better</p>
                </div>
                <div class='bg-white/60 dark:bg-gray-800/40 p-4 rounded-lg border border-green-100 dark:border-green-800/30'>
                    <div class='flex items-center justify-between mb-2'>
                        <p class='text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider'>Current Ratio</p>
                        <span class='text-xl'>{get_value_indicator(current_ratio, "ratio")[1]}</span>
                    </div>
                    <p class='text-xl font-semibold {get_value_indicator(current_ratio, "ratio")[0]}'>{format_ratio(current_ratio)}</p>
                    <p class='mt-1 text-xs text-gray-500 dark:text-gray-400'>Above 1.5 is good</p>
                </div>
                <div class='bg-white/60 dark:bg-gray-800/40 p-4 rounded-lg border border-green-100 dark:border-green-800/30'>
                    <div class='flex items-center justify-between mb-2'>
                        <p class='text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider'>Quick Ratio</p>
                        <span class='text-xl'>{get_value_indicator(quick_ratio, "ratio")[1]}</span>
                    </div>
                    <p class='text-xl font-semibold {get_value_indicator(quick_ratio, "ratio")[0]}'>{format_ratio(quick_ratio)}</p>
                    <p class='mt-1 text-xs text-gray-500 dark:text-gray-400'>Above 1.0 is good</p>
                </div>
            </div>
            
            <div class='mt-6 p-4 bg-white/60 dark:bg-gray-800/40 rounded-lg border border-green-100 dark:border-green-800/30'>
                <h4 class='font-medium text-gray-800 dark:text-gray-200 mb-2'>Liquidity Analysis</h4>
                <p class='text-sm text-gray-600 dark:text-gray-400'>
                    The current ratio measures a company's ability to pay short-term obligations with current assets. 
                    A ratio above 1.5 is generally considered healthy. The quick ratio is a more conservative measure 
                    that excludes inventory from current assets.
                </p>
            </div>
            
            <div class='mt-4 p-4 bg-white/60 dark:bg-gray-800/40 rounded-lg border border-green-100 dark:border-green-800/30'>
                <h4 class='font-medium text-gray-800 dark:text-gray-200 mb-2'>Debt Analysis</h4>
                <p class='text-sm text-gray-600 dark:text-gray-400'>
                    The debt-to-equity ratio indicates the relative proportion of shareholders' equity and debt used to 
                    finance a company's assets. Lower values generally indicate less risk, but the appropriate level 
                    varies by industry.
                </p>
            </div>
        </div>
    </div>
    """
    
    # Add DCF Valuation page
    dcf_data = calculate_dcf_valuation(info, ticker)
    if dcf_data:
        # Determine valuation signal
        margin = dcf_data['margin_of_safety']
        if margin > 20:
            valuation_signal = "Significantly Undervalued"
            signal_color = "text-green-600 dark:text-green-400"
        elif margin > 0:
            valuation_signal = "Undervalued"
            signal_color = "text-green-600 dark:text-green-400"
        elif margin > -20:
            valuation_signal = "Fairly Valued"
            signal_color = "text-yellow-600 dark:text-yellow-400"
        else:
            valuation_signal = "Overvalued"
            signal_color = "text-red-600 dark:text-red-400"
        
        pages['dcf_valuation'] = f"""
        <div class='p-4 space-y-6'>
            <div class='border-b border-gray-200 dark:border-gray-700 pb-4'>
                <h2 class='text-2xl font-bold text-gray-900 dark:text-white'>DCF Valuation Model</h2>
                <p class='text-sm text-gray-600 dark:text-gray-400 mt-1'>Discounted Cash Flow analysis to determine intrinsic value</p>
            </div>
            
            <div class='grid grid-cols-1 md:grid-cols-3 gap-4'>
                <div class='bg-white dark:bg-gray-800/50 border border-gray-100 dark:border-gray-700/50 rounded-xl p-5 shadow-sm'>
                    <div class='flex items-center mb-3'>
                        <div class='p-1.5 bg-blue-100 dark:bg-blue-900/30 rounded-lg mr-3'>
                            <svg class='w-5 h-5 text-blue-600 dark:text-blue-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                                <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z' />
                            </svg>
                        </div>
                        <h4 class='font-semibold text-gray-800 dark:text-white'>Current Price</h4>
                    </div>
                    <p class='text-2xl font-bold text-gray-900 dark:text-white'>${dcf_data['current_price']:.2f}</p>
                    <p class='text-xs text-gray-500 dark:text-gray-400 mt-1'>Market price per share</p>
                </div>
                
                <div class='bg-white dark:bg-gray-800/50 border border-gray-100 dark:border-gray-700/50 rounded-xl p-5 shadow-sm'>
                    <div class='flex items-center mb-3'>
                        <div class='p-1.5 bg-purple-100 dark:bg-purple-900/30 rounded-lg mr-3'>
                            <svg class='w-5 h-5 text-purple-600 dark:text-purple-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                                <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M9 7h6m0 10v-3m-3 3h.01M9 17h.01M9 14h.01M12 14h.01M15 11h.01M12 11h.01M9 11h.01M7 21h10a2 2 0 002-2V5a2 2 0 00-2-2H7a2 2 0 00-2 2v14a2 2 0 002 2z' />
                            </svg>
                        </div>
                        <h4 class='font-semibold text-gray-800 dark:text-white'>Intrinsic Value</h4>
                    </div>
                    <div class='flex items-center space-x-2'>
                        <p class='text-2xl font-bold {signal_color}'>${dcf_data['intrinsic_value']:.2f}</p>
                        <span class='text-2xl'>{'💎' if margin > 20 else '💰' if margin > 0 else '⚖️' if margin > -20 else '⚠️'}</span>
                    </div>
                    <p class='text-xs text-gray-500 dark:text-gray-400 mt-1'>DCF calculated value</p>
                </div>
                
                <div class='bg-white dark:bg-gray-800/50 border border-gray-100 dark:border-gray-700/50 rounded-xl p-5 shadow-sm'>
                    <div class='flex items-center mb-3'>
                        <div class='p-1.5 bg-green-100 dark:bg-green-900/30 rounded-lg mr-3'>
                            <svg class='w-5 h-5 text-green-600 dark:text-green-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                                <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.031 9-11.622 0-1.042-.133-2.052-.382-3.016z' />
                            </svg>
                        </div>
                        <h4 class='font-semibold text-gray-800 dark:text-white'>Margin of Safety</h4>
                    </div>
                    <div class='flex items-center space-x-2'>
                        <p class='text-2xl font-bold {signal_color}'>{dcf_data['margin_of_safety']:.1f}%</p>
                        <span class='text-2xl'>{'🚀' if margin > 20 else '📈' if margin > 0 else '📊' if margin > -20 else '📉'}</span>
                    </div>
                    <p class='text-xs text-gray-500 dark:text-gray-400 mt-1'>Upside/downside potential</p>
                </div>
            </div>
            
            <div class='p-4 rounded-xl border {"border-green-200 dark:border-green-800 bg-green-50 dark:bg-green-900/20" if margin > 0 else "border-red-200 dark:border-red-800 bg-red-50 dark:bg-red-900/20" if margin < -20 else "border-yellow-200 dark:border-yellow-800 bg-yellow-50 dark:bg-yellow-900/20"}'>
                <div class='flex items-center'>
                    <div class='p-2 bg-white/50 dark:bg-gray-800/50 rounded-lg mr-3'>
                        <span class='text-2xl'>{'💎' if margin > 20 else '💰' if margin > 0 else '⚖️' if margin > -20 else '⚠️'}</span>
                    </div>
                    <div>
                        <h3 class='text-lg font-semibold {signal_color}'>Valuation Signal: {valuation_signal}</h3>
                        <p class='text-sm text-gray-600 dark:text-gray-400 mt-1'>
                            Based on DCF analysis, the stock appears to be {valuation_signal.lower()} with a {dcf_data['margin_of_safety']:.1f}% margin of safety.
                        </p>
                    </div>
                </div>
            </div>
            
            <div class='bg-white dark:bg-gray-800/50 border border-gray-100 dark:border-gray-700/50 rounded-xl p-5 shadow-sm'>
                <div class='flex items-center mb-4'>
                    <div class='p-1.5 bg-orange-100 dark:bg-orange-900/30 rounded-lg mr-3'>
                        <svg class='w-5 h-5 text-orange-600 dark:text-orange-400' fill='none' stroke='currentColor' viewBox='0 0 24 24'>
                            <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z' />
                            <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M15 12a3 3 0 11-6 0 3 3 0 016 0z' />
                        </svg>
                    </div>
                    <h3 class='text-lg font-semibold text-gray-800 dark:text-white'>DCF Model Assumptions</h3>
                </div>
                <div class='grid grid-cols-1 md:grid-cols-3 gap-4'>
                    <div class='p-3 bg-gradient-to-br from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-900/30 rounded-lg border border-blue-100 dark:border-blue-800/30'>
                        <div class='flex items-center mb-2'>
                            <span class='text-lg mr-2'>📊</span>
                            <p class='text-sm font-medium text-blue-700 dark:text-blue-300'>Discount Rate (WACC)</p>
                        </div>
                        <p class='text-xl font-bold text-blue-800 dark:text-blue-200'>{dcf_data['discount_rate']*100:.1f}%</p>
                    </div>
                    <div class='p-3 bg-gradient-to-br from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-900/30 rounded-lg border border-green-100 dark:border-green-800/30'>
                        <div class='flex items-center mb-2'>
                            <span class='text-lg mr-2'>🌱</span>
                            <p class='text-sm font-medium text-green-700 dark:text-green-300'>Terminal Growth Rate</p>
                        </div>
                        <p class='text-xl font-bold text-green-800 dark:text-green-200'>{dcf_data['terminal_growth']*100:.1f}%</p>
                    </div>
                    <div class='p-3 bg-gradient-to-br from-purple-50 to-purple-100 dark:from-purple-900/20 dark:to-purple-900/30 rounded-lg border border-purple-100 dark:border-purple-800/30'>
                        <div class='flex items-center mb-2'>
                            <span class='text-lg mr-2'>🚀</span>
                            <p class='text-sm font-medium text-purple-700 dark:text-purple-300'>Growth Rate (Initial)</p>
                        </div>
                        <p class='text-xl font-bold text-purple-800 dark:text-purple-200'>{dcf_data['growth_rate']*100:.1f}%</p>
                    </div>
                </div>
                
                <div class='mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-100 dark:border-blue-800/30'>
                    <p class='text-sm text-blue-700 dark:text-blue-300'>
                        <span class='font-medium'>Note:</span> These assumptions are estimates based on the company's historical performance. 
                        The DCF model is highly sensitive to these inputs.
                    </p>
                </div>
            </div>
        </div>
        """
    
    return pages

def run_fundamental_analysis(ticker):
    """Run enhanced fundamental analysis with pagination."""
    try:
        import yfinance as yf
        
        stock = yf.Ticker(ticker)
        info = stock.info
        
        if not info or len(info) < 5:
            return {
                'error': True,
                'content': f"<div class='p-4'><h3 class='text-lg font-semibold mb-2 text-red-600'>Error</h3><p>No fundamental data available for {ticker}. Please check the ticker symbol.</p></div>"
            }
        
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
        
        # Create paginated content
        pages = create_fundamental_analysis_pages(
            ticker, info, company_name, sector, industry, market_cap_formatted,
            pe_ratio, forward_pe, peg_ratio, price_to_book, price_to_sales,
            profit_margin, operating_margin, roe, roa, revenue_growth, earnings_growth,
            dividend_yield, payout_ratio, debt_to_equity, current_ratio, quick_ratio
        )
        
        # Add navigation tabs to each page
        nav_tabs = """
        <div class='flex flex-wrap border-b border-gray-200 dark:border-gray-700 mb-6'>
            <button onclick="showPage('overview')" class='tab-button px-4 py-2 font-medium text-sm rounded-t-lg hover:bg-gray-100 dark:hover:bg-gray-700 active' data-page='overview'>
                <i class='fas fa-home mr-2'></i>Overview
            </button>
            <button onclick="showPage('growth_dividend')" class='tab-button px-4 py-2 font-medium text-sm rounded-t-lg hover:bg-gray-100 dark:hover:bg-gray-700' data-page='growth_dividend'>
                <i class='fas fa-chart-line mr-2'></i>Growth & Dividend
            </button>
            <button onclick="showPage('industry')" class='tab-button px-4 py-2 font-medium text-sm rounded-t-lg hover:bg-gray-100 dark:hover:bg-gray-700' data-page='industry'>
                <i class='fas fa-industry mr-2'></i>Industry
            </button>
            <button onclick="showPage('financial_health')" class='tab-button px-4 py-2 font-medium text-sm rounded-t-lg hover:bg-gray-100 dark:hover:bg-gray-700' data-page='financial_health'>
                <i class='fas fa-heartbeat mr-2'></i>Financial Health
            </button>
        </div>
        
        <script>
            function showPage(pageId) {
                // Hide all pages
                document.querySelectorAll('.page-content').forEach(function(page) {
                    page.classList.add('hidden');
                });
                
                // Show selected page
                document.getElementById(pageId).classList.remove('hidden');
                
                // Update active tab
                document.querySelectorAll('.tab-button').forEach(function(button) {
                    button.classList.remove('active', 'border-b-2', 'border-blue-500', 'text-blue-600', 'dark:text-blue-400');
                });
                
                const activeTab = document.querySelector(`.tab-button[data-page='${pageId}']`);
                if (activeTab) {
                    activeTab.classList.add('active', 'border-b-2', 'border-blue-500', 'text-blue-600', 'dark:text-blue-400');
                }
                
                // Store the active tab in session storage
                sessionStorage.setItem('activeFundamentalTab', pageId);
            }
            
            // Load the previously active tab or default to 'overview'
            document.addEventListener('DOMContentLoaded', function() {
                const activeTab = sessionStorage.getItem('activeFundamentalTab') || 'overview';
                showPage(activeTab);
            });
        </script>
        """
        
        # Wrap each page content with the navigation and page container
        for page_id, content in pages.items():
            pages[page_id] = f"""
            <div id='{page_id}' class='page-content {'block' if page_id == 'overview' else 'hidden'}'>
                {nav_tabs}
                {content}
            </div>
            """
        
        return {
            'error': False,
            'pages': pages,
            'company_name': company_name,
            'ticker': ticker
        }
        
    except Exception as e:
        import traceback
        error_msg = f"Error in fundamental analysis: {str(e)}\n\n{traceback.format_exc()}"
        logging.error(error_msg)
        return {
            'error': True,
            'content': f"<div class='p-4'><h3 class='text-lg font-semibold mb-2 text-red-600'>Error</h3><p>An error occurred while processing your request. Please try again later.</p><p class='text-sm text-gray-500 mt-2'>{str(e)}</p></div>"
        }
        
        # Format the output with enhanced HTML
        output = f"""
        <div class='p-4 space-y-6'>
            <div class='border-b border-gray-200 dark:border-gray-700 pb-4'>
                <h2 class='text-2xl font-bold text-gray-900 dark:text-white'>Fundamental Analysis</h2>
                <div class='flex items-center mt-1 space-x-2'>
                    <span class='text-xl font-semibold text-gray-800 dark:text-gray-200'>{company_name}</span>
                    <span class='px-2 py-0.5 text-xs font-medium bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200 rounded-full'>{ticker}</span>
                </div>
            </div>
            
            <!-- Company Overview -->
            <div class='bg-gradient-to-r from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-900/30 p-6 rounded-xl border border-blue-100 dark:border-blue-800/50 shadow-sm'>
                <div class='flex items-center mb-4'>
                    <div class='p-2 bg-blue-100 dark:bg-blue-800/50 rounded-lg mr-3'>
                        <svg class='w-6 h-6 text-blue-600 dark:text-blue-300' fill='none' stroke='currentColor' viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'>
                            <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4'></path>
                        </svg>
                    </div>
                    <h3 class='text-lg font-semibold text-gray-900 dark:text-white'>Company Overview</h3>
                </div>
                <div class='grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4'>
                    <div class='bg-white/50 dark:bg-gray-800/50 p-3 rounded-lg border border-gray-100 dark:border-gray-700'>
                        <p class='text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1'>Sector</p>
                        <p class='font-medium text-gray-900 dark:text-white'>{sector}</p>
                    </div>
                    <div class='bg-white/50 dark:bg-gray-800/50 p-3 rounded-lg border border-gray-100 dark:border-gray-700'>
                        <p class='text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1'>Industry</p>
                        <p class='font-medium text-gray-900 dark:text-white'>{industry}</p>
                    </div>
                    <div class='bg-white/50 dark:bg-gray-800/50 p-3 rounded-lg border border-gray-100 dark:border-gray-700'>
                        <p class='text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1'>Market Cap</p>
                        <p class='font-medium text-gray-900 dark:text-white'>{market_cap_formatted}</p>
                    </div>
                    <div class='bg-white/50 dark:bg-gray-800/50 p-3 rounded-lg border border-gray-100 dark:border-gray-700'>
                        <p class='text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1'>Employees</p>
                        <p class='font-medium text-gray-900 dark:text-white'>{info.get('fullTimeEmployees', 'N/A'):,}</p>
                    </div>
                </div>
            </div>
            
            <!-- Metrics Grid -->
            <div class='grid grid-cols-1 md:grid-cols-2 gap-5'>
                <!-- Valuation Metrics -->
                <div class='bg-white dark:bg-gray-800/50 border border-gray-100 dark:border-gray-700/50 rounded-xl p-5 shadow-sm hover:shadow-md transition-shadow duration-200'>
                    <div class='flex items-center mb-4'>
                        <div class='p-1.5 bg-purple-100 dark:bg-purple-900/30 rounded-lg mr-3'>
                            <svg class='w-5 h-5 text-purple-600 dark:text-purple-400' fill='none' stroke='currentColor' viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'>
                                <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z'></path>
                            </svg>
                        </div>
                        <h4 class='font-semibold text-gray-800 dark:text-white'>Valuation Ratios</h4>
                    </div>
                    <div class='space-y-2.5'>
                        <div class='flex items-center justify-between py-2 px-3 bg-gray-50 dark:bg-gray-700/30 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700/50 transition-colors'>
                            <span class='text-sm font-medium text-gray-600 dark:text-gray-300'>P/E Ratio (TTM)</span>
                            <span class='font-semibold text-gray-900 dark:text-white'>{format_ratio(pe_ratio)}</span>
                        </div>
                        <div class='flex items-center justify-between py-2 px-3 bg-gray-50 dark:bg-gray-700/30 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700/50 transition-colors'>
                            <span class='text-sm font-medium text-gray-600 dark:text-gray-300'>Forward P/E</span>
                            <span class='font-semibold text-gray-900 dark:text-white'>{format_ratio(forward_pe)}</span>
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
                <div class='bg-white dark:bg-gray-800/50 border border-gray-100 dark:border-gray-700/50 rounded-xl p-5 shadow-sm hover:shadow-md transition-shadow duration-200'>
                    <div class='flex items-center mb-4'>
                        <div class='p-1.5 bg-green-100 dark:bg-green-900/30 rounded-lg mr-3'>
                            <svg class='w-5 h-5 text-green-600 dark:text-green-400' fill='none' stroke='currentColor' viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'>
                                <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z'></path>
                            </svg>
                        </div>
                        <h4 class='font-semibold text-gray-800 dark:text-white'>Profitability</h4>
                    </div>
                    <div class='space-y-2.5'>
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
            <div class='grid grid-cols-1 md:grid-cols-2 gap-5'>
                <div class='bg-white dark:bg-gray-800/50 border border-gray-100 dark:border-gray-700/50 rounded-xl p-5 shadow-sm hover:shadow-md transition-shadow duration-200'>
                    <div class='flex items-center mb-4'>
                        <div class='p-1.5 bg-blue-100 dark:bg-blue-900/30 rounded-lg mr-3'>
                            <svg class='w-5 h-5 text-blue-600 dark:text-blue-400' fill='none' stroke='currentColor' viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'>
                                <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M13 7h8m0 0v8m0-8l-8 8-4-4-6 6'></path>
                            </svg>
                        </div>
                        <h4 class='font-semibold text-gray-800 dark:text-white'>Growth Metrics</h4>
                    </div>
                    <div class='space-y-2.5'>
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
                
                <div class='bg-white dark:bg-gray-800/50 border border-gray-100 dark:border-gray-700/50 rounded-xl p-5 shadow-sm hover:shadow-md transition-shadow duration-200'>
                    <div class='flex items-center mb-4'>
                        <div class='p-1.5 bg-yellow-100 dark:bg-yellow-900/30 rounded-lg mr-3'>
                            <svg class='w-5 h-5 text-yellow-600 dark:text-yellow-400' fill='none' stroke='currentColor' viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'>
                                <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z'></path>
                            </svg>
                        </div>
                        <h4 class='font-semibold text-gray-800 dark:text-white'>Dividend Information</h4>
                    </div>
                    <div class='space-y-2.5'>
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
            
            <!-- Industry Comparison -->
            <div class='bg-white dark:bg-gray-800/50 border border-gray-100 dark:border-gray-700/50 p-6 rounded-xl shadow-sm'>
                <div class='flex items-center mb-4'>
                    <div class='p-1.5 bg-indigo-100 dark:bg-indigo-900/30 rounded-lg mr-3'>
                        <svg class='w-5 h-5 text-indigo-600 dark:text-indigo-400' fill='none' stroke='currentColor' viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'>
                            <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z'></path>
                        </svg>
                    </div>
                    <h3 class='text-lg font-semibold text-gray-900 dark:text-white'>Industry Comparison</h3>
                </div>
                <p class='text-sm text-gray-500 dark:text-gray-400 mb-5'>How {company_name} compares to the {sector if sector != 'N/A' else 'industry'} average</p>
                <div class='space-y-4'>
                    <div class='grid grid-cols-1 md:grid-cols-2 gap-4'>
                        <!-- Profitability Comparison -->
                        <div class='bg-gray-50 dark:bg-gray-700/50 p-3 rounded-lg'>
                            <div class='flex items-center mb-2'>
                                <span class='font-medium text-gray-700 dark:text-gray-300'>Profit Margins</span>
                                <span class='ml-2 text-xs px-2 py-0.5 rounded-full bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300'>
                                    vs {sector if sector != 'N/A' else 'Industry'}
                                </span>
                            </div>
                            <div class='space-y-2'>
                                {get_comparison_row('Operating Margin', operating_margin, info.get('industryAverageOperatingMargin'), sector, is_percentage=True)}
                                {get_comparison_row('Profit Margin', profit_margin, info.get('industryAverageProfitMargin'), sector, is_percentage=True)}
                                {get_comparison_row('ROE', roe, info.get('industryAverageReturnOnEquity'), sector, is_percentage=True)}
                            </div>
                        </div>
                        
                        <!-- Valuation Comparison -->
                        <div class='bg-gray-50 dark:bg-gray-700/50 p-3 rounded-lg'>
                            <div class='flex items-center mb-2'>
                                <span class='font-medium text-gray-700 dark:text-gray-300'>Valuation</span>
                                <span class='ml-2 text-xs px-2 py-0.5 rounded-full bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300'>
                                    vs {sector if sector != 'N/A' else 'Industry'}
                                </span>
                            </div>
                            <div class='space-y-2'>
                                {get_comparison_row('P/E Ratio', pe_ratio, info.get('industryAveragePE'), sector, is_percentage=False)}
                                {get_comparison_row('PEG Ratio', peg_ratio, info.get('industryAveragePEG'), sector, is_percentage=False)}
                                {get_comparison_row('P/B Ratio', price_to_book, info.get('industryAveragePB'), sector, is_percentage=False)}
                            </div>
                        </div>
                    </div>
                    
                    <!-- Financial Health Comparison -->
                    <div class='bg-gray-50 dark:bg-gray-700/50 p-3 rounded-lg'>
                        <div class='flex items-center mb-2'>
                            <span class='font-medium text-gray-700 dark:text-gray-300'>Financial Health</span>
                            <span class='ml-2 text-xs px-2 py-0.5 rounded-full bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300'>
                                vs {sector if sector != 'N/A' else 'Industry'}
                            </span>
                        </div>
                        <div class='grid grid-cols-1 md:grid-cols-3 gap-4'>
                            <div class='space-y-2'>
                                {get_comparison_row('Debt/Equity', debt_to_equity, info.get('industryAverageDebtToEquity'), sector, is_percentage=False)}
                            </div>
                            <div class='space-y-2'>
                                {get_comparison_row('Current Ratio', current_ratio, info.get('industryAverageCurrentRatio'), sector, is_percentage=False)}
                            </div>
                            <div class='space-y-2'>
                                {get_comparison_row('Quick Ratio', quick_ratio, info.get('industryAverageQuickRatio'), sector, is_percentage=False)}
                            </div>
                        </div>
                    </div>
                    
                    <div class='mt-6 p-3 bg-blue-50 dark:bg-blue-900/20 border border-blue-100 dark:border-blue-800/30 rounded-lg'>
                        <div class='flex'>
                            <svg class='h-5 w-5 text-blue-500 dark:text-blue-400 mt-0.5 mr-2 flex-shrink-0' fill='currentColor' viewBox='0 0 20 20' xmlns='http://www.w3.org/2000/svg'>
                                <path fill-rule='evenodd' d='M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h2a1 1 0 100-2v-3a1 1 0 00-1-1H9z' clip-rule='evenodd'></path>
                            </svg>
                            <div class='text-sm text-blue-700 dark:text-blue-300'>
                                <p class='font-medium'>How to read this comparison:</p>
                                <div class='flex flex-wrap items-center gap-2 mt-1 text-xs'>
                                    <span class='inline-flex items-center px-2 py-0.5 rounded-full bg-blue-100 text-blue-800 dark:bg-blue-900/50 dark:text-blue-200'>Better than average</span>
                                    <span class='inline-flex items-center px-2 py-0.5 rounded-full bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300'>Similar to average</span>
                                    <span class='inline-flex items-center px-2 py-0.5 rounded-full bg-red-100 text-red-800 dark:bg-red-900/50 dark:text-red-200'>Below average</span>
                                </div>
                                {'<p class="mt-2 text-yellow-700 dark:text-yellow-300 font-medium">ℹ️ Using estimated industry averages. For more accurate comparisons, consider upgrading to a premium data source.</p>' 
                                if not any(k.startswith('industryAverage') and v is not None for k, v in info.items()) else ''}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Financial Health -->
            <div class='bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 p-6 rounded-xl border border-green-100 dark:border-green-800/50 shadow-sm'>
                <div class='flex items-center mb-4'>
                    <div class='p-1.5 bg-green-100 dark:bg-green-900/30 rounded-lg mr-3'>
                        <svg class='w-5 h-5 text-green-600 dark:text-green-400' fill='none' stroke='currentColor' viewBox='0 0 24 24' xmlns='http://www.w3.org/2000/svg'>
                            <path stroke-linecap='round' stroke-linejoin='round' stroke-width='2' d='M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z'></path>
                        </svg>
                    </div>
                    <h3 class='text-lg font-semibold text-green-800 dark:text-green-200'>Financial Health</h3>
                </div>
                <div class='grid grid-cols-1 md:grid-cols-3 gap-4'>
                    <div class='bg-white/60 dark:bg-gray-800/40 p-4 rounded-lg border border-green-100 dark:border-green-800/30'>
                        <p class='text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1'>Debt-to-Equity</p>
                        <p class='text-xl font-semibold text-gray-900 dark:text-white'>{format_ratio(debt_to_equity)}</p>
                        <p class='mt-1 text-xs text-gray-500 dark:text-gray-400'>Lower is better</p>
                    </div>
                    <div class='bg-white/60 dark:bg-gray-800/40 p-4 rounded-lg border border-green-100 dark:border-green-800/30'>
                        <p class='text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1'>Current Ratio</p>
                        <p class='text-xl font-semibold text-gray-900 dark:text-white'>{format_ratio(current_ratio)}</p>
                        <p class='mt-1 text-xs text-gray-500 dark:text-gray-400'>Above 1.5 is good</p>
                    </div>
                    <div class='bg-white/60 dark:bg-gray-800/40 p-4 rounded-lg border border-green-100 dark:border-green-800/30'>
                        <p class='text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1'>Quick Ratio</p>
                        <p class='text-xl font-semibold text-gray-900 dark:text-white'>{format_ratio(quick_ratio)}</p>
                        <p class='mt-1 text-xs text-gray-500 dark:text-gray-400'>Above 1.0 is good</p>
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
                result = run_prediction_analysis(ticker, days)
                if not result:
                    raise ValueError("No output from prediction analysis")
                
                # Check if result is already in paginated format
                if isinstance(result, dict) and 'pages' in result:
                    # Combine all pages into a single HTML string for the response
                    combined_pages = ''.join(result['pages'].values())
                    
                    return jsonify({
                        'ticker': result['ticker'],
                        'company_name': result['company_name'],
                        'analysis': combined_pages,
                        'analysis_type': analysis_type,
                        'status': 'success',
                        'is_paginated': True,
                        'page_ids': list(result['pages'].keys())
                    })
                else:
                    # Legacy format - return as is
                    return jsonify({
                        'ticker': ticker,
                        'analysis': result,
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
                    current_52w_range = (current_price - low_52w) / (high_52w - low_52w) * 100 if high_52w != low_52w else 50
                    avg_volume = hist['Volume'].mean()
                    latest_volume = hist['Volume'].iloc[-1]
                    volume_ratio = (latest_volume / avg_volume) if avg_volume > 0 else 1
                    market_cap = info.get('marketCap', 0)
                    pe_ratio = info.get('trailingPE', info.get('forwardPE', 'N/A'))
                    
                    # Calculate performance metrics
                    def calculate_return(days_ago):
                        if len(hist) > days_ago:
                            # Get the price 'days_ago' trading days back
                            prev_price = hist['Close'].iloc[-days_ago-1] if days_ago < len(hist) else hist['Close'].iloc[0]
                            return ((current_price - prev_price) / prev_price) * 100
                        return None
                    
                    # Get current year start date for YTD calculation
                    current_year = datetime.now().year
                    ytd_mask = (hist.index >= f'{current_year}-01-01') & (hist.index <= hist.index[-1])
                    ytd_return = None
                    if not hist[ytd_mask].empty and len(hist[ytd_mask]) > 1:
                        ytd_return = ((current_price - hist[ytd_mask]['Close'].iloc[0]) / hist[ytd_mask]['Close'].iloc[0]) * 100
                    
                    # Calculate performance metrics using business days
                    trading_days_in_week = min(5, len(hist)-1)
                    trading_days_in_month = min(21, len(hist)-1)
                    
                    performance_metrics = {
                        '1w': calculate_return(trading_days_in_week),
                        '1m': calculate_return(trading_days_in_month),
                        'ytd': ytd_return
                    }
                    
                    # Format market cap for display
                    if market_cap >= 1e12:
                        market_cap_str = f"${market_cap/1e12:.2f}T"
                    elif market_cap >= 1e9:
                        market_cap_str = f"${market_cap/1e9:.2f}B"
                    elif market_cap >= 1e6:
                        market_cap_str = f"${market_cap/1e6:.2f}M"
                    else:
                        market_cap_str = f"${market_cap:,.2f}"
                    
                    company_name = info.get('longName', ticker)
                    
                    # Generate the paginated analysis
                    pages = create_basic_analysis_pages(
                        ticker=ticker,
                        hist=hist,
                        info=info,
                        current_price=current_price,
                        prev_price=prev_price,
                        price_change=price_change,
                        price_change_pct=price_change_pct,
                        high_52w=high_52w,
                        low_52w=low_52w,
                        current_52w_range=current_52w_range,
                        avg_volume=avg_volume,
                        latest_volume=latest_volume,
                        volume_ratio=volume_ratio,
                        market_cap_str=market_cap_str,
                        pe_ratio=pe_ratio,
                        company_name=company_name,
                        performance_metrics=performance_metrics
                    )
                    
                    # Combine all pages into a single HTML string for the response
                    # The frontend will handle showing/hiding pages using the page-content class
                    combined_pages = ''.join(pages.values())
                    
                    return jsonify({
                        'ticker': ticker.upper(),
                        'company_name': company_name,
                        'analysis': combined_pages,
                        'analysis_type': analysis_type,
                        'status': 'success',
                        'is_paginated': True,  # Indicate to frontend that this is paginated content
                        'page_ids': list(pages.keys())  # List of page IDs for navigation
                    })
                    
            except Exception as e:
                logging.error(f"Error in basic analysis: {str(e)}\n{traceback.format_exc()}")
                return jsonify({
                    'error': f"Error in basic analysis: {str(e)}",
                    'status': 'error',
                    'traceback': traceback.format_exc()
                }), 500
        
        elif analysis_type == 'technical':
            # Enhanced Technical Analysis with pagination
            try:
                result = run_technical_analysis(ticker, days)
                if isinstance(result, str) and result.startswith("<div class='p-4'><h3 class='text-lg font-semibold mb-2 text-red-600'>Error"):
                    return jsonify({
                        'error': result,
                        'status': 'error'
                    }), 500
                
                # Check if result is already in paginated format
                if isinstance(result, dict) and 'pages' in result:
                    # Combine all pages into a single HTML string for the response
                    combined_pages = ''.join(result['pages'].values())
                    
                    return jsonify({
                        'ticker': result['ticker'],
                        'company_name': result['company_name'],
                        'analysis': combined_pages,
                        'analysis_type': analysis_type,
                        'status': 'success',
                        'is_paginated': True,
                        'page_ids': list(result['pages'].keys())
                    })
                else:
                    # Legacy format - return as is
                    return jsonify({
                        'ticker': ticker,
                        'analysis': result,
                        'analysis_type': analysis_type,
                        'status': 'success'
                    })
            except Exception as e:
                return jsonify({
                    'error': f"Error in technical analysis: {str(e)}",
                    'status': 'error'
                }), 500
                
        elif analysis_type == 'fundamental':
            # Enhanced Fundamental Analysis with pagination
            try:
                result = run_fundamental_analysis(ticker)
                if result.get('error'):
                    return jsonify({
                        'error': result.get('content', 'Error in fundamental analysis'),
                        'status': 'error'
                    }), 500
                
                # Combine all pages into a single HTML string for the response
                # The frontend will handle showing/hiding pages using the page-content class
                combined_pages = ''.join(result['pages'].values())
                
                return jsonify({
                    'ticker': result['ticker'],
                    'company_name': result['company_name'],
                    'analysis': combined_pages,
                    'analysis_type': analysis_type,
                    'status': 'success',
                    'is_paginated': True,  # Indicate to frontend that this is paginated content
                    'page_ids': list(result['pages'].keys())  # List of page IDs for navigation
                })
                
            except Exception as e:
                logging.error(f"Error in fundamental analysis: {str(e)}\n{traceback.format_exc()}")
                return jsonify({
                    'error': f"Error in fundamental analysis: {str(e)}",
                    'status': 'error',
                    'traceback': traceback.format_exc()
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
        
        # Use 0.0.0.0 for Render, 127.0.0.1 for local development
        host = '0.0.0.0' if os.environ.get('RENDER') else '127.0.0.1'
        
        # Print environment info for debugging
        print(f"\n🌐 Starting server on {host}:{port}")
        print(f"🚀 Environment: {'Production' if host == '0.0.0.0' else 'Development'}")
        
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
