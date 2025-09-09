from flask import Flask, render_template, request, jsonify, redirect, url_for
import sys
import os
import threading
import time
from market_analyzer import analyze_stock
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

@app.route('/analyze', methods=['POST'])
def analyze():
    ticker = request.form.get('ticker', '').upper()
    days = int(request.form.get('days', 90))
    
    if not ticker:
        return jsonify({'error': 'Please enter a stock ticker'}), 400
    
    try:
        # Redirect stdout to capture the output
        from io import StringIO
        import sys
        
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        # Run the analysis
        analyze_stock(
            ticker=ticker,
            show_history=True,
            history_days=days,
            show_graphs=False,
            skip_plot=True
        )
        
        # Get the output
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        # Process the output (you might want to format this better)
        return jsonify({
            'ticker': ticker,
            'analysis': output,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

def open_browser():
    time.sleep(1)  # Give the server a second to start
    webbrowser.open('http://127.0.0.1:5000')

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Start the browser automatically
    threading.Thread(target=open_browser).start()
    
    # Run the app
    try:
        app.run(debug=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        print("\nServer has been shut down. You can close this window.")
