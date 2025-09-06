#!/usr/bin/env python3
"""
Stock Analysis v2
A Python-based, interactive and CLI-capable stock analysis tool built around Yahoo Finance data.
"""

import argparse
import os
import sys
import tempfile
import subprocess
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf
from datetime import datetime, timedelta
from scipy import stats


class StockAnalyzer:
    """Main stock analysis class with interactive and CLI capabilities."""
    
    def __init__(self, ticker: str = "AAPL", period: str = "1y"):
        self.ticker = ticker.upper()
        self.period = period
        self.data = None
        self.stock_info = None
        self.project_root = Path(__file__).parent.parent
        self.plots_dir = self.project_root / "data" / "plots"
        
    def fetch_data(self) -> bool:
        """Fetch stock data from Yahoo Finance."""
        try:
            stock = yf.Ticker(self.ticker)
            self.data = stock.history(period=self.period)
            self.stock_info = stock.info
            
            if self.data.empty:
                print(f"No data found for ticker {self.ticker}")
                return False
                
            # Normalize MultiIndex columns if present
            if isinstance(self.data.columns, pd.MultiIndex):
                self.data.columns = self.data.columns.droplevel(1)
                
            return True
        except Exception as e:
            print(f"Error fetching data for {self.ticker}: {e}")
            return False
    
    def get_company_info(self) -> Dict[str, Any]:
        """Get company fundamental information."""
        if not self.stock_info:
            return {}
            
        info = {}
        fields = {
            'longName': 'Company Name',
            'sector': 'Sector',
            'industry': 'Industry',
            'currentPrice': 'Current Price',
            'marketCap': 'Market Cap',
            'fiftyTwoWeekHigh': '52W High',
            'fiftyTwoWeekLow': '52W Low',
            'trailingPE': 'P/E Ratio',
            'dividendYield': 'Dividend Yield'
        }
        
        for key, label in fields.items():
            value = self.stock_info.get(key, 'N/A')
            if key == 'marketCap' and value != 'N/A':
                value = f"${value:,.0f}"
            elif key in ['currentPrice', 'fiftyTwoWeekHigh', 'fiftyTwoWeekLow'] and value != 'N/A':
                value = f"${value:.2f}"
            elif key == 'dividendYield' and value != 'N/A':
                value = f"{value*100:.2f}%" if value else 'N/A'
            info[label] = value
            
        return info
    
    def calculate_moving_averages(self) -> pd.DataFrame:
        """Calculate moving averages for the stock data."""
        if self.data is None or self.data.empty:
            return pd.DataFrame()
            
        df = self.data.copy()
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()
        return df
    
    def calculate_rsi(self, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        if self.data is None or self.data.empty:
            return pd.Series()
            
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator."""
        if self.data is None or self.data.empty:
            return pd.Series(), pd.Series(), pd.Series()
            
        exp1 = self.data['Close'].ewm(span=fast).mean()
        exp2 = self.data['Close'].ewm(span=slow).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    def calculate_daily_returns(self) -> pd.Series:
        """Calculate daily returns."""
        if self.data is None or self.data.empty:
            return pd.Series()
        return self.data['Close'].pct_change() * 100
    
    def calculate_bollinger_bands(self, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        if self.data is None or self.data.empty:
            return pd.Series(), pd.Series(), pd.Series()
            
        sma = self.data['Close'].rolling(window=window).mean()
        std = self.data['Close'].rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, sma, lower_band
    
    def calculate_volume_ma(self, window: int = 20) -> pd.Series:
        """Calculate volume moving average."""
        if self.data is None or self.data.empty:
            return pd.Series()
        return self.data['Volume'].rolling(window=window).mean()
    
    def generate_signals(self) -> pd.DataFrame:
        """Generate comprehensive buy/sell signals."""
        if self.data is None or self.data.empty:
            return pd.DataFrame()
            
        df = self.data.copy()
        df['Returns'] = self.calculate_daily_returns()
        
        # Calculate indicators
        df_ma = self.calculate_moving_averages()
        df[['MA20', 'MA50', 'MA200']] = df_ma[['MA20', 'MA50', 'MA200']]
        
        rsi = self.calculate_rsi()
        df['RSI'] = rsi
        
        macd, signal, histogram = self.calculate_macd()
        df['MACD'] = macd
        df['MACD_Signal'] = signal
        df['MACD_Histogram'] = histogram
        
        upper_bb, middle_bb, lower_bb = self.calculate_bollinger_bands()
        df['BB_Upper'] = upper_bb
        df['BB_Middle'] = middle_bb
        df['BB_Lower'] = lower_bb
        
        df['Volume_MA'] = self.calculate_volume_ma()
        
        # Generate signals
        df['Signal'] = 'HOLD'
        df['Signal_Reason'] = ''
        
        # RSI signals
        rsi_oversold = df['RSI'] < 30
        rsi_overbought = df['RSI'] > 70
        
        # MACD signals
        macd_bullish = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))
        macd_bearish = (df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1))
        
        # Moving average crossover signals
        ma_golden_cross = (df['MA20'] > df['MA50']) & (df['MA20'].shift(1) <= df['MA50'].shift(1))
        ma_death_cross = (df['MA20'] < df['MA50']) & (df['MA20'].shift(1) >= df['MA50'].shift(1))
        
        # Bollinger Band signals
        bb_oversold = df['Close'] < df['BB_Lower']
        bb_overbought = df['Close'] > df['BB_Upper']
        
        # Combine signals for BUY
        buy_conditions = (
            (rsi_oversold & macd_bullish) |
            (ma_golden_cross & (df['RSI'] < 50)) |
            (bb_oversold & (df['RSI'] < 40))
        )
        
        # Combine signals for SELL
        sell_conditions = (
            (rsi_overbought & macd_bearish) |
            (ma_death_cross & (df['RSI'] > 50)) |
            (bb_overbought & (df['RSI'] > 60))
        )
        
        # Apply signals
        df.loc[buy_conditions, 'Signal'] = 'BUY'
        df.loc[sell_conditions, 'Signal'] = 'SELL'
        
        # Add signal reasons
        df.loc[rsi_oversold & macd_bullish, 'Signal_Reason'] = 'RSI Oversold + MACD Bullish'
        df.loc[ma_golden_cross & (df['RSI'] < 50), 'Signal_Reason'] = 'Golden Cross + RSI Favorable'
        df.loc[bb_oversold & (df['RSI'] < 40), 'Signal_Reason'] = 'BB Oversold + RSI Oversold'
        df.loc[rsi_overbought & macd_bearish, 'Signal_Reason'] = 'RSI Overbought + MACD Bearish'
        df.loc[ma_death_cross & (df['RSI'] > 50), 'Signal_Reason'] = 'Death Cross + RSI Unfavorable'
        df.loc[bb_overbought & (df['RSI'] > 60), 'Signal_Reason'] = 'BB Overbought + RSI Overbought'
        
        return df
    
    def run_backtest(self, initial_capital: float = 10000) -> Dict[str, Any]:
        """Run backtest simulation on buy/sell signals."""
        df = self.generate_signals()
        if df.empty:
            return {}
            
        capital = initial_capital
        shares = 0
        trades = []
        portfolio_values = []
        
        for i, row in df.iterrows():
            if row['Signal'] == 'BUY' and capital > 0:
                shares_to_buy = capital // row['Close']
                if shares_to_buy > 0:
                    cost = shares_to_buy * row['Close']
                    capital -= cost
                    shares += shares_to_buy
                    trades.append({
                        'date': i,
                        'action': 'BUY',
                        'shares': shares_to_buy,
                        'price': row['Close'],
                        'cost': cost
                    })
            
            elif row['Signal'] == 'SELL' and shares > 0:
                proceeds = shares * row['Close']
                capital += proceeds
                trades.append({
                    'date': i,
                    'action': 'SELL',
                    'shares': shares,
                    'price': row['Close'],
                    'proceeds': proceeds
                })
                shares = 0
            
            # Calculate portfolio value
            portfolio_value = capital + (shares * row['Close'])
            portfolio_values.append(portfolio_value)
        
        # Final portfolio value
        final_value = capital + (shares * df['Close'].iloc[-1])
        total_return = (final_value - initial_capital) / initial_capital * 100
        
        # Calculate buy and hold return for comparison
        buy_hold_return = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100
        
        # Calculate Sharpe ratio
        returns = pd.Series(portfolio_values).pct_change().dropna()
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        # Calculate maximum drawdown
        portfolio_series = pd.Series(portfolio_values)
        rolling_max = portfolio_series.expanding().max()
        drawdown = (portfolio_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min() * 100
        
        return {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'buy_hold_return': buy_hold_return,
            'num_trades': len(trades),
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trades': trades,
            'portfolio_values': portfolio_values
        }
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Calculate comprehensive summary statistics."""
        if self.data is None or self.data.empty:
            return {}
            
        df = self.generate_signals()
        returns = df['Returns'].dropna()
        
        # Basic statistics
        stats = {
            'period_high': df['High'].max(),
            'period_low': df['Low'].min(),
            'average_close': df['Close'].mean(),
            'average_volume': df['Volume'].mean(),
            'total_trading_days': len(df),
            'current_price': df['Close'].iloc[-1],
            
            # Returns analysis
            'avg_daily_return': returns.mean(),
            'volatility': returns.std(),
            'annualized_volatility': returns.std() * np.sqrt(252),
            'best_day': returns.max(),
            'worst_day': returns.min(),
            
            # Up/Down days
            'up_days': (returns > 0).sum(),
            'down_days': (returns < 0).sum(),
            'up_days_pct': (returns > 0).mean() * 100,
            'down_days_pct': (returns < 0).mean() * 100,
            
            # Signal analysis
            'buy_signals': (df['Signal'] == 'BUY').sum(),
            'sell_signals': (df['Signal'] == 'SELL').sum(),
            'hold_days': (df['Signal'] == 'HOLD').sum(),
        }
        
        # Calculate streaks
        stats.update(self._calculate_streaks(returns))
        
        # Calculate Sharpe ratio
        if returns.std() > 0:
            stats['sharpe_ratio'] = (returns.mean() / returns.std()) * np.sqrt(252)
        else:
            stats['sharpe_ratio'] = 0
            
        # High volatility days (> 2 standard deviations)
        volatility_threshold = returns.std() * 2
        high_vol_days = returns[abs(returns) > volatility_threshold]
        stats['high_volatility_days'] = len(high_vol_days)
        stats['high_vol_dates'] = high_vol_days.index.tolist()
        
        return stats
    
    def _calculate_streaks(self, returns: pd.Series) -> Dict[str, int]:
        """Calculate longest up and down streaks."""
        if returns.empty:
            return {'longest_up_streak': 0, 'longest_down_streak': 0}
            
        up_streak = 0
        down_streak = 0
        max_up_streak = 0
        max_down_streak = 0
        
        for ret in returns:
            if ret > 0:
                up_streak += 1
                down_streak = 0
                max_up_streak = max(max_up_streak, up_streak)
            elif ret < 0:
                down_streak += 1
                up_streak = 0
                max_down_streak = max(max_down_streak, down_streak)
            else:
                up_streak = 0
                down_streak = 0
                
        return {
            'longest_up_streak': max_up_streak,
            'longest_down_streak': max_down_streak
        }
    
    def get_technical_indicators(self) -> Dict[str, float]:
        """Get latest technical indicators."""
        if self.data is None or self.data.empty:
            return {}
            
        rsi = self.calculate_rsi()
        macd, signal, histogram = self.calculate_macd()
        returns = self.calculate_daily_returns()
        
        indicators = {}
        if not rsi.empty:
            indicators['RSI (14)'] = rsi.iloc[-1]
        if not macd.empty:
            indicators['MACD'] = macd.iloc[-1]
            indicators['MACD Signal'] = signal.iloc[-1]
            indicators['MACD Histogram'] = histogram.iloc[-1]
        if not returns.empty:
            indicators['Daily Return (%)'] = returns.iloc[-1]
            indicators['Volatility (30d)'] = returns.tail(30).std()
            
        return indicators
    
    def create_price_chart(self, save_path: Optional[str] = None, use_preview: bool = False) -> Optional[str]:
        """Create price history chart with moving averages."""
        if self.data is None or self.data.empty:
            print("No data available for charting")
            return None
            
        df_ma = self.calculate_moving_averages()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot price and moving averages
        ax.plot(df_ma.index, df_ma['Close'], label='Close Price', linewidth=2)
        ax.plot(df_ma.index, df_ma['MA20'], label='MA20', alpha=0.7)
        ax.plot(df_ma.index, df_ma['MA50'], label='MA50', alpha=0.7)
        ax.plot(df_ma.index, df_ma['MA200'], label='MA200', alpha=0.7)
        
        ax.set_title(f'{self.ticker} - Price History with Moving Averages', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        return self._handle_chart_display(fig, save_path, use_preview, f"{self.ticker}_price_history")
    
    def create_candlestick_chart(self, candle_period: str = "1mo", save_path: Optional[str] = None, 
                               use_preview: bool = False) -> Optional[str]:
        """Create candlestick chart with volume."""
        if self.data is None or self.data.empty:
            print("No data available for charting")
            return None
            
        # Fetch data for candlestick period
        try:
            stock = yf.Ticker(self.ticker)
            candle_data = stock.history(period=candle_period)
            
            if candle_data.empty:
                print(f"No data available for candlestick chart with period {candle_period}")
                return None
                
            # Normalize MultiIndex columns if present
            if isinstance(candle_data.columns, pd.MultiIndex):
                candle_data.columns = candle_data.columns.droplevel(1)
            
            # Ensure required columns are numeric
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_cols:
                if col in candle_data.columns:
                    candle_data[col] = pd.to_numeric(candle_data[col], errors='coerce')
            
            # Remove rows with NaN values in OHLC columns
            candle_data = candle_data.dropna(subset=['Open', 'High', 'Low', 'Close'])
            
            if candle_data.empty:
                print("No valid OHLC data available for candlestick chart")
                return None
            
            # Create candlestick chart
            if save_path:
                file_path = save_path
            elif use_preview:
                file_path = tempfile.mktemp(suffix='.png')
            else:
                file_path = None
            
            # Configure mplfinance style
            mc = mpf.make_marketcolors(up='g', down='r', edge='inherit',
                                     wick={'up':'green', 'down':'red'},
                                     volume='in')
            s = mpf.make_mpf_style(marketcolors=mc, gridstyle='-', gridcolor='lightgray')
            
            # Plot candlestick chart
            mpf.plot(candle_data, type='candle', style=s, volume=True,
                    title=f'{self.ticker} - Candlestick Chart ({candle_period})',
                    ylabel='Price ($)', ylabel_lower='Volume',
                    figsize=(12, 10), savefig=file_path if file_path else None)
            
            if use_preview and file_path:
                self._open_with_preview(file_path)
                print(f"(preview opened: {file_path})")
                return file_path
            elif not save_path:
                plt.show()
                
            return file_path
            
        except Exception as e:
            print(f"Error creating candlestick chart: {e}")
            return None
    
    def _handle_chart_display(self, fig, save_path: Optional[str], use_preview: bool, 
                            filename_base: str) -> Optional[str]:
        """Handle chart display logic (save, preview, or show)."""
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved to: {save_path}")
            plt.close(fig)
            return save_path
        elif use_preview:
            temp_path = tempfile.mktemp(suffix='.png')
            fig.savefig(temp_path, dpi=300, bbox_inches='tight')
            self._open_with_preview(temp_path)
            print(f"(preview opened: {temp_path})")
            plt.close(fig)
            return temp_path
        else:
            plt.show()
            return None
    
    def _open_with_preview(self, file_path: str):
        """Open file with macOS Preview."""
        try:
            subprocess.run(['open', '-a', 'Preview', file_path], check=True)
        except subprocess.CalledProcessError:
            print(f"Could not open with Preview. File saved at: {file_path}")
    
    def save_data_to_csv(self, file_path: str) -> bool:
        """Save stock data to CSV file."""
        if self.data is None or self.data.empty:
            print("No data available to save")
            return False
            
        try:
            self.data.to_csv(file_path)
            print(f"Data saved to: {file_path}")
            return True
        except Exception as e:
            print(f"Error saving data: {e}")
            return False
    
    def print_company_info(self):
        """Print company information in a formatted table."""
        info = self.get_company_info()
        if not info:
            print("Company information not available")
            return
            
        print(f"\n{'='*50}")
        print(f"COMPANY INFORMATION - {self.ticker}")
        print(f"{'='*50}")
        
        for label, value in info.items():
            print(f"{label:<20}: {value}")
        print()
    
    def print_technical_indicators(self):
        """Print technical indicators in a formatted table."""
        indicators = self.get_technical_indicators()
        if not indicators:
            print("Technical indicators not available")
            return
            
        print(f"\n{'='*50}")
        print(f"TECHNICAL INDICATORS - {self.ticker}")
        print(f"{'='*50}")
        
        for name, value in indicators.items():
            if isinstance(value, float):
                print(f"{name:<20}: {value:.4f}")
            else:
                print(f"{name:<20}: {value}")
        print()


def interactive_mode():
    """Run the interactive menu-driven interface."""
    print("="*60)
    print("STOCK ANALYSIS v2 - Interactive Mode")
    print("="*60)
    
    # Prompt for ticker symbol at startup
    ticker_input = input("Enter a stock ticker symbol (e.g., AAPL, TSLA, MSFT): ").strip().upper()
    if not ticker_input:
        ticker_input = "AAPL"  # Default fallback
        print(f"No ticker entered, using default: {ticker_input}")
    
    analyzer = StockAnalyzer(ticker=ticker_input)
    
    # Initial data fetch
    print(f"Fetching data for {analyzer.ticker}...")
    if not analyzer.fetch_data():
        ticker = input("Enter a valid ticker symbol: ").strip().upper()
        analyzer.ticker = ticker
        if not analyzer.fetch_data():
            print("Failed to fetch data. Exiting.")
            return
    
    while True:
        print(f"\n{'='*40}")
        print(f"Current Ticker: {analyzer.ticker}")
        print(f"{'='*40}")
        print("1. View Company Information")
        print("2. View Price History (chart)")
        print("3. View Candlestick Chart (chart)")
        print("4. View Technical Indicators (printed table)")
        print("5. Save Data to CSV")
        print("6. Change Ticker")
        print("0. Exit")
        print("-" * 40)
        
        choice = input("Select an option (0-6): ").strip()
        
        if choice == "0":
            print("Goodbye!")
            break
        elif choice == "1":
            analyzer.print_company_info()
        elif choice == "2":
            print("Generating price history chart...")
            analyzer.create_price_chart(use_preview=True)
        elif choice == "3":
            while True:
                period = input("Enter candlestick period (1w/1mo/3mo) [default: 1mo]: ").strip().lower()
                if not period:
                    period = "1mo"
                    break
                elif period in ["1w", "1mo", "3mo"]:
                    break
                else:
                    print("Invalid period. Please enter '1w', '1mo', or '3mo'.")
            print(f"Generating candlestick chart for {period}...")
            analyzer.create_candlestick_chart(candle_period=period, use_preview=True)
        elif choice == "4":
            analyzer.print_technical_indicators()
        elif choice == "5":
            filename = input("Enter CSV filename (e.g., data.csv): ").strip()
            if filename:
                analyzer.save_data_to_csv(filename)
        elif choice == "6":
            new_ticker = input("Enter new ticker symbol: ").strip().upper()
            if new_ticker:
                analyzer.ticker = new_ticker
                print(f"Fetching data for {new_ticker}...")
                if not analyzer.fetch_data():
                    print("Failed to fetch data for new ticker.")
        else:
            print("Invalid option. Please try again.")


def cli_mode(args):
    """Run in CLI mode with provided arguments."""
    analyzer = StockAnalyzer(ticker=args.ticker, period=args.period)
    
    print(f"Fetching data for {analyzer.ticker} (period: {analyzer.period})...")
    if not analyzer.fetch_data():
        print(f"Failed to fetch data for {analyzer.ticker}")
        return
    
    # Ensure plots directory exists if saving
    if args.save:
        analyzer.plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle chart generation
    if args.chart in ['price', 'both']:
        save_path = None
        if args.save:
            save_path = analyzer.plots_dir / f"{analyzer.ticker}_price_history.png"
        
        print("Generating price history chart...")
        analyzer.create_price_chart(save_path=str(save_path) if save_path else None, 
                                  use_preview=args.use_preview)
    
    if args.chart in ['candle', 'both']:
        save_path = None
        if args.save:
            save_path = analyzer.plots_dir / f"{analyzer.ticker}_candlestick_{args.candle_period}.png"
        
        print(f"Generating candlestick chart ({args.candle_period})...")
        analyzer.create_candlestick_chart(candle_period=args.candle_period,
                                        save_path=str(save_path) if save_path else None,
                                        use_preview=args.use_preview)
    
    # Handle indicators
    if args.indicators:
        analyzer.print_technical_indicators()
    
    # Handle CSV export
    if args.save_data:
        analyzer.save_data_to_csv(args.save_data)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Stock Analysis v2 - Interactive and CLI stock analysis tool')
    
    # Basic options
    parser.add_argument('--ticker', default='AAPL', help='Stock ticker symbol (default: AAPL)')
    parser.add_argument('--period', default='1y', 
                       choices=['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'],
                       help='History period for price data (default: 1y)')
    
    # Chart options
    parser.add_argument('--chart', choices=['price', 'candle', 'both'], 
                       help='Which chart(s) to render')
    parser.add_argument('--candle-period', default='1mo', choices=['1w', '1mo', '3mo'],
                       help='Window for candlestick chart (default: 1mo)')
    
    # Action options
    parser.add_argument('--indicators', action='store_true', 
                       help='Print latest technical indicators')
    parser.add_argument('--save-data', metavar='PATH.csv', 
                       help='Save fetched price history to CSV at PATH')
    
    # Display options
    parser.add_argument('--save', action='store_true', 
                       help='Save generated chart images to data/plots/')
    parser.add_argument('--use-preview', action='store_true',
                       help='Display charts via macOS Preview (temporary PNG)')
    parser.add_argument('--non-interactive', action='store_true',
                       help='Run actions and exit')
    
    args = parser.parse_args()
    
    # Determine if we should run in CLI mode
    cli_mode_triggers = [
        args.chart, args.indicators, args.save_data, args.non_interactive
    ]
    
    if any(cli_mode_triggers):
        cli_mode(args)
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
