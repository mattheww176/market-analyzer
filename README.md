# Stock Analysis v2

A Python-based, interactive and CLI-capable stock analysis tool built around Yahoo Finance data.

It supports quick visual analysis (charts shown without saving by default), optional saving, and popular technical indicators.

## Features

- Interactive menu-driven workflow (terminal UI)
- Non-interactive CLI mode with flags for automation
- Price history chart with moving averages (20/50/200)
- Candlestick chart (1w, 1mo, 3mo) with volume
- Technical indicators: RSI, MACD, daily returns
- View company fundamentals (name, sector, industry, price, market cap, 52w high/low, PE, dividend)
- Flexible display behavior:
  - Default: show charts without saving
  - Optional: open charts via macOS Preview (temporary PNG)
  - Optional: save charts to `data/plots/` with `--save`

## Project Structure

```
stock_analysis_v2/
├── data/
│   └── plots/                 # (Created as needed) Saved charts when --save is used
├── notebooks/                 # For your future experiments (empty)
├── src/
│   └── stock_analyzer.py      # Main script (interactive + CLI)
└── requirements.txt
```

## Installation

1) Create a virtual environment (recommended) and install dependencies:

```bash
cd /Users/mattcat1778/CascadeProjects/stock_analysis_v2
python3 -m venv venv
source venv/bin/activate   # On macOS/Linux
# Windows PowerShell: .\venv\Scripts\Activate.ps1

python3 -m pip install -r requirements.txt
```

## Usage

### Interactive Mode (default)

Launch the tool and follow the on-screen prompts.

```bash
python3 src/stock_analyzer.py
```

Menu options:
- 1: View Company Information
- 2: View Price History (chart)
- 3: View Candlestick Chart (chart)
- 4: View Technical Indicators (printed table)
- 5: Save Data to CSV
- 6: Change Ticker
- 0: Exit

Charts are shown on screen by default. The interactive menu is configured to use macOS Preview for reliable display (temporary PNG files are created and opened, then left in your system temp folder).

### Non-Interactive CLI Mode

Add flags to generate outputs without the interactive prompts. When any actionable flags are provided or `--non-interactive` is used, the tool runs once and exits.

Common flags:
- `--ticker TICKER`                Stock ticker (e.g., AAPL, MSFT, NVDA)
- `--period P`                     History period for fetching price data (default: `1y`). One of: `1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max`
- `--chart {price,candle,both}`    Which chart(s) to render
- `--candle-period {1w,1mo,3mo}`   Window for candlestick chart (default: `1mo`)
- `--indicators`                   Print latest technical indicators
- `--save-data PATH.csv`           Save fetched price history to CSV at PATH
- `--save`                         Save generated chart images to `data/plots/` (without this, charts are shown but not saved)
- `--use-preview`                  Display charts via macOS Preview (temporary PNG). Recommended for reliability inside IDEs
- `--non-interactive`              Run actions and exit

Examples:

- Show price chart for AAPL (Preview, no saving):
```bash
python3 src/stock_analyzer.py --ticker AAPL --chart price --use-preview --non-interactive
```

- Show candlestick chart for NVDA last month (Preview, no saving):
```bash
python3 src/stock_analyzer.py --ticker NVDA --chart candle --candle-period 1mo --use-preview --non-interactive
```

- Generate both charts and save them to `data/plots/`:
```bash
python3 src/stock_analyzer.py --ticker MSFT --period 6mo --chart both --save --non-interactive
```

- Print indicators only:
```bash
python3 src/stock_analyzer.py --ticker GOOGL --indicators --non-interactive
```

- Save price history to CSV:
```bash
python3 src/stock_analyzer.py --ticker NVDA --save-data nvda_history.csv --non-interactive
```

## Notes on Charts & Display

- By default, charts are displayed and not saved.
- In interactive mode, the app prefers Preview-based display on macOS for reliability. Temporary PNGs are created and opened. The script prints the temporary file path when opened.
- If Preview cannot open the file (very rare), the script falls back to a matplotlib window.
- Use `--save` to write persistent chart images to `data/plots/`.

## Data Details

- Data is sourced via the `yfinance` library (Yahoo Finance). As of recent updates, `yfinance.download` may return MultiIndex columns (e.g., first level = Price, second = Ticker). The script normalizes these columns for candlestick plotting.
- Candlestick charts explicitly require numeric `Open, High, Low, Close, Volume` data. The script coerces and validates these before plotting.

## Troubleshooting

- No chart window appears:
  - Try using the Preview-based path in CLI: add `--use-preview`.
  - In interactive mode, the script already prefers Preview; check the terminal output for `(preview opened: /path/to/tmp.png)` and open that path manually if needed.
  - Ensure you’re running in a local GUI session (not an SSH/headless environment).

- Candlestick error about missing/invalid OHLC columns:
  - This can occur if data is sparse for the selected period. Try a longer period, e.g., `--candle-period 3mo`.
  - The script already flattens MultiIndex columns and coerces numeric types.

- Module not found (e.g., yfinance):
  - Ensure the virtual environment is active and dependencies installed:
    ```bash
    source venv/bin/activate
    python3 -m pip install -r requirements.txt
    ```

## License

MIT (provide a LICENSE file if distributing).
