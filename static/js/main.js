// Global watchlist variable
let watchlist;

// Main application entry point
document.addEventListener('DOMContentLoaded', function() {
    // Initialize components
    initChart();
    
    // Initialize watchlist
    watchlist = new Watchlist();
    window.watchlist = watchlist; // Make globally accessible
    
    // Setup event listeners
    setupEventListeners();
    
    // Setup watchlist event listeners
    watchlist.setupEventListeners();
    
    // Setup other components
    setupTickerInput();
    setupCustomDaysToggle();
    setupCollapsibleSections();
});

// Setup all event listeners
function setupEventListeners() {
    // Tab switching
    document.querySelectorAll('.tab-button').forEach(button => {
        button.addEventListener('click', function() {
            const tab = this.getAttribute('data-tab');
            document.querySelectorAll('.tab-button').forEach(btn => {
                btn.classList.remove('active', 'border-blue-500', 'text-blue-600', 'dark:text-blue-400');
                btn.classList.add('border-transparent', 'text-gray-500', 'hover:text-gray-700', 'hover:border-gray-300', 'dark:text-gray-400', 'dark:hover:text-gray-200');
            });
            this.classList.add('active', 'border-blue-500', 'text-blue-600', 'dark:text-blue-400');
            this.classList.remove('border-transparent', 'text-gray-500', 'hover:text-gray-700', 'hover:border-gray-300', 'dark:text-gray-400', 'dark:hover:text-gray-200');
            
            updateChartForTab(tab);
        });
    });
    
    // Form submission
    const form = document.getElementById('analysisForm');
    if (form) {
        form.addEventListener('submit', handleFormSubmit);
    }
    
    // Fullscreen toggle
    const fullscreenBtn = document.getElementById('fullscreenBtn');
    if (fullscreenBtn) {
        fullscreenBtn.addEventListener('click', toggleFullscreen);
    }
    
    // Reset zoom
    const resetZoomBtn = document.getElementById('resetZoomBtn');
    if (resetZoomBtn) {
        resetZoomBtn.addEventListener('click', resetChartZoom);
    }
}

// Handle form submission
async function handleFormSubmit(e) {
    e.preventDefault();
    e.stopPropagation();
    
    // Get form elements
    const form = e.target;
    const ticker = form.ticker.value.trim().toUpperCase();
    const days = form.days.value;
    const analysisType = form.analysisType.value;
    
    // UI elements
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const error = document.getElementById('error');
    const errorMessage = document.getElementById('errorMessage');
    const submitButton = form.querySelector('button[type="submit"]');
    
    // Reset UI
    if (error) error.classList.add('hidden');
    if (loading) loading.classList.remove('hidden');
    if (results) results.classList.add('hidden');
    
    // Disable submit button
    if (submitButton) {
        submitButton.disabled = true;
        submitButton.innerHTML = `
            <i class="fas fa-spinner fa-spin mr-2"></i>
            Analyzing...
        `;
    }
    
    try {
        console.log('Starting analysis for:', ticker);
        
        // Basic ticker validation
        if (!ticker) {
            throw new Error('Please enter a stock ticker');
        }
        
        // Show loading state
        const loadingStatus = document.getElementById('loadingStatus');
        if (loadingStatus) {
            loadingStatus.textContent = 'Fetching data...';
        }
        
        // Fetch both chart data and analysis in parallel
        const [chartResponse, analysisResponse] = await Promise.all([
            fetch(`/api/chart-data/${ticker}?days=${days}`),
            fetch('/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                body: `ticker=${ticker}&days=${days}&analysis_type=${analysisType}`
            })
        ]);
        
        if (!chartResponse.ok) {
            const errorData = await chartResponse.json().catch(() => ({}));
            throw new Error(errorData.error || `Failed to fetch chart data (${chartResponse.status})`);
        }
        
        const chartDataResponse = await chartResponse.json();
        const analysisData = await analysisResponse.json();
        
        console.log('Chart data response:', chartDataResponse);
        console.log('Analysis data response:', analysisData);
        
        // Store chart data globally for tab switching
        chartData = chartDataResponse;
        
        // Update chart and stock info
        updateChart(chartDataResponse, 'price');
        updateStockInfo(ticker, chartDataResponse);
        
        // Show/hide technical analysis section based on analysis type
        const technicalAnalysisSection = document.getElementById('technicalAnalysisSection');
        if (technicalAnalysisSection) {
            // Show technical analysis for both 'technical' and 'full' analysis types
            if (analysisType === 'technical' || analysisType === 'full') {
                technicalAnalysisSection.style.display = 'block';
                updateTechnicalAnalysisData(chartDataResponse, ticker);
            } else {
                technicalAnalysisSection.style.display = 'none';
            }
        }
        
        // Update analysis text based on analysis type
        const analysisText = document.getElementById('analysisText');
        const analysisContainer = analysisText?.parentElement;
        
        if (analysisText && analysisContainer) {
            if (analysisType === 'basic') {
                // For basic analysis, show simple summary with key metrics
                analysisContainer.style.display = 'block';
                
                // Generate basic analysis content
                let basicAnalysisContent = generateBasicAnalysis(ticker, chartDataResponse);
                analysisText.innerHTML = basicAnalysisContent;
            } else if (analysisType === 'technical') {
                // For technical analysis, hide the analysis text section entirely
                analysisContainer.style.display = 'none';
            } else if (analysisType === 'full' || analysisType === 'prediction') {
                // For full or prediction analysis, show comprehensive analysis
                analysisContainer.style.display = 'block';
                if (analysisData && analysisData.analysis) {
                    // Format the analysis text with line breaks and proper spacing
                    const formattedAnalysis = analysisData.analysis
                        .replace(/\n\n+/g, '<br><br>')  // Double newlines to paragraph breaks
                        .replace(/\n/g, '<br>')       // Single newlines to line breaks
                        .replace(/📈/g, '📈 ')
                        .replace(/🔮/g, '🔮 ')
                        .replace(/📊/g, '📊 ')
                        .replace(/🎯/g, '🎯 ');
                    
                    analysisText.innerHTML = `
                        <div class="bg-white dark:bg-gray-800 rounded-lg p-4 shadow">
                            <div class="analysis-results prose dark:prose-invert max-w-none">
                                ${formattedAnalysis}
                            </div>
                        </div>`;
                } else if (analysisData && analysisData.error) {
                    analysisText.innerHTML = `
                        <div class="bg-red-50 dark:bg-red-900/20 border-l-4 border-red-500 p-4 rounded">
                            <div class="flex">
                                <div class="flex-shrink-0">
                                    <i class="fas fa-exclamation-triangle text-red-500"></i>
                                </div>
                                <div class="ml-3">
                                    <p class="text-sm text-red-700 dark:text-red-300">
                                        Analysis Error: ${analysisData.error}
                                    </p>
                                </div>
                            </div>
                        </div>`;
                } else {
                    analysisText.innerHTML = `
                        <div class="bg-gray-50 dark:bg-gray-800 rounded-lg p-4 text-center">
                            <p class="text-gray-600 dark:text-gray-400">
                                Analysis complete. Review the results above.
                            </p>
                        </div>`;
                }
            }
        }
        
        // Show results
        if (results) results.classList.remove('hidden');
        
    } catch (err) {
        console.error('Analysis error:', err);
        
        // Update error message
        if (errorMessage) {
            errorMessage.textContent = err.message || 'An unexpected error occurred';
            if (error) error.classList.remove('hidden');
        }
        
    } finally {
        // Reset UI
        if (loading) loading.classList.add('hidden');
        
        if (submitButton) {
            submitButton.disabled = false;
            submitButton.innerHTML = `
                <i class="fas fa-chart-line mr-2"></i>
                Analyze Stock
            `;
        }
        
        // Scroll to results or error
        const elementToScroll = error && !error.classList.contains('hidden') ? error : results;
        if (elementToScroll) {
            elementToScroll.scrollIntoView({ behavior: 'smooth' });
        }
    }
}

// Update stock info in the UI
function updateStockInfo(ticker, chartData) {
    // Update stock ticker and name at the top
    const stockTicker = document.getElementById('stockTicker');
    const companyName = document.getElementById('companyName');
    
    if (stockTicker) stockTicker.textContent = ticker;
    if (companyName) companyName.textContent = `${ticker} - Stock Analysis`;
    
    // Update price metrics if data is available
    if (chartData.prices && chartData.prices.length > 0) {
        const currentPrice = chartData.prices[chartData.prices.length - 1];
        const previousPrice = chartData.prices.length > 1 ? chartData.prices[chartData.prices.length - 2] : currentPrice;
        const priceChange = currentPrice - previousPrice;
        const percentChange = previousPrice !== 0 ? (priceChange / previousPrice) * 100 : 0;
        
        // Update current price
        const currentPriceEl = document.getElementById('currentPrice');
        const stockSymbol = document.getElementById('stockSymbol');
        const stockName = document.getElementById('stockName');
        
        if (currentPriceEl) currentPriceEl.textContent = `$${currentPrice.toFixed(2)}`;
        if (stockSymbol) stockSymbol.textContent = ticker;
        if (stockName) stockName.textContent = `${ticker} Stock`;
        
        // Update price change
        const priceChangeEl = document.getElementById('priceChange');
        const percentChangeEl = document.getElementById('percentChange');
        
        if (priceChangeEl) {
            priceChangeEl.textContent = `${priceChange >= 0 ? '+' : ''}$${Math.abs(priceChange).toFixed(2)}`;
            priceChangeEl.className = `text-lg font-semibold ${priceChange >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`;
        }
        
        if (percentChangeEl) {
            percentChangeEl.textContent = `${priceChange >= 0 ? '+' : ''}${Math.abs(percentChange).toFixed(2)}%`;
            percentChangeEl.className = `text-lg font-semibold ${percentChange >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}`;
        }
    }
}

// Auto-uppercase ticker input and setup form validation
function setupTickerInput() {
    const tickerInput = document.getElementById('ticker');
    if (tickerInput) {
        tickerInput.addEventListener('input', function() {
            this.value = this.value.toUpperCase();
            // Clear any validation errors
            this.setCustomValidity('');
        });
        
        // Override form validation completely
        tickerInput.addEventListener('invalid', function(e) {
            e.preventDefault();
            console.log('Validation error prevented:', e);
        });
    }
}

// Toggle custom days input
function setupCustomDaysToggle() {
    const daysSelect = document.getElementById('days');
    const customDaysContainer = document.getElementById('customDaysContainer');
    
    if (daysSelect && customDaysContainer) {
        daysSelect.addEventListener('change', function() {
            if (this.value === 'custom') {
                customDaysContainer.classList.remove('hidden');
                customDaysContainer.classList.add('block');
            } else {
                customDaysContainer.classList.remove('block');
                customDaysContainer.classList.add('hidden');
            }
        });
    }
}

/**
 * Generate basic analysis content
 * @param {string} ticker - Stock ticker symbol
 * @param {Object} chartData - Chart data object
 * @returns {string} HTML content for basic analysis
 */
function generateBasicAnalysis(ticker, chartData) {
    if (!chartData || !chartData.prices || chartData.prices.length === 0) {
        return `<div class="text-gray-600 dark:text-gray-400">Unable to generate analysis - no price data available for ${ticker}.</div>`;
    }
    
    const prices = chartData.prices;
    const volumes = chartData.volumes || [];
    const opens = chartData.opens || [];
    const highs = chartData.highs || [];
    const lows = chartData.lows || [];
    const rsi = chartData.rsi || [];
    const ma50 = chartData.ma50 || [];
    const ma200 = chartData.ma200 || [];
    
    const currentPrice = prices[prices.length - 1];
    const previousPrice = prices.length > 1 ? prices[prices.length - 2] : currentPrice;
    const priceChange = currentPrice - previousPrice;
    const percentChange = previousPrice !== 0 ? (priceChange / previousPrice) * 100 : 0;
    
    // Calculate price statistics
    const highPrice = Math.max(...prices);
    const lowPrice = Math.min(...prices);
    const priceRange = highPrice - lowPrice;
    
    // Calculate volatility (standard deviation of returns)
    const returns = [];
    for (let i = 1; i < prices.length; i++) {
        returns.push((prices[i] - prices[i-1]) / prices[i-1]);
    }
    const avgReturn = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
    const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - avgReturn, 2), 0) / returns.length;
    const volatility = Math.sqrt(variance) * Math.sqrt(252) * 100; // Annualized volatility
    
    // Calculate volume metrics
    let avgVolume = 0;
    let volumeTrend = 'stable';
    if (volumes.length > 0) {
        avgVolume = volumes.reduce((sum, vol) => sum + vol, 0) / volumes.length;
        const recentVolume = volumes.slice(-5).reduce((sum, vol) => sum + vol, 0) / Math.min(5, volumes.length);
        const earlierVolume = volumes.slice(0, Math.max(1, volumes.length - 5)).reduce((sum, vol) => sum + vol, 0) / Math.max(1, volumes.length - 5);
        if (recentVolume > earlierVolume * 1.2) {
            volumeTrend = 'increasing';
        } else if (recentVolume < earlierVolume * 0.8) {
            volumeTrend = 'decreasing';
        }
    }
    
    // Determine trend and momentum
    const startPrice = prices[0];
    const endPrice = prices[prices.length - 1];
    const overallChange = ((endPrice - startPrice) / startPrice) * 100;
    
    // Calculate momentum (recent 20% vs earlier 80%)
    const recentPeriod = Math.max(1, Math.floor(prices.length * 0.2));
    const recentPrices = prices.slice(-recentPeriod);
    const recentChange = ((recentPrices[recentPrices.length - 1] - recentPrices[0]) / recentPrices[0]) * 100;
    
    let trendDescription = '';
    let momentum = '';
    
    if (overallChange > 5) {
        trendDescription = 'strong upward trend';
    } else if (overallChange > 1) {
        trendDescription = 'moderate upward trend';
    } else if (overallChange < -5) {
        trendDescription = 'strong downward trend';
    } else if (overallChange < -1) {
        trendDescription = 'moderate downward trend';
    } else {
        trendDescription = 'sideways movement';
    }
    
    if (recentChange > overallChange + 2) {
        momentum = 'accelerating upward';
    } else if (recentChange < overallChange - 2) {
        momentum = 'decelerating';
    } else {
        momentum = 'steady';
    }
    
    // Technical indicators summary
    let technicalSignals = [];
    if (rsi.length > 0) {
        const currentRSI = rsi[rsi.length - 1];
        if (currentRSI > 70) {
            technicalSignals.push('RSI indicates overbought conditions');
        } else if (currentRSI < 30) {
            technicalSignals.push('RSI indicates oversold conditions');
        } else {
            technicalSignals.push('RSI is in neutral territory');
        }
    }
    
    if (ma50.length > 0 && ma200.length > 0) {
        const currentMA50 = ma50[ma50.length - 1];
        const currentMA200 = ma200[ma200.length - 1];
        if (currentPrice > currentMA50 && currentMA50 > currentMA200) {
            technicalSignals.push('Price is above both moving averages (bullish)');
        } else if (currentPrice < currentMA50 && currentMA50 < currentMA200) {
            technicalSignals.push('Price is below both moving averages (bearish)');
        } else {
            technicalSignals.push('Mixed signals from moving averages');
        }
    }
    
    // Risk assessment
    let riskLevel = '';
    if (volatility > 40) {
        riskLevel = 'High';
    } else if (volatility > 20) {
        riskLevel = 'Moderate';
    } else {
        riskLevel = 'Low';
    }
    
    const changeColor = priceChange >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400';
    const trendColor = overallChange >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400';
    const riskColor = riskLevel === 'High' ? 'text-red-600 dark:text-red-400' : 
                     riskLevel === 'Moderate' ? 'text-yellow-600 dark:text-yellow-400' : 
                     'text-green-600 dark:text-green-400';
    
    return `
        <div class="space-y-4">
            <div class="border-l-4 border-blue-500 pl-4">
                <h5 class="font-semibold text-gray-900 dark:text-gray-100 mb-2">Enhanced Basic Analysis for ${ticker}</h5>
                <p class="text-gray-700 dark:text-gray-300">
                    Current price: <span class="font-medium">${formatPrice(currentPrice)}</span>
                    <span class="${changeColor} ml-2">
                        ${priceChange >= 0 ? '+' : ''}${formatPrice(priceChange).replace('$', '$')} 
                        (${percentChange >= 0 ? '+' : ''}${percentChange.toFixed(2)}%)
                    </span>
                </p>
            </div>
            
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                <div>
                    <h6 class="font-medium text-gray-900 dark:text-gray-100 mb-1">Price Metrics</h6>
                    <p class="text-gray-600 dark:text-gray-400">
                        High: <span class="font-medium">${formatPrice(highPrice)}</span><br>
                        Low: <span class="font-medium">${formatPrice(lowPrice)}</span><br>
                        Range: <span class="font-medium">${formatPrice(priceRange)}</span><br>
                        Volatility: <span class="${riskColor} font-medium">${volatility.toFixed(1)}% (${riskLevel})</span>
                    </p>
                </div>
                
                <div>
                    <h6 class="font-medium text-gray-900 dark:text-gray-100 mb-1">Performance</h6>
                    <p class="text-gray-600 dark:text-gray-400">
                        Period change: <span class="${trendColor} font-medium">
                            ${overallChange >= 0 ? '+' : ''}${overallChange.toFixed(2)}%
                        </span><br>
                        Trend: <span class="font-medium">${trendDescription}</span><br>
                        Momentum: <span class="font-medium">${momentum}</span><br>
                        Recent trend: <span class="${recentChange >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'} font-medium">
                            ${recentChange >= 0 ? '+' : ''}${recentChange.toFixed(1)}%
                        </span>
                    </p>
                </div>
                
                <div>
                    <h6 class="font-medium text-gray-900 dark:text-gray-100 mb-1">Volume & Signals</h6>
                    <p class="text-gray-600 dark:text-gray-400">
                        ${avgVolume > 0 ? `Avg. Volume: <span class="font-medium">${formatVolume(avgVolume)}</span><br>` : ''}
                        ${avgVolume > 0 ? `Volume trend: <span class="font-medium">${volumeTrend}</span><br>` : ''}
                        ${rsi.length > 0 ? `RSI: <span class="font-medium">${rsi[rsi.length - 1].toFixed(1)}</span><br>` : ''}
                        Support: <span class="font-medium">${formatPrice(lowPrice)}</span><br>
                        Resistance: <span class="font-medium">${formatPrice(highPrice)}</span>
                    </p>
                </div>
            </div>
            
            ${technicalSignals.length > 0 ? `
            <div class="bg-blue-50 dark:bg-blue-900/30 p-3 rounded-lg">
                <h6 class="font-medium text-blue-900 dark:text-blue-100 mb-2">Technical Signals</h6>
                <ul class="text-sm text-blue-800 dark:text-blue-200 space-y-1">
                    ${technicalSignals.map(signal => `<li>• ${signal}</li>`).join('')}
                </ul>
            </div>
            ` : ''}
            
            <div class="bg-gray-50 dark:bg-gray-700 p-3 rounded-lg">
                <p class="text-sm text-gray-600 dark:text-gray-400">
                    <strong>Summary:</strong> ${ticker} is showing ${trendDescription} with ${momentum} momentum over the analyzed period. 
                    ${priceChange >= 0 ? 'The stock gained' : 'The stock lost'} 
                    ${Math.abs(percentChange).toFixed(2)}% in the most recent session.
                    The stock exhibits ${riskLevel.toLowerCase()} volatility (${volatility.toFixed(1)}%) and volume is ${volumeTrend}.
                    ${overallChange > 10 ? ' This represents significant growth with strong bullish momentum.' : 
                      overallChange < -10 ? ' This represents a notable decline requiring caution.' : 
                      ' The stock has shown relatively balanced performance with moderate price action.'}
                </p>
            </div>
        </div>
    `;
}

/**
 * Update stock info in the UI
 * @param {string} ticker - Stock ticker symbol
 * @param {Object} chartData - Chart data object
 */
function updateStockInfo(ticker, chartData) {
    // Update stock ticker and company name
    const stockTickerElement = document.getElementById('stockTicker');
    const companyNameElement = document.getElementById('companyName');
    
    if (stockTickerElement) {
        stockTickerElement.textContent = ticker;
    }
    if (companyNameElement) {
        companyNameElement.textContent = `${ticker} - Stock Analysis`;
    }
    
    // Update price summary
    if (chartData.opens && chartData.highs && chartData.lows && chartData.prices) {
        const latestIndex = chartData.prices.length - 1;
        
        // Update OHLC prices
        const openPrice = document.getElementById('openPrice');
        const highPrice = document.getElementById('highPrice');
        const lowPrice = document.getElementById('lowPrice');
        const closePrice = document.getElementById('closePrice');
        
        if (openPrice && chartData.opens[latestIndex]) {
            openPrice.textContent = formatPrice(chartData.opens[latestIndex]);
        }
        if (highPrice && chartData.highs[latestIndex]) {
            highPrice.textContent = formatPrice(chartData.highs[latestIndex]);
        }
        if (lowPrice && chartData.lows[latestIndex]) {
            lowPrice.textContent = formatPrice(chartData.lows[latestIndex]);
        }
        if (closePrice && chartData.prices[latestIndex]) {
            closePrice.textContent = formatPrice(chartData.prices[latestIndex]);
        }
    }
    
    // Update technical indicators
    if (chartData.rsi && chartData.rsi.length > 0) {
        const rsiValue = document.getElementById('rsiValue');
        if (rsiValue) {
            rsiValue.textContent = chartData.rsi[chartData.rsi.length - 1].toFixed(1);
        }
    }
    
    if (chartData.macdLine && chartData.macdLine.length > 0) {
        const macdValue = document.getElementById('macdValue');
        if (macdValue) {
            macdValue.textContent = chartData.macdLine[chartData.macdLine.length - 1].toFixed(4);
        }
    }
    
    if (chartData.ma50 && chartData.ma50.length > 0) {
        const sma50 = document.getElementById('sma50');
        if (sma50) {
            sma50.textContent = formatPrice(chartData.ma50[chartData.ma50.length - 1]);
        }
    }
    
    if (chartData.ma200 && chartData.ma200.length > 0) {
        const sma200 = document.getElementById('sma200');
        if (sma200) {
            sma200.textContent = formatPrice(chartData.ma200[chartData.ma200.length - 1]);
        }
    }
    
    // Update main price display
    if (chartData.prices && chartData.prices.length > 0) {
        const currentPrice = chartData.prices[chartData.prices.length - 1];
        const previousPrice = chartData.prices.length > 1 ? chartData.prices[chartData.prices.length - 2] : currentPrice;
        const priceChange = currentPrice - previousPrice;
        const percentChange = previousPrice !== 0 ? (priceChange / previousPrice) * 100 : 0;
        
        // Update current price display
        const currentPriceElement = document.getElementById('currentPrice');
        const stockSymbolElement = document.getElementById('stockSymbol');
        const stockNameElement = document.getElementById('stockName');
        
        if (currentPriceElement) {
            currentPriceElement.textContent = formatPrice(currentPrice);
        }
        if (stockSymbolElement) {
            stockSymbolElement.textContent = ticker;
        }
        if (stockNameElement) {
            stockNameElement.textContent = `${ticker} Stock`;
        }
        
        // Update price change
        const priceChangeElement = document.getElementById('priceChange');
        const percentChangeElement = document.getElementById('percentChange');
        
        if (priceChangeElement) {
            priceChangeElement.textContent = `${priceChange >= 0 ? '+' : ''}${formatPrice(priceChange).replace('$', '$')}`;
            const changeClass = priceChange >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400';
            priceChangeElement.className = `text-lg font-semibold ${changeClass}`;
        }
        
        if (percentChangeElement) {
            percentChangeElement.textContent = `${percentChange >= 0 ? '+' : ''}${percentChange.toFixed(2)}%`;
            const changeClass = priceChange >= 0 ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400';
            percentChangeElement.className = `text-lg font-semibold ${changeClass}`;
        }
    }
}

/**
 * Format a number with commas and optional decimal places
 * @param {number} num - The number to format
 * @param {number} [decimals=2] - Number of decimal places to show
 * @returns {string} Formatted number string
 */
function formatNumber(num, decimals = 2) {
    if (num === null || num === undefined) return 'N/A';
    if (typeof num === 'string') num = parseFloat(num);
    if (isNaN(num)) return 'N/A';
    
    return num.toLocaleString(undefined, {
        minimumFractionDigits: 0,
        maximumFractionDigits: decimals
    });
}

/**
 * Update a metric in the UI
 * @param {string} id - The element ID
 * @param {*} value - The value to display
 * @param {string} [suffix=''] - Optional suffix to add after the value
 * @param {number} [decimals=2] - Number of decimal places to show
 */
function updateMetric(id, value, suffix = '', decimals = 2) {
    const element = document.getElementById(id);
    if (!element) return;
    
    if (value === null || value === undefined || value === '') {
        element.textContent = 'N/A';
        return;
    }
    
    if (typeof value === 'number') {
        element.textContent = formatNumber(value, decimals) + suffix;
    } else {
        element.textContent = value + suffix;
    }
}

/**
 * Update the fundamental analysis UI with data
 * @param {Object} data - The fundamental data to display
 */
function updateFundamentalUI(data) {
    if (!data) return;
    
    // Update valuation metrics
    if (data.valuation) {
        updateMetric('pe-ratio', data.valuation.peRatio, 'x');
        updateMetric('ps-ratio', data.valuation.psRatio, 'x');
        updateMetric('pb-ratio', data.valuation.pbRatio, 'x');
        
        // Format market cap with appropriate suffix (B for billion, M for million)
        if (data.valuation.marketCap) {
            let marketCap = data.valuation.marketCap;
            let suffix = '';
            if (marketCap >= 1e9) {
                marketCap = marketCap / 1e9;
                suffix = 'B';
            } else if (marketCap >= 1e6) {
                marketCap = marketCap / 1e6;
                suffix = 'M';
            }
            updateMetric('market-cap', marketCap, ' ' + suffix + ' USD');
        } else {
            updateMetric('market-cap', null);
        }
        
        // Format dividend yield as percentage
        if (data.valuation.dividendYield) {
            const divYield = data.valuation.dividendYield * 100; // Convert to percentage
            updateMetric('dividend-yield', divYield, '%', 2);
        } else {
            updateMetric('dividend-yield', null);
        }
    }
    
    // Update financial health metrics
    if (data.financialHealth) {
        updateMetric('current-ratio', data.financialHealth.currentRatio, 'x');
        updateMetric('debt-equity', data.financialHealth.debtToEquity, 'x');
        updateMetric('quick-ratio', data.financialHealth.quickRatio, 'x');
    }
    
    // Update growth metrics
    if (data.growth) {
        // Convert growth rates to percentages
        const revenueGrowth = data.growth.revenueGrowth ? data.growth.revenueGrowth * 100 : null;
        const earningsGrowth = data.growth.earningsGrowth ? data.growth.earningsGrowth * 100 : null;
        
        updateMetric('revenue-growth', revenueGrowth, '%', 2);
        updateMetric('earnings-growth', earningsGrowth, '%', 2);
        
        // Update growth summary
        const growthSummary = document.getElementById('growth-summary');
        if (growthSummary) {
            if (revenueGrowth !== null && earningsGrowth !== null) {
                growthSummary.textContent = `The company has shown a revenue growth of ${formatNumber(revenueGrowth, 2)}% ` +
                                         `and earnings growth of ${formatNumber(earningsGrowth, 2)}% year over year.`;
            } else if (revenueGrowth !== null) {
                growthSummary.textContent = `The company has shown a revenue growth of ${formatNumber(revenueGrowth, 2)}% year over year.`;
            } else if (earningsGrowth !== null) {
                growthSummary.textContent = `The company has shown an earnings growth of ${formatNumber(earningsGrowth, 2)}% year over year.`;
            } else {
                growthSummary.textContent = 'Growth data not available.';
            }
        }
    }
}

/**
 * Fetch and update fundamental analysis data for a ticker
 * @param {string} ticker - The stock ticker symbol
 */
function updateFundamentalAnalysis(ticker) {
    if (!ticker) return;
    
    const loadingElement = document.getElementById('fundamental-loading');
    const errorElement = document.getElementById('fundamental-error');
    
    // Show loading state
    if (loadingElement) loadingElement.classList.remove('hidden');
    if (errorElement) errorElement.classList.add('hidden');
    
    // Fetch fundamental data from the backend
    fetch(`/api/stock-fundamentals/${ticker}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            // Update the UI with the fetched data
            updateFundamentalUI(data);
            
            // Hide loading state
            if (loadingElement) loadingElement.classList.add('hidden');
        })
        .catch(error => {
            console.error('Error fetching fundamental data:', error);
            
            // Show error message
            if (errorElement) {
                errorElement.textContent = `Failed to load fundamental data: ${error.message}`;
                errorElement.classList.remove('hidden');
            }
            
            // Hide loading state
            if (loadingElement) loadingElement.classList.add('hidden');
        });
}

/**
 * Setup collapsible sections for technical analysis
 */
function setupCollapsibleSections() {
    // Find all elements with data-toggle attribute
    const toggleElements = document.querySelectorAll('[data-toggle]');
    
    toggleElements.forEach(toggleElement => {
        toggleElement.addEventListener('click', function() {
            const targetId = this.getAttribute('data-toggle');
            const targetElement = document.getElementById(targetId);
            const chevronIcon = this.querySelector('.fa-chevron-down, .fa-chevron-up');
            
            if (targetElement) {
                // Toggle the hidden class
                targetElement.classList.toggle('hidden');
                
                // Toggle chevron icon
                if (chevronIcon) {
                    if (targetElement.classList.contains('hidden')) {
                        chevronIcon.classList.remove('fa-chevron-up');
                        chevronIcon.classList.add('fa-chevron-down');
                    } else {
                        chevronIcon.classList.remove('fa-chevron-down');
                        chevronIcon.classList.add('fa-chevron-up');
                    }
                }
            }
        });
    });
}
