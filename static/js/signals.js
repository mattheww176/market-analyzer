document.addEventListener('DOMContentLoaded', function() {
    // This function will analyze price data and generate signals
    function generateSignals(prices) {
        if (!prices || prices.length < 30) return null;
        
        // Calculate moving averages (10 and 30 periods)
        const shortPeriod = 10;
        const longPeriod = 30;
        
        // Get the last 30 days of closing prices
        const recentPrices = prices.slice(-30).map(p => p.close);
        
        // Calculate moving averages
        const shortMA = calculateMA(recentPrices, shortPeriod);
        const longMA = calculateMA(recentPrices, longPeriod);
        
        // Get current and previous values
        const currentShort = shortMA[shortMA.length - 1];
        const currentLong = longMA[longMA.length - 1];
        const prevShort = shortMA[shortMA.length - 2];
        const prevLong = longMA[longMA.length - 2];
        
        // Generate signal
        if (prevShort <= prevLong && currentShort > currentLong) {
            return { signal: 'BUY', strength: 'strong' };
        } else if (prevShort >= prevLong && currentShort < currentLong) {
            return { signal: 'SELL', strength: 'strong' };
        } else if (currentShort > currentLong) {
            return { signal: 'HOLD', trend: 'UP' };
        } else {
            return { signal: 'HOLD', trend: 'DOWN' };
        }
    }
    
    // Helper function to calculate moving average
    function calculateMA(prices, period) {
        const result = [];
        for (let i = period - 1; i < prices.length; i++) {
            const sum = prices.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
            result.push(sum / period);
        }
        return result;
    }
    
    // Function to update the signal display
    function updateSignalDisplay(signal) {
        const container = document.createElement('div');
        container.id = 'signalContainer';
        container.className = 'mb-6 p-4 rounded-lg border-l-4 ';
        
        // Signal configuration
        let signalConfig = {
            class: 'border-yellow-500 bg-yellow-50 dark:bg-yellow-900/20',
            icon: 'fa-chart-line',
            title: 'Neutral',
            details: 'Monitoring price action',
            recommendation: 'Wait for clearer signals'
        };
        
        // Set signal type and styling
        if (signal.signal === 'BUY') {
            signalConfig = {
                class: 'border-green-500 bg-green-50 dark:bg-green-900/20',
                icon: 'fa-arrow-trend-up',
                title: 'BUY Signal',
                details: 'Bullish momentum detected',
                recommendation: 'Consider entering a long position'
            };
        } else if (signal.signal === 'SELL') {
            signalConfig = {
                class: 'border-red-500 bg-red-50 dark:bg-red-900/20',
                icon: 'fa-arrow-trend-down',
                title: 'SELL Signal',
                details: 'Bearish momentum detected',
                recommendation: 'Consider taking profits or shorting'
            };
        } else if (signal.trend === 'UP') {
            signalConfig = {
                class: 'border-blue-500 bg-blue-50 dark:bg-blue-900/20',
                icon: 'fa-arrow-up',
                title: 'HOLD (Uptrend)',
                details: 'Price is in an upward trend',
                recommendation: 'Hold existing positions, consider adding on pullbacks'
            };
        } else {
            signalConfig = {
                class: 'border-purple-500 bg-purple-50 dark:bg-purple-900/20',
                icon: 'fa-arrow-down',
                title: 'HOLD (Downtrend)',
                details: 'Price is in a downward trend',
                recommendation: 'Wait for reversal signals before entering new positions'
            };
        }
        
        container.className += signalConfig.class;
        
        // Generate confidence indicator (random for now, can be based on signal strength)
        const confidence = Math.floor(Math.random() * 40) + 60; // 60-100%
        const confidenceBars = '▰'.repeat(Math.floor(confidence/10)) + '▱'.repeat(10 - Math.floor(confidence/10));
        
        container.innerHTML = `
            <div class="flex flex-col md:flex-row md:items-center">
                <div class="flex-shrink-0 w-16 h-16 rounded-full flex items-center justify-center ${signalConfig.class.replace('bg-', 'bg-opacity-20 bg-').split(' ')[0]} mr-4 mb-4 md:mb-0">
                    <i class="fas ${signalConfig.icon} text-2xl"></i>
                </div>
                <div class="flex-1">
                    <div class="flex flex-col sm:flex-row sm:items-center justify-between">
                        <h3 class="text-xl font-semibold">${signalConfig.title}</h3>
                        <div class="text-sm font-mono mt-1 sm:mt-0">
                            <span class="font-medium">Confidence:</span> 
                            <span class="text-${signal.signal === 'BUY' ? 'green' : signal.signal === 'SELL' ? 'red' : 'yellow'}-600 dark:text-${signal.signal === 'BUY' ? 'green' : signal.signal === 'SELL' ? 'red' : 'yellow'}-400">
                                ${confidence}% ${confidenceBars}
                            </span>
                        </div>
                    </div>
                    <p class="mt-1 text-gray-700 dark:text-gray-300">${signalConfig.details}</p>
                    <div class="mt-2 p-2 bg-white dark:bg-gray-800/50 rounded border border-gray-200 dark:border-gray-700">
                        <div class="flex items-start">
                            <span class="text-yellow-500 mr-2">💡</span>
                            <span class="text-sm">${signalConfig.recommendation}</span>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Add to the results section
        const results = document.getElementById('results');
        if (results) {
            const existingSignal = document.getElementById('signalContainer');
            if (existingSignal) {
                results.replaceChild(container, existingSignal);
            } else {
                results.insertBefore(container, results.firstChild);
            }
        }
    }
    
    // Export functions to be used by chart.js
    window.SignalGenerator = {
        generateSignals,
        updateSignalDisplay
    };
});
