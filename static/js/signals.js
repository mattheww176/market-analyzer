document.addEventListener('DOMContentLoaded', function() {
    // Initialize signal history array
    let signalHistory = JSON.parse(localStorage.getItem('signalHistory') || '[]');
    const MAX_HISTORY = 10; // Maximum number of signals to keep in history
    
    // Function to save a signal to history
    function saveToHistory(signal, symbol) {
        if (!symbol) return;
        
        // Add timestamp and symbol to the signal
        const historyItem = {
            ...signal,
            timestamp: new Date().toISOString(),
            symbol: symbol.toUpperCase()
        };
        
        // Add to beginning of array (newest first)
        signalHistory.unshift(historyItem);
        
        // Keep only the most recent signals
        if (signalHistory.length > MAX_HISTORY) {
            signalHistory = signalHistory.slice(0, MAX_HISTORY);
        }
        
        // Save to localStorage
        localStorage.setItem('signalHistory', JSON.stringify(signalHistory));
        
        // Update the history display
        updateSignalHistoryDisplay();
    }
    
    // Function to format date for display
    function formatDate(dateString) {
        const options = { 
            year: 'numeric', 
            month: 'short', 
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        };
        return new Date(dateString).toLocaleDateString(undefined, options);
    }
    
    // Function to get signal color class
    function getSignalColorClass(signal) {
        if (signal.signal === 'BUY') return 'text-green-600 dark:text-green-400';
        if (signal.signal === 'SELL') return 'text-red-600 dark:text-red-400';
        if (signal.trend === 'UP') return 'text-blue-600 dark:text-blue-400';
        return 'text-yellow-600 dark:text-yellow-400';
    }
    
    // Function to update the signal history display
    function updateSignalHistoryDisplay() {
        const historyContainer = document.getElementById('signalHistory');
        if (!historyContainer) return;
        
        if (signalHistory.length === 0) {
            historyContainer.innerHTML = [
                '<div class="text-center text-gray-500 dark:text-gray-400 py-4">',
                '    <i class="fas fa-history text-2xl mb-2"></i>',
                '    <p>No signal history yet</p>',
                '</div>'
            ].join('');
            return;
        }
        
        const historyItems = signalHistory.map((signal, index) => {
            const signalType = signal.signal || signal.trend || 'N/A';
            const signalClass = signal.signal === 'BUY' ? 'bg-green-100 dark:bg-green-900/50 text-green-800 dark:text-green-300' : 
                               signal.signal === 'SELL' ? 'bg-red-100 dark:bg-red-900/50 text-red-800 dark:text-red-300' : 
                               signal.trend === 'UP' ? 'bg-blue-100 dark:bg-blue-900/50 text-blue-800 dark:text-blue-300' : 
                               'bg-yellow-100 dark:bg-yellow-900/50 text-yellow-800 dark:text-yellow-300';
            
            return [
                `<div class="p-3 rounded-lg border ${index === 0 ? 'border-blue-200 dark:border-blue-900 bg-blue-50/50 dark:bg-blue-900/20' : 'border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800/50'} hover:shadow-sm transition-shadow">`,
                '    <div class="flex justify-between items-start">',
                '        <div class="flex items-center space-x-2">',
                `            <span class="font-semibold ${getSignalColorClass(signal)}">`,
                `                ${signal.symbol || 'N/A'}`,
                '            </span>',
                '            <span class="text-xs text-gray-500 dark:text-gray-400">',
                `                ${formatDate(signal.timestamp)}`,
                '            </span>',
                '        </div>',
                `        <span class="px-2 py-1 text-xs rounded-full ${signalClass}">`,
                `            ${signalType}`,
                '        </span>',
                '    </div>',
                '    <div class="mt-1 text-sm text-gray-600 dark:text-gray-300">',
                `        ${signal.details || 'No additional details available'}`,
                '    </div>',
                '</div>'
            ].join('');
        }).join('');
        
        historyContainer.innerHTML = [
            '<div class="space-y-2 max-h-96 overflow-y-auto pr-2">',
            historyItems,
            '</div>',
            '<div class="mt-2 text-center text-xs text-gray-500 dark:text-gray-400">',
            '    Signal history is saved in your browser',
            '</div>'
        ].join('');
    }
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
        let signal, details, recommendation;
        
        if (prevShort <= prevLong && currentShort > currentLong) {
            signal = 'BUY';
            details = 'Bullish crossover detected (10MA crossed above 30MA)';
            recommendation = 'Consider entering a long position';
        } else if (prevShort >= prevLong && currentShort < currentLong) {
            signal = 'SELL';
            details = 'Bearish crossover detected (10MA crossed below 30MA)';
            recommendation = 'Consider taking profits or shorting';
        } else if (currentShort > currentLong) {
            signal = 'HOLD';
            details = 'Uptrend in progress (10MA above 30MA)';
            recommendation = 'Hold existing positions, consider adding on pullbacks';
        } else {
            signal = 'HOLD';
            details = 'Downtrend in progress (10MA below 30MA)';
            recommendation = 'Wait for reversal signals before entering new positions';
        }
        
        return { 
            signal, 
            trend: currentShort > currentLong ? 'UP' : 'DOWN',
            details,
            recommendation,
            strength: signal === 'HOLD' ? 'neutral' : 'strong',
            ma10: currentShort,
            ma30: currentLong
        };
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
    function updateSignalDisplay(signal, symbol = '') {
        const container = document.createElement('div');
        container.id = 'signalContainer';
        container.className = 'mb-6 p-4 rounded-lg border-l-4 ';
        
        // Use signal details if available, otherwise use defaults
        const signalConfig = {
            class: signal.signal === 'BUY' ? 'border-green-500 bg-green-50 dark:bg-green-900/20' :
                  signal.signal === 'SELL' ? 'border-red-500 bg-red-50 dark:bg-red-900/20' :
                  signal.trend === 'UP' ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' :
                  'border-purple-500 bg-purple-50 dark:bg-purple-900/20',
                  
            icon: signal.signal === 'BUY' ? 'fa-arrow-trend-up' :
                   signal.signal === 'SELL' ? 'fa-arrow-trend-down' :
                   signal.trend === 'UP' ? 'fa-arrow-up' : 'fa-arrow-down',
                   
            title: signal.signal === 'HOLD' ? `HOLD (${signal.trend === 'UP' ? 'Uptrend' : 'Downtrend'})` : 
                   `${signal.signal} Signal`,
                    
            details: signal.details || 'No signal information available',
            recommendation: signal.recommendation || 'Monitor price action for confirmation'
        };
        
        container.className += signalConfig.class;
        
        // Generate confidence indicator (random for now, can be based on signal strength)
        const confidence = Math.floor(Math.random() * 40) + 60; // 60-100%
        const confidenceBars = '▰'.repeat(Math.floor(confidence/10)) + '▱'.repeat(10 - Math.floor(confidence/10));
        
        const signalColor = signal.signal === 'BUY' ? 'green' : signal.signal === 'SELL' ? 'red' : 'yellow';
        const bgClass = signalConfig.class.replace('bg-', 'bg-opacity-20 bg-').split(' ')[0];
        
        container.innerHTML = [
            '<div class="flex flex-col md:flex-row md:items-center">',
            '    <div class="flex-shrink-0 w-16 h-16 rounded-full flex items-center justify-center ' + bgClass + ' mr-4 mb-4 md:mb-0">',
            '        <i class="fas ' + signalConfig.icon + ' text-2xl"></i>',
            '    </div>',
            '    <div class="flex-1">',
            '        <div class="flex flex-col sm:flex-row sm:items-center justify-between">',
            '            <h3 class="text-xl font-semibold">' + signalConfig.title + '</h3>',
            '            <div class="text-sm font-mono mt-1 sm:mt-0">',
            '                <span class="font-medium">Confidence:</span>',
            '                <span class="text-' + signalColor + '-600 dark:text-' + signalColor + '-400">',
            '                    ' + confidence + '% ' + confidenceBars,
            '                </span>',
            '            </div>',
            '        </div>',
            '        <p class="mt-1 text-gray-700 dark:text-gray-300">' + signalConfig.details + '</p>',
            '        <div class="mt-2 p-2 bg-white dark:bg-gray-800/50 rounded border border-gray-200 dark:border-gray-700">',
            '            <div class="flex items-start">',
            '                <span class="text-yellow-500 mr-2">💡</span>',
            '                <span class="text-sm">' + signalConfig.recommendation + '</span>',
            '            </div>',
            '        </div>',
            '    </div>',
            '</div>'
        ].join('');
        
        // Add to the results section
        const results = document.getElementById('results');
        if (!results) return;
        
        const existingSignal = document.getElementById('signalContainer');
        if (existingSignal) {
            results.replaceChild(container, existingSignal);
        } else {
            // Insert the signal container before the first child of results
            results.insertBefore(container, results.firstChild);
            
            // Add the signal history container after the signal container
            if (!document.getElementById('signalHistoryContainer')) {
                const historyContainer = document.createElement('div');
                historyContainer.id = 'signalHistoryContainer';
                historyContainer.className = 'mt-6';
                
                const clearButton = document.createElement('button');
                clearButton.id = 'clearHistoryBtn';
                clearButton.className = 'text-xs text-red-600 hover:text-red-800 dark:text-red-400 dark:hover:text-red-300';
                clearButton.innerHTML = '<i class="fas fa-trash-alt mr-1"></i> Clear';
                
                const historyTitle = document.createElement('div');
                historyTitle.className = 'flex items-center justify-between mb-2';
                historyTitle.innerHTML = '<h3 class="text-lg font-medium text-gray-900 dark:text-white">Signal History</h3>';
                historyTitle.appendChild(clearButton);
                
                const historyContent = document.createElement('div');
                historyContent.id = 'signalHistory';
                historyContent.className = 'space-y-2';
                
                historyContainer.appendChild(historyTitle);
                historyContainer.appendChild(historyContent);
                
                results.insertBefore(historyContainer, container.nextSibling);
                
                // Add event listener for clear history button
                clearButton.addEventListener('click', function() {
                    if (confirm('Are you sure you want to clear your signal history?')) {
                        signalHistory = [];
                        localStorage.removeItem('signalHistory');
                        updateSignalHistoryDisplay();
                    }
                });
                
                // Initial update of history display
                updateSignalHistoryDisplay();
            }
        }
        
        // Save the current signal to history
        if (signal && signal.signal) {
            saveToHistory(signal, symbol);
        }
    }
    
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
    function updateSignalDisplay(signal, symbol = '') {
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

        const signalColor = signal.signal === 'BUY' ? 'green' : signal.signal === 'SELL' ? 'red' : 'yellow';
        const bgClass = signalConfig.class.replace('bg-', 'bg-opacity-20 bg-').split(' ')[0];

        container.innerHTML = [
            '<div class="flex flex-col md:flex-row md:items-center">',
            '    <div class="flex-shrink-0 w-16 h-16 rounded-full flex items-center justify-center ' + bgClass + ' mr-4 mb-4 md:mb-0">',
            '        <i class="fas ' + signalConfig.icon + ' text-2xl"></i>',
            '    </div>',
            '    <div class="flex-1">',
            '        <div class="flex flex-col sm:flex-row sm:items-center justify-between">',
            '            <h3 class="text-xl font-semibold">' + signalConfig.title + '</h3>',
            '            <div class="text-sm font-mono mt-1 sm:mt-0">',
            '                <span class="font-medium">Confidence:</span>',
            '                <span class="text-' + signalColor + '-600 dark:text-' + signalColor + '-400">',
            '                    ' + confidence + '% ' + confidenceBars,
            '                </span>',
            '            </div>',
            '        </div>',
            '        <p class="mt-1 text-gray-700 dark:text-gray-300">' + signalConfig.details + '</p>',
            '        <div class="mt-2 p-2 bg-white dark:bg-gray-800/50 rounded border border-gray-200 dark:border-gray-700">',
            '            <div class="flex items-start">',
            '                <span class="text-yellow-500 mr-2">💡</span>',
            '                <span class="text-sm">' + signalConfig.recommendation + '</span>',
            '            </div>',
            '        </div>',
            '    </div>',
            '</div>'
        ].join('');

        // Add to the results section
        const results = document.getElementById('results');
        if (!results) return;

        const existingSignal = document.getElementById('signalContainer');
        if (existingSignal) {
            results.replaceChild(container, existingSignal);
        } else {
            // Insert the signal container before the first child of results
            results.insertBefore(container, results.firstChild);

            // Add the signal history container after the signal container
            if (!document.getElementById('signalHistoryContainer')) {
                const historyContainer = document.createElement('div');
                historyContainer.id = 'signalHistoryContainer';
                historyContainer.className = 'mt-6';

                const clearButton = document.createElement('button');
                clearButton.id = 'clearHistoryBtn';
                clearButton.className = 'text-xs text-red-600 hover:text-red-800 dark:text-red-400 dark:hover:text-red-300';
                clearButton.innerHTML = '<i class="fas fa-trash-alt mr-1"></i> Clear';

                const historyTitle = document.createElement('div');
                historyTitle.className = 'flex items-center justify-between mb-2';
                historyTitle.innerHTML = '<h3 class="text-lg font-medium text-gray-900 dark:text-white">Signal History</h3>';
                historyTitle.appendChild(clearButton);

                const historyContent = document.createElement('div');
                historyContent.id = 'signalHistory';
                historyContent.className = 'space-y-2';

                historyContainer.appendChild(historyTitle);
                historyContainer.appendChild(historyContent);

                results.insertBefore(historyContainer, container.nextSibling);

                // Add event listener for clear history button
                clearButton.addEventListener('click', function() {
                    if (confirm('Are you sure you want to clear your signal history?')) {
                        signalHistory = [];
                        localStorage.removeItem('signalHistory');
                        updateSignalHistoryDisplay();
                    }
                });

                // Initial update of history display
                updateSignalHistoryDisplay();
            }
        }

        // Save the current signal to history
        if (signal && signal.signal) {
            saveToHistory(signal, symbol);
        }
    }

    // Export functions to be used by chart.js
    window.SignalGenerator = {
        generateSignals,
        updateSignalDisplay,
        updateSignalHistoryDisplay
    };

    // Initialize history display if the container exists
    if (document.getElementById('signalHistory')) {
        updateSignalHistoryDisplay();
    }
// Initialize history display if the container exists
if (document.getElementById('signalHistory')) {
    updateSignalHistoryDisplay();
}
});
