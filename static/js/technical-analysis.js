/**
 * Update technical analysis data in the UI
 * @param {Object} chartData - Chart data object
 * @param {string} ticker - Stock ticker symbol
 */
function updateTechnicalAnalysisData(chartData, ticker) {
    console.log(`Updating technical analysis for ${ticker}`, chartData);
    
    // Update RSI values if available
    if (chartData.rsi && chartData.rsi.length > 0) {
        updateRSIAnalysis(chartData);
    }
    
    // Update moving averages if available
    if (chartData.ma50 && chartData.ma200) {
        updateMovingAverages(chartData);
    }
    
    // Update volume data if available
    if (chartData.volumes && chartData.volumes.length > 0) {
        updateVolumeAnalysis(chartData);
    }
    
    console.log(`Technical analysis updated for ${ticker}`);
}

/**
 * Update RSI analysis section
 * @param {Object} chartData - Chart data object
 */
function updateRSIAnalysis(chartData) {
    const currentRSI = chartData.rsi[chartData.rsi.length - 1];
    const rsiElements = document.querySelectorAll('#rsi-analysis .text-2xl.font-bold');
    
    if (rsiElements.length > 0) {
        rsiElements[0].textContent = currentRSI.toFixed(1);
    }
    
    // Calculate 14-day average RSI
    const last14RSI = chartData.rsi.slice(-14);
    const avgRSI = last14RSI.reduce((a, b) => a + b, 0) / last14RSI.length;
    
    if (rsiElements.length > 1) {
        rsiElements[1].textContent = avgRSI.toFixed(1);
    }
    
    // Update RSI status
    const rsiStatusElement = document.querySelector('[data-toggle="rsi-analysis"] .px-2\\.5');
    if (rsiStatusElement) {
        if (currentRSI > 70) {
            rsiStatusElement.textContent = 'Overbought';
            rsiStatusElement.className = 'px-2.5 py-0.5 rounded-full text-xs font-medium bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-200 mr-3';
        } else if (currentRSI < 30) {
            rsiStatusElement.textContent = 'Oversold';
            rsiStatusElement.className = 'px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-200 mr-3';
        } else {
            rsiStatusElement.textContent = 'Neutral';
            rsiStatusElement.className = 'px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-200 mr-3';
        }
    }
    
    // Update signal strength bar
    const strengthBar = document.querySelector('#rsi-analysis .bg-red-500');
    const strengthText = document.querySelector('#rsi-analysis .text-red-600');
    
    if (strengthBar && strengthText) {
        let strength, color, text;
        if (currentRSI > 70 || currentRSI < 30) {
            strength = Math.abs(currentRSI - 50) * 2; // 0-100 scale
            color = currentRSI > 70 ? 'bg-purple-500' : 'bg-red-500';
            text = strength > 60 ? 'Strong' : 'Moderate';
        } else {
            strength = 100 - Math.abs(currentRSI - 50) * 2;
            color = 'bg-gray-500';
            text = 'Weak';
        }
        strengthBar.style.width = `${Math.min(strength, 100)}%`;
        strengthBar.className = `${color} h-2.5 rounded-full`;
        strengthText.textContent = text;
    }
}

/**
 * Update moving averages analysis
 * @param {Object} chartData - Chart data object
 */
function updateMovingAverages(chartData) {
    const currentPrice = chartData.prices[chartData.prices.length - 1];
    const currentMA50 = chartData.ma50[chartData.ma50.length - 1];
    const currentMA200 = chartData.ma200[chartData.ma200.length - 1];
    
    // Update MA values
    const maElements = document.querySelectorAll('#ma-analysis .text-2xl.font-bold');
    
    if (maElements.length >= 3) {
        // 20-day MA (calculated from price data)
        const last20Prices = chartData.prices.slice(-20);
        const ma20 = last20Prices.reduce((a, b) => a + b, 0) / last20Prices.length;
        maElements[0].textContent = `$${ma20.toFixed(2)}`;
        
        // 50-day MA
        maElements[1].textContent = `$${currentMA50.toFixed(2)}`;
        
        // 200-day MA
        maElements[2].textContent = `$${currentMA200.toFixed(2)}`;
        
        // Calculate percentage differences
        const ma20Comparison = ((currentPrice - ma20) / ma20 * 100).toFixed(1);
        const ma50Comparison = ((currentPrice - currentMA50) / currentMA50 * 100).toFixed(1);
        const ma200Comparison = ((currentPrice - currentMA200) / currentMA200 * 100).toFixed(1);
        
        // Update comparison text
        const comparisonElements = document.querySelectorAll('#ma-analysis .text-xs');
        if (comparisonElements.length >= 3) {
            updateComparisonElement(comparisonElements[0], ma20Comparison, 'current');
            updateComparisonElement(comparisonElements[1], ma50Comparison, '50-day MA');
            updateComparisonElement(comparisonElements[2], ma200Comparison, '200-day MA');
        }
        
        // Update trend analysis
        updateTrendAnalysis(currentPrice, ma20, currentMA50, currentMA200);
    }
}

/**
 * Update a comparison element with the appropriate styling
 * @param {HTMLElement} element - The element to update
 * @param {number} value - The comparison value
 * @param {string} label - The label for the comparison
 */
function updateComparisonElement(element, value, label) {
    if (!element) return;
    
    const isPositive = parseFloat(value) >= 0;
    const sign = isPositive ? '+' : '';
    
    element.textContent = `${sign}${value}% vs ${label}`;
    element.className = `text-xs ${isPositive ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'} mt-1`;
}

/**
 * Update trend analysis text
 * @param {number} price - Current price
 * @param {number} ma20 - 20-day moving average
 * @param {number} ma50 - 50-day moving average
 * @param {number} ma200 - 200-day moving average
 */
function updateTrendAnalysis(price, ma20, ma50, ma200) {
    const maTrendElement = document.querySelector('#ma-analysis .prose p');
    if (!maTrendElement) return;
    
    let trendText = '';
    
    if (price > ma50 && ma50 > ma200) {
        trendText = `Strong bullish trend: Price ($${price.toFixed(2)}) is above both moving averages, with 50-day MA ($${ma50.toFixed(2)}) above 200-day MA ($${ma200.toFixed(2)}). This indicates strong upward momentum.`;
    } else if (price < ma50 && ma50 < ma200) {
        trendText = `Strong bearish trend: Price ($${price.toFixed(2)}) is below both moving averages, with 50-day MA ($${ma50.toFixed(2)}) below 200-day MA ($${ma200.toFixed(2)}). This indicates strong downward momentum.`;
    } else if (price > ma50) {
        trendText = `Mixed trend: Price ($${price.toFixed(2)}) is above 50-day MA ($${ma50.toFixed(2)}) but below 200-day MA ($${ma200.toFixed(2)}). Watch for confirmation signals.`;
    } else {
        trendText = `Bearish bias: Price ($${price.toFixed(2)}) is below 50-day MA ($${ma50.toFixed(2)}). Consider waiting for better entry points.`;
    }
    
    maTrendElement.textContent = trendText;
}

/**
 * Update volume analysis section
 * @param {Object} chartData - Chart data object
 */
function updateVolumeAnalysis(chartData) {
    const currentVolume = chartData.volumes[chartData.volumes.length - 1];
    const volumeElements = document.querySelectorAll('#volume-analysis .text-2xl.font-bold');
    
    if (volumeElements.length > 0) {
        volumeElements[0].textContent = formatVolume(currentVolume);
        
        // Calculate average volume for comparison
        const last20Volumes = chartData.volumes.slice(-20);
        const avgVolume = last20Volumes.reduce((a, b) => a + b, 0) / last20Volumes.length;
        
        if (volumeElements.length > 1) {
            volumeElements[1].textContent = formatVolume(avgVolume);
        }
        
        // Update volume trend indicator
        const volumeTrendElement = document.querySelector('#volume-analysis .volume-trend');
        if (volumeTrendElement) {
            const trend = currentVolume > avgVolume * 1.5 ? 'High' : 
                         currentVolume > avgVolume * 0.75 ? 'Normal' : 'Low';
            
            let trendClass = 'bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-200';
            if (trend === 'High') trendClass = 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-200';
            if (trend === 'Low') trendClass = 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-200';
            
            volumeTrendElement.textContent = trend;
            volumeTrendElement.className = `px-2.5 py-0.5 rounded-full text-xs font-medium ${trendClass} ml-2`;
        }
    }
}

