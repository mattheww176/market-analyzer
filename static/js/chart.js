// Global chart instance and data
let chart = null;
let chartData = null;

/**
 * Initialize the chart with default configuration
 */
function initChart() {
    const ctx = document.getElementById('stockChart').getContext('2d');
    
    chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Price',
                data: [],
                borderColor: '#3B82F6',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.1,
                pointRadius: 0,
                pointHoverRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                duration: 0
            },
            scales: {
                x: {
                    grid: {
                        display: true,
                        color: 'rgba(0, 0, 0, 0.05)'
                    },
                    ticks: {
                        color: '#6B7280',
                        maxRotation: 0,
                        autoSkip: true,
                        maxTicksLimit: 8
                    }
                },
                y: {
                    position: 'right',
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)'
                    },
                    ticks: {
                        color: '#6B7280',
                        callback: function(value) {
                            return formatPrice(value);
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: {
                        color: '#6B7280',
                        font: { 
                            size: 12,
                            family: "'Inter', 'Helvetica Neue', 'Arial', sans-serif"
                        },
                        padding: 20,
                        usePointStyle: true,
                    }
                },
                tooltip: {
                    enabled: true,
                    mode: 'index',
                    intersect: false,
                    backgroundColor: 'rgba(17, 24, 39, 0.95)',
                    titleColor: '#F3F4F6',
                    bodyColor: '#9CA3AF',
                    borderColor: 'rgba(255, 255, 255, 0.1)',
                    borderWidth: 1,
                    padding: 12,
                    cornerRadius: 6,
                    titleFont: { 
                        size: 14, 
                        weight: 'bold',
                        family: "'Inter', 'Helvetica Neue', 'Arial', sans-serif"
                    },
                    bodyFont: { 
                        size: 13,
                        family: "'Inter', 'Helvetica Neue', 'Arial', sans-serif"
                    },
                    callbacks: {
                        title: function(context) {
                            if (context && context[0] && context[0].label) {
                                const date = new Date(context[0].label);
                                if (!isNaN(date.getTime())) {
                                    return date.toLocaleDateString('en-US', {
                                        year: 'numeric',
                                        month: 'short',
                                        day: 'numeric'
                                    });
                                }
                            }
                            return context[0].label;
                        },
                        label: function(context) {
                            const label = context.dataset.label || '';
                            if (label) {
                                const value = context.parsed.y;
                                if (label.toLowerCase().includes('volume')) {
                                    return `${label}: ${formatVolume(value)}`;
                                } else if (label.toLowerCase().includes('rsi')) {
                                    return `${label}: ${value.toFixed(1)}`;
                                } else if (label.toLowerCase().includes('histogram')) {
                                    return `${label}: ${value.toFixed(4)}`;
                                } else {
                                    return `${label}: ${formatPrice(value)}`;
                                }
                            }
                            return null;
                        }
                    }
                },
                zoom: {
                    pan: {
                        enabled: true,
                        mode: 'x',
                        threshold: 10
                    },
                    zoom: {
                        wheel: {
                            enabled: true,
                            speed: 0.1
                        },
                        pinch: {
                            enabled: true
                        },
                        mode: 'x',
                        onZoomComplete: function({ chart }) {
                            // This prevents the chart from resetting the zoom level
                            chart.update('none');
                        }
                    }
                }
            },
            interaction: {
                mode: 'index',
                intersect: false
            },
            hover: {
                mode: 'index',
                intersect: false
            },
            elements: {
                line: {
                    tension: 0.1,
                    borderWidth: 2,
                    borderCapStyle: 'round'
                },
                point: {
                    radius: 0,
                    hitRadius: 8,
                    hoverRadius: 4,
                    hoverBorderWidth: 2
                }
            }
        }
    });
    
    // Apply dark mode if needed
    updateChartTheme();
}

/**
 * Update chart with new data
 * @param {Object} data - Chart data
 * @param {string} type - Chart type ('price', 'volume', 'rsi', etc.)
 */
function updateChart(data, type = 'price') {
    if (!chart) return;
    
    // Store the data for later use
    chartData = data;
    
    // Generate and display signals if we have price data
    if (data && data.prices && data.prices.length > 30) {
        const signal = window.SignalGenerator?.generateSignals(data.prices);
        if (signal) {
            // Get the current stock symbol from the search input or URL
            const symbolInput = document.getElementById('symbol');
            const symbol = symbolInput ? symbolInput.value : '';
            window.SignalGenerator.updateSignalDisplay(signal, symbol);
        }
    }
    
    // Clear any existing annotations
    if (chart.options.plugins.annotation) {
        chart.options.plugins.annotation.annotations = [];
    }
    
    const dates = data.dates || [];
    const prices = data.prices || [];
    const volumes = data.volumes || [];
    const rsi = data.rsi || [];
    const macdLine = data.macdLine || [];
    const signalLine = data.signalLine || [];
    const histogram = data.histogram || [];
    const ma50 = data.ma50 || [];
    const ma200 = data.ma200 || [];
    
    // Update chart data based on type
    if (type === 'price') {
        // Update chart labels and data
        chart.data.labels = dates;
        chart.data.datasets = [{
            label: 'Price',
            data: prices,
            borderColor: '#3B82F6',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            borderWidth: 2,
            fill: true,
            tension: 0.1,
            pointRadius: 0,
            pointHoverRadius: 4
        }];
        
        // Add moving averages if available
        if (ma50.length > 0) {
            chart.data.datasets.push({
                label: 'SMA 50',
                data: ma50,
                borderColor: '#F59E0B',
                backgroundColor: 'transparent',
                borderWidth: 1.5,
                pointRadius: 0,
                borderDash: [5, 5],
                fill: false
            });
        }
        
        if (ma200.length > 0) {
            chart.data.datasets.push({
                label: 'SMA 200',
                data: ma200,
                borderColor: '#8B5CF6',
                backgroundColor: 'transparent',
                borderWidth: 1.5,
                pointRadius: 0,
                borderDash: [5, 5],
                fill: false
            });
        }
        
    } else if (type === 'volume') {
        // Update chart for volume
        chart.data.labels = dates;
        chart.data.datasets = [{
            label: 'Volume',
            data: volumes,
            type: 'bar',
            backgroundColor: '#10B981',
            borderColor: 'transparent',
            borderWidth: 0,
            borderRadius: 2,
            barPercentage: 0.8,
            categoryPercentage: 1.0
        }];
        
    } else if (type === 'rsi') {
        // Update chart for RSI
        chart.data.labels = dates;
        chart.data.datasets = [{
            label: 'RSI',
            data: rsi,
            borderColor: '#8B5CF6',
            backgroundColor: 'rgba(139, 92, 246, 0.1)',
            borderWidth: 1.5,
            pointRadius: 0,
            fill: true
        }];
        
        // Add RSI levels
        chart.data.datasets.push({
            label: 'Overbought',
            data: Array(dates.length).fill(70),
            borderColor: '#EF4444',
            borderWidth: 1,
            borderDash: [5, 5],
            pointRadius: 0,
            fill: false
        }, {
            label: 'Oversold',
            data: Array(dates.length).fill(30),
            borderColor: '#10B981',
            borderWidth: 1,
            borderDash: [5, 5],
            pointRadius: 0,
            fill: false
        });
        
    } else if (type === 'macd') {
        // Update chart for MACD
        chart.data.labels = dates;
        chart.data.datasets = [{
            label: 'MACD Line',
            data: macdLine,
            borderColor: '#3B82F6',
            backgroundColor: 'transparent',
            borderWidth: 1.5,
            pointRadius: 0,
            fill: false
        }, {
            label: 'Signal Line',
            data: signalLine,
            borderColor: '#F59E0B',
            backgroundColor: 'transparent',
            borderWidth: 1.5,
            pointRadius: 0,
            fill: false
        }];
        
        // Add histogram if available
        if (histogram.length > 0) {
            chart.data.datasets.push({
                label: 'Histogram',
                data: histogram,
                type: 'bar',
                backgroundColor: histogram.map(value => value >= 0 ? '#10B981' : '#EF4444'),
                borderColor: 'transparent',
                borderWidth: 0,
                borderRadius: 2,
                barPercentage: 0.8,
                categoryPercentage: 1.0
            });
        }
    }
    
    // Update chart
    chart.update('none');
}

/**
 * Update chart for the active tab
 * @param {string} tabType - Type of tab ('price', 'volume', 'rsi', 'macd')
 */
function updateChartForTab(tabType) {
    if (!chart) return;
    
    // If no data is loaded yet, just return
    if (!chartData) {
        console.log('No chart data available yet');
        return;
    }
    
    // Show loading indicator
    const chartLoading = document.getElementById('chartLoading');
    if (chartLoading) {
        chartLoading.classList.remove('hidden');
    }
    
    // Small delay to show loading state
    setTimeout(() => {
        updateChart(chartData, tabType);
        if (chartLoading) {
            chartLoading.classList.add('hidden');
        }
    }, 100);
}

/**
 * Toggle fullscreen for the chart
 */
function toggleFullscreen() {
    const chartContainer = document.getElementById('chartContainer');
    if (!document.fullscreenElement) {
        if (chartContainer.requestFullscreen) {
            chartContainer.requestFullscreen();
        } else if (chartContainer.webkitRequestFullscreen) {
            chartContainer.webkitRequestFullscreen();
        } else if (chartContainer.msRequestFullscreen) {
            chartContainer.msRequestFullscreen();
        }
    } else {
        if (document.exitFullscreen) {
            document.exitFullscreen();
        } else if (document.webkitExitFullscreen) {
            document.webkitExitFullscreen();
        } else if (document.msExitFullscreen) {
            document.msExitFullscreen();
        }
    }
}

/**
 * Reset chart zoom
 */
function resetChartZoom() {
    if (chart) {
        chart.resetZoom();
    }
}

/**
 * Update chart theme based on dark/light mode
 */
function updateChartTheme() {
    if (!chart) return;
    
    const isDarkMode = document.documentElement.classList.contains('dark');
    
    // Update chart options
    chart.options.scales.x.ticks.color = isDarkMode ? '#9CA3AF' : '#6B7280';
    chart.options.scales.y.ticks.color = isDarkMode ? '#9CA3AF' : '#6B7280';
    chart.options.scales.y.grid.color = isDarkMode ? 'rgba(255, 255, 255, 0.05)' : 'rgba(0, 0, 0, 0.05)';
    
    // Update tooltip
    chart.options.plugins.tooltip.backgroundColor = isDarkMode ? 'rgba(31, 41, 55, 0.95)' : 'rgba(255, 255, 255, 0.95)';
    chart.options.plugins.tooltip.titleColor = isDarkMode ? '#F3F4F6' : '#111827';
    chart.options.plugins.tooltip.bodyColor = isDarkMode ? '#9CA3AF' : '#4B5563';
    chart.options.plugins.tooltip.borderColor = isDarkMode ? 'rgba(255, 255, 255, 0.1)' : 'rgba(0, 0, 0, 0.1)';
    
    // Update line colors for dark mode
    if (chart.data.datasets.length > 0) {
        chart.data.datasets[0].borderColor = isDarkMode ? '#60A5FA' : '#3B82F6';
        chart.data.datasets[0].backgroundColor = isDarkMode ? 'rgba(96, 165, 250, 0.1)' : 'rgba(59, 130, 246, 0.1)';
    }
    
    // Update the chart
    chart.update('none');
}

// Listen for dark mode changes
const darkModeToggle = document.getElementById('darkModeToggle');
if (darkModeToggle) {
    darkModeToggle.addEventListener('change', updateChartTheme);
}
