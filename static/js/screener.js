
document.addEventListener('DOMContentLoaded', function() {
    const screenerForm = document.getElementById('screenerForm');
    const resetBtn = document.getElementById('resetFilters');
    const resultsContainer = document.getElementById('screenerResultsContainer');
    const stockRowTemplate = document.getElementById('stockRowTemplate');
    const resultsTable = document.getElementById('screenerResultsTable');
    const resultsBody = document.getElementById('screenerResults');

    // Format number with commas
    function formatNumber(num) {
        return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
    }

    // Format percentage
    function formatPercent(num) {
        return num.toFixed(2) + '%';
    }

    // Format price change with color
    function formatChange(change, percent) {
        const isPositive = change >= 0;
        const sign = isPositive ? '+' : '';
        const colorClass = isPositive ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400';
        return `
            <span class="${colorClass}">
                ${sign}${change.toFixed(2)} (${sign}${percent.toFixed(2)}%)
            </span>
        `;
    }

    // Format market cap
    function formatMarketCap(marketCap) {
        if (marketCap >= 1000000000000) {
            return '$' + (marketCap / 1000000000000).toFixed(2) + 'T';
        } else if (marketCap >= 1000000000) {
            return '$' + (marketCap / 1000000000).toFixed(2) + 'B';
        } else if (marketCap >= 1000000) {
            return '$' + (marketCap / 1000000).toFixed(2) + 'M';
        } else {
            return '$' + marketCap.toFixed(2);
        }
    }

    // Display results in the table - make it global
    window.displayResults = function displayResults(stocks) {
        console.log('displayResults called with:', stocks ? stocks.length : 'null', 'stocks');
        
        // Get DOM elements
        const resultsBody = document.getElementById('screenerResults');
        if (!resultsBody) {
            console.error('resultsBody element not found!');
            return;
        }
        console.log('resultsBody found:', resultsBody);
        
        // Clear previous results
        resultsBody.innerHTML = '';
        
        if (!stocks || stocks.length === 0) {
            const row = document.createElement('tr');
            row.className = 'text-center py-4';
            row.innerHTML = `
                <td colspan="8" class="px-6 py-4 text-gray-500 dark:text-gray-400">
                    No stocks match your criteria. Try adjusting your filters.
                </td>
            `;
            resultsBody.appendChild(row);
            return;
        }

        // Add each stock to the table
        stocks.forEach(stock => {
            if (!stock) return;
            
            const symbol = stock.symbol || 'N/A';
            const name = stock.name || 'N/A';
            const price = stock.price !== undefined ? stock.price.toFixed(2) : 'N/A';
            const change = stock.change || 0;
            const changePercent = stock.change_percent || 0;
            const marketCap = stock.market_cap !== undefined ? stock.market_cap : 0;
            const volume = stock.volume !== undefined ? stock.volume : 0;
            const peRatio = stock.pe_ratio !== undefined ? stock.pe_ratio : null;
            const dividendYield = stock.dividend_yield !== undefined ? stock.dividend_yield : 0;
            const sector = stock.sector || 'N/A';

            const row = document.createElement('tr');
            row.className = 'hover:bg-gray-50 dark:hover:bg-gray-750 transition-colors';
            row.innerHTML = `
                <td class="px-6 py-4 whitespace-nowrap">
                    <div class="flex items-center">
                        <div class="flex-shrink-0 h-10 w-10 flex items-center justify-center bg-blue-100 dark:bg-blue-900 rounded-md">
                            <span class="text-blue-600 dark:text-blue-300 font-medium">${symbol[0] || '?'}</span>
                        </div>
                        <div class="ml-4">
                            <div class="text-sm font-medium text-gray-900 dark:text-white">${symbol}</div>
                            <div class="text-sm text-gray-500 dark:text-gray-400">${name}</div>
                        </div>
                    </div>
                </td>
                <td class="px-6 py-4 whitespace-nowrap">
                    <div class="text-sm text-gray-900 dark:text-white">${price === 'N/A' ? price : '$' + price}</div>
                </td>
                <td class="px-6 py-4 whitespace-nowrap">
                    ${formatChange(change, changePercent)}
                </td>
                <td class="px-6 py-4 whitespace-nowrap">
                    <div class="text-sm text-gray-900 dark:text-white">${formatMarketCap(marketCap)}</div>
                </td>
                <td class="px-6 py-4 whitespace-nowrap">
                    <div class="text-sm text-gray-900 dark:text-white">${formatNumber(volume)}</div>
                </td>
                <td class="px-6 py-4 whitespace-nowrap">
                    <div class="text-sm text-gray-900 dark:text-white">${peRatio !== null ? peRatio.toFixed(2) : 'N/A'}</div>
                </td>
                <td class="px-6 py-4 whitespace-nowrap">
                    <div class="text-sm text-gray-900 dark:text-white">${dividendYield > 0 ? dividendYield.toFixed(2) + '%' : 'N/A'}</div>
                </td>
                <td class="px-6 py-4 whitespace-nowrap">
                    <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200">
                        ${sector}
                    </span>
                </td>
            `;
            resultsBody.appendChild(row);
        });
    }

    // Filter stocks based on form inputs
    function filterStocks() {
        const loadingIndicator = document.getElementById('loadingIndicator');
        const resultsTable = document.getElementById('screenerResultsTable');
        
        const formData = new FormData(screenerForm);
        const filters = {
            minPrice: formData.get('minPrice') ? parseFloat(formData.get('minPrice')) : null,
            maxPrice: formData.get('maxPrice') ? parseFloat(formData.get('maxPrice')) : null,
            marketCap: formData.get('marketCap'),
            minVolume: formData.get('minVolume') ? parseInt(formData.get('minVolume')) : 0,
            sector: formData.get('sector') || '',
            maxPERatio: formData.get('maxPERatio') ? parseFloat(formData.get('maxPERatio')) : Infinity,
            minDividendYield: formData.get('minDividendYield') ? parseFloat(formData.get('minDividendYield')) : 0,
            min52WeekChange: formData.get('min52WeekChange') ? parseFloat(formData.get('min52WeekChange')) : -Infinity,
            max52WeekChange: formData.get('max52WeekChange') ? parseFloat(formData.get('max52WeekChange')) : Infinity
        };

        // Show loading indicator and hide results if elements exist
        if (loadingIndicator) loadingIndicator.classList.remove('hidden');
        if (resultsTable) resultsTable.classList.add('hidden');

        // Use requestAnimationFrame to ensure UI updates before heavy computation
        requestAnimationFrame(() => {
            const filteredStocks = window.stocksData.filter(stock => {
                if (!stock) return false;

                // Price filter
                if (filters.minPrice !== null && stock.price < filters.minPrice) return false;
                if (filters.maxPrice !== null && stock.price > filters.maxPrice) return false;

                // Market cap filter
                if (filters.marketCap && stock.market_cap !== undefined) {
                    const marketCap = stock.market_cap;
                    switch(filters.marketCap) {
                        case 'mega':
                            if (marketCap < 200000000000) return false;
                            break;
                        case 'large':
                            if (marketCap < 10000000000 || marketCap > 200000000000) return false;
                            break;
                        case 'mid':
                            if (marketCap < 2000000000 || marketCap > 10000000000) return false;
                            break;
                        case 'small':
                            if (marketCap > 2000000000) return false;
                            break;
                    }
                }

                // Volume filter
                if (stock.volume < filters.minVolume) return false;

                // Sector filter
                if (filters.sector && stock.sector && stock.sector !== filters.sector) return false;

                // P/E ratio filter
                if (stock.pe_ratio !== undefined && stock.pe_ratio > filters.maxPERatio) return false;

                // Dividend yield filter
                if ((stock.dividend_yield || 0) < filters.minDividendYield) return false;

                // 52-week change filter
                const changePercent = stock.change_percent || 0;
                if (changePercent < filters.min52WeekChange || changePercent > filters.max52WeekChange) return false;

                return true;
            });

            // Display results
            window.displayResults(filteredStocks);
            
            // Hide loading indicator and show results if elements exist
            if (loadingIndicator) loadingIndicator.classList.add('hidden');
            if (resultsTable) resultsTable.classList.remove('hidden');
        });
    }

    // Handle form submission
    screenerForm.addEventListener('submit', function(e) {
        e.preventDefault();
        filterStocks();
    });

    // Handle reset button
    resetBtn.addEventListener('click', function() {
        screenerForm.reset();
        // Show all stocks when resetting
        if (window.stocksData && window.stocksData.length > 0) {
            window.displayResults(window.stocksData);
        }
    });

});
