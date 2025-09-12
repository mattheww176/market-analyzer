/**
 * Watchlist class to manage stock watchlist functionality
 */
class Watchlist {
    constructor() {
        this.watchlist = new Set(JSON.parse(localStorage.getItem('watchlist') || '[]'));
        this.setupEventListeners();
        this.updateUI();
    }

    /**
     * Add a ticker to the watchlist
     * @param {string} ticker - Stock ticker to add
     */
    add(ticker) {
        if (!ticker) return;
        
        ticker = ticker.trim().toUpperCase();
        if (!this.watchlist.has(ticker)) {
            this.watchlist.add(ticker);
            this.save();
            this.updateUI();
            this.showNotification(`${ticker} added to watchlist`);
        }
    }

    /**
     * Remove a ticker from the watchlist
     * @param {string} ticker - Stock ticker to remove
     */
    remove(ticker) {
        if (!ticker) return;
        
        ticker = ticker.trim().toUpperCase();
        if (this.watchlist.has(ticker)) {
            this.watchlist.delete(ticker);
            this.save();
            this.updateUI();
            this.showNotification(`${ticker} removed from watchlist`);
        }
    }

    /**
     * Clear the entire watchlist
     */
    clear() {
        this.watchlist.clear();
        this.save();
        this.updateUI();
        this.showNotification('Watchlist cleared');
    }

    /**
     * Check if a ticker is in the watchlist
     * @param {string} ticker - Stock ticker to check
     * @returns {boolean} - True if ticker is in watchlist
     */
    has(ticker) {
        return this.watchlist.has(ticker.trim().toUpperCase());
    }

    /**
     * Save watchlist to localStorage
     */
    save() {
        localStorage.setItem('watchlist', JSON.stringify(Array.from(this.watchlist)));
    }

    /**
     * Update the UI to reflect the current watchlist state
     */
    updateUI() {
        // Update watchlist dropdown
        const watchlistDropdown = document.getElementById('watchlistDropdown');
        if (watchlistDropdown) {
            if (this.watchlist.size === 0) {
                watchlistDropdown.innerHTML = `
                    <div class="px-4 py-2 text-sm text-gray-500 dark:text-gray-400">
                        Your watchlist is empty
                    </div>
                `;
            } else {
                watchlistDropdown.innerHTML = Array.from(this.watchlist).map(ticker => `
                    <div class="flex items-center justify-between px-4 py-2 hover:bg-gray-100 dark:hover:bg-gray-700">
                        <a href="#" class="text-gray-700 dark:text-gray-200 hover:text-blue-600 dark:hover:text-blue-400" 
                           onclick="document.getElementById('ticker').value='${ticker}'; document.getElementById('analysisForm').dispatchEvent(new Event('submit'))">
                            ${ticker}
                        </a>
                        <button class="text-gray-400 hover:text-red-500" onclick="watchlist.remove('${ticker}')">
                            <i class="fas fa-times"></i>
                        </button>
                    </div>
                `).join('');
                
                // Add clear all button
                watchlistDropdown.innerHTML += `
                    <div class="border-t border-gray-200 dark:border-gray-700 mt-2 pt-2">
                        <button onclick="watchlist.clear()" class="w-full text-left px-4 py-2 text-sm text-red-600 hover:bg-red-50 dark:text-red-400 dark:hover:bg-gray-700">
                            <i class="fas fa-trash-alt mr-2"></i> Clear All
                        </button>
                    </div>
                `;
            }
        }
        
        // Update watchlist buttons
        const watchlistButtons = document.querySelectorAll('[data-watchlist-ticker]');
        watchlistButtons.forEach(button => {
            const ticker = button.getAttribute('data-watchlist-ticker');
            if (this.has(ticker)) {
                button.innerHTML = '<i class="fas fa-heart"></i> Remove from Watchlist';
                button.classList.remove('bg-gray-100', 'text-gray-800', 'hover:bg-gray-200');
                button.classList.add('bg-blue-100', 'text-blue-800', 'hover:bg-blue-200');
            } else {
                button.innerHTML = '<i class="far fa-heart"></i> Add to Watchlist';
                button.classList.remove('bg-blue-100', 'text-blue-800', 'hover:bg-blue-200');
                button.classList.add('bg-gray-100', 'text-gray-800', 'hover:bg-gray-200');
            }
        });
    }

    /**
     * Show a notification message
     * @param {string} message - Message to display
     */
    showNotification(message) {
        const notification = document.createElement('div');
        notification.className = 'fixed bottom-4 right-4 bg-green-500 text-white px-4 py-2 rounded-lg shadow-lg flex items-center space-x-2 z-50';
        notification.innerHTML = `
            <i class="fas fa-check-circle"></i>
            <span>${message}</span>
        `;
        
        document.body.appendChild(notification);
        
        // Auto-remove notification after 3 seconds
        setTimeout(() => {
            notification.classList.add('opacity-0', 'transition-opacity', 'duration-500');
            setTimeout(() => notification.remove(), 500);
        }, 3000);
    }

    /**
     * Set up event listeners for watchlist functionality
     */
    setupEventListeners() {
        // Toggle watchlist dropdown
        const watchlistButton = document.getElementById('watchlistButton');
        const watchlistDropdown = document.getElementById('watchlistDropdown');
        
        if (watchlistButton && watchlistDropdown) {
            watchlistButton.addEventListener('click', (e) => {
                e.stopPropagation();
                watchlistDropdown.classList.toggle('hidden');
            });
            
            // Close dropdown when clicking outside
            document.addEventListener('click', (e) => {
                if (!watchlistButton.contains(e.target) && !watchlistDropdown.contains(e.target)) {
                    watchlistDropdown.classList.add('hidden');
                }
            });
        }
        
        // Add to watchlist button
        document.addEventListener('click', (e) => {
            const addToWatchlistBtn = e.target.closest('[data-watchlist-ticker]');
            if (addToWatchlistBtn) {
                e.preventDefault();
                const ticker = addToWatchlistBtn.getAttribute('data-watchlist-ticker');
                if (this.has(ticker)) {
                    this.remove(ticker);
                } else {
                    this.add(ticker);
                }
            }
        });
    }
}
