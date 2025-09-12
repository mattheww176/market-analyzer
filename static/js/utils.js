/**
 * Format price with proper decimal places
 * @param {number} price - Price to format
 * @returns {string} Formatted price
 */
function formatPrice(price) {
    if (price === null || price === undefined) return 'N/A';
    
    // Convert to number if it's a string
    const num = typeof price === 'string' ? parseFloat(price) : price;
    
    // Handle invalid numbers
    if (isNaN(num)) return 'N/A';
    
    // Format based on the value
    if (num >= 1e9) {
        return `$${(num / 1e9).toFixed(2)}B`;
    } else if (num >= 1e6) {
        return `$${(num / 1e6).toFixed(2)}M`;
    } else if (num >= 1e3) {
        return `$${(num / 1e3).toFixed(2)}K`;
    } else if (num >= 1) {
        return `$${num.toFixed(2)}`;
    } else if (num >= 0.01) {
        return `$${num.toFixed(4)}`;
    } else {
        return `$${num.toFixed(8)}`;
    }
}

/**
 * Format percentage values
 * @param {number} value - Percentage value (0-100)
 * @param {number} decimals - Number of decimal places
 * @returns {string} Formatted percentage
 */
function formatPercentage(value, decimals = 2) {
    if (value === null || value === undefined) return 'N/A';
    const num = typeof value === 'string' ? parseFloat(value) : value;
    if (isNaN(num)) return 'N/A';
    return `${num >= 0 ? '+' : ''}${num.toFixed(decimals)}%`;
}

/**
 * Format a date string to a more readable format
 * @param {string|Date} date - Date to format
 * @param {boolean} includeTime - Whether to include time in the output
 * @returns {string} Formatted date string
 */
function formatDate(date, includeTime = false) {
    if (!date) return 'N/A';
    
    const d = new Date(date);
    if (isNaN(d.getTime())) return 'Invalid Date';
    
    const options = { 
        year: 'numeric', 
        month: 'short', 
        day: 'numeric',
        hour12: true
    };
    
    if (includeTime) {
        options.hour = '2-digit';
        options.minute = '2-digit';
    }
    
    return d.toLocaleDateString('en-US', options);
}

/**
 * Debounce a function
 * @param {Function} func - Function to debounce
 * @param {number} wait - Wait time in milliseconds
 * @returns {Function} Debounced function
 */
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

/**
 * Throttle a function
 * @param {Function} func - Function to throttle
 * @param {number} limit - Time limit in milliseconds
 * @returns {Function} Throttled function
 */
function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

/**
 * Check if the current device is mobile
 * @returns {boolean} True if the device is mobile
 */
function isMobile() {
    return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
}

/**
 * Copy text to clipboard
 * @param {string} text - Text to copy
 * @returns {Promise<boolean>} True if successful
 */
async function copyToClipboard(text) {
    try {
        await navigator.clipboard.writeText(text);
        return true;
    } catch (err) {
        console.error('Failed to copy text: ', err);
        return false;
    }
}

/**
 * Generate a random ID
 * @param {number} length - Length of the ID
 * @returns {string} Random ID
 */
function generateId(length = 8) {
    return Math.random().toString(36).substring(2, 2 + length);
}

/**
 * Format a number with commas as thousand separators
 * @param {number} num - Number to format
 * @returns {string} Formatted number
 */
function numberWithCommas(num) {
    if (num === null || num === undefined) return '0';
    const n = typeof num === 'string' ? parseFloat(num) : num;
    if (isNaN(n)) return '0';
    return n.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

/**
 * Truncate text to a specified length
 * @param {string} text - Text to truncate
 * @param {number} maxLength - Maximum length
 * @param {string} ellipsis - Ellipsis character(s)
 * @returns {string} Truncated text
 */
function truncateText(text, maxLength = 100, ellipsis = '...') {
    if (!text) return '';
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + ellipsis;
}

/**
 * Convert a string to title case
 * @param {string} str - String to convert
 * @returns {string} Title-cased string
 */
function toTitleCase(str) {
    if (!str) return '';
    return str.replace(/\w\S*/g, function(txt) {
        return txt.charAt(0).toUpperCase() + txt.substr(1).toLowerCase();
    });
}

/**
 * Check if a string is a valid URL
 * @param {string} str - String to check
 * @returns {boolean} True if the string is a valid URL
 */
function isValidUrl(str) {
    try {
        new URL(str);
        return true;
    } catch (_) {
        return false;
    }
}

/**
 * Get a cookie by name
 * @param {string} name - Cookie name
 * @returns {string|null} Cookie value or null if not found
 */
function getCookie(name) {
    const value = `; ${document.cookie}`;
    const parts = value.split(`; ${name}=`);
    if (parts.length === 2) return parts.pop().split(';').shift();
    return null;
}

/**
 * Set a cookie
 * @param {string} name - Cookie name
 * @param {string} value - Cookie value
 * @param {number} days - Expiration in days
 */
function setCookie(name, value, days = 30) {
    const date = new Date();
    date.setTime(date.getTime() + (days * 24 * 60 * 60 * 1000));
    const expires = `expires=${date.toUTCString()}`;
    document.cookie = `${name}=${value};${expires};path=/`;
}

/**
 * Remove a cookie
 * @param {string} name - Cookie name
 */
function removeCookie(name) {
    document.cookie = `${name}=;expires=Thu, 01 Jan 1970 00:00:00 GMT;path=/`;
}

/**
 * Format volume numbers with appropriate units
 * @param {number} volume - Volume to format
 * @returns {string} Formatted volume
 */
function formatVolume(volume) {
    if (volume === null || volume === undefined) return 'N/A';
    
    const num = typeof volume === 'string' ? parseFloat(volume) : volume;
    if (isNaN(num)) return 'N/A';
    
    if (num >= 1e9) {
        return `${(num / 1e9).toFixed(2)}B`;
    } else if (num >= 1e6) {
        return `${(num / 1e6).toFixed(2)}M`;
    } else if (num >= 1e3) {
        return `${(num / 1e3).toFixed(2)}K`;
    } else {
        return num.toLocaleString();
    }
}
