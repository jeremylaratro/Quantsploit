/**
 * Quantsploit Dashboard - DOM Utilities
 * Safe DOM manipulation functions to prevent XSS vulnerabilities
 * All functions use textContent or DOM APIs instead of raw HTML insertion
 */

(function(window) {
    'use strict';

    /**
     * Escape HTML special characters to prevent XSS
     * @param {string} unsafe - Untrusted string input
     * @returns {string} HTML-escaped string
     */
    function escapeHtml(unsafe) {
        if (unsafe === null || unsafe === undefined) {
            return '';
        }
        return String(unsafe)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }

    /**
     * Create a text node safely (automatically escapes HTML)
     * @param {string} text - Text content
     * @returns {Text} Text node
     */
    function createTextNode(text) {
        return document.createTextNode(text || '');
    }

    /**
     * Set text content safely (no HTML interpretation)
     * @param {Element} element - DOM element
     * @param {string} text - Text content
     */
    function setText(element, text) {
        if (element) {
            element.textContent = text || '';
        }
    }

    /**
     * Create an element with safe text content
     * @param {string} tagName - HTML tag name
     * @param {string} text - Text content (will not be interpreted as HTML)
     * @param {Object} attributes - Optional attributes to set
     * @returns {Element} Created element
     */
    function createElement(tagName, text, attributes) {
        const el = document.createElement(tagName);
        if (text !== null && text !== undefined) {
            el.textContent = text;
        }
        if (attributes) {
            for (const [key, value] of Object.entries(attributes)) {
                if (key === 'className') {
                    el.className = value;
                } else if (key === 'dataset') {
                    for (const [dataKey, dataValue] of Object.entries(value)) {
                        el.dataset[dataKey] = dataValue;
                    }
                } else if (key.startsWith('on')) {
                    // Event handlers - only allow if value is a function
                    if (typeof value === 'function') {
                        el.addEventListener(key.slice(2).toLowerCase(), value);
                    }
                } else {
                    el.setAttribute(key, value);
                }
            }
        }
        return el;
    }

    /**
     * Create a table row with safe cell contents
     * @param {Array} cells - Array of cell contents (strings will be escaped)
     * @param {Object} options - Optional settings (className, onClick)
     * @returns {HTMLTableRowElement} Created row
     */
    function createTableRow(cells, options) {
        const row = document.createElement('tr');
        options = options || {};

        if (options.className) {
            row.className = options.className;
        }

        cells.forEach(function(cell) {
            const td = document.createElement('td');

            if (typeof cell === 'string' || typeof cell === 'number') {
                td.textContent = cell;
            } else if (cell instanceof HTMLElement) {
                td.appendChild(cell);
            } else if (cell && typeof cell === 'object') {
                // Object with content and attributes
                td.textContent = cell.text || '';
                if (cell.className) {
                    td.className = cell.className;
                }
            }

            row.appendChild(td);
        });

        if (options.onClick && typeof options.onClick === 'function') {
            row.style.cursor = 'pointer';
            row.addEventListener('click', options.onClick);
        }

        return row;
    }

    /**
     * Create a badge/label element with safe text
     * @param {string} text - Badge text
     * @param {string} variant - Bootstrap variant (primary, success, danger, etc.)
     * @returns {HTMLSpanElement} Badge element
     */
    function createBadge(text, variant) {
        const span = document.createElement('span');
        span.className = 'badge bg-' + (variant || 'secondary');
        span.textContent = text || '';
        return span;
    }

    /**
     * Create an anchor/link element with safe text
     * @param {string} href - Link URL
     * @param {string} text - Link text
     * @param {Object} options - Optional settings (target, className, onClick)
     * @returns {HTMLAnchorElement} Anchor element
     */
    function createLink(href, text, options) {
        const a = document.createElement('a');
        options = options || {};

        a.href = href || '#';
        a.textContent = text || '';

        if (options.target) {
            a.target = options.target;
        }
        if (options.className) {
            a.className = options.className;
        }
        if (options.onClick && typeof options.onClick === 'function') {
            a.addEventListener('click', function(e) {
                e.preventDefault();
                options.onClick(e);
            });
        }

        return a;
    }

    /**
     * Safely update an element's content with escaped text
     * @param {string} selector - CSS selector
     * @param {string} text - Text content
     */
    function updateText(selector, text) {
        const el = document.querySelector(selector);
        if (el) {
            el.textContent = text || '';
        }
    }

    /**
     * Append multiple children to a parent element
     * @param {Element} parent - Parent element
     * @param {Array} children - Array of child elements
     */
    function appendChildren(parent, children) {
        if (parent && Array.isArray(children)) {
            children.forEach(function(child) {
                if (child instanceof Node) {
                    parent.appendChild(child);
                }
            });
        }
    }

    /**
     * Clear all children from an element
     * @param {Element} element - Element to clear
     */
    function clearElement(element) {
        if (element) {
            while (element.firstChild) {
                element.removeChild(element.firstChild);
            }
        }
    }

    /**
     * Format a number with commas for display
     * @param {number} num - Number to format
     * @param {number} decimals - Decimal places (default 2)
     * @returns {string} Formatted number
     */
    function formatNumber(num, decimals) {
        if (num === null || num === undefined || isNaN(num)) {
            return '-';
        }
        decimals = typeof decimals === 'number' ? decimals : 2;
        return Number(num).toLocaleString('en-US', {
            minimumFractionDigits: decimals,
            maximumFractionDigits: decimals
        });
    }

    /**
     * Format a percentage for display
     * @param {number} num - Number (0.15 = 15%)
     * @param {number} decimals - Decimal places (default 2)
     * @returns {string} Formatted percentage
     */
    function formatPercent(num, decimals) {
        if (num === null || num === undefined || isNaN(num)) {
            return '-';
        }
        decimals = typeof decimals === 'number' ? decimals : 2;
        return (Number(num) * 100).toFixed(decimals) + '%';
    }

    /**
     * Get CSRF token from page
     * @returns {string} CSRF token value
     */
    function getCsrfToken() {
        const meta = document.querySelector('meta[name="csrf-token"]');
        if (meta) {
            return meta.getAttribute('content');
        }
        const input = document.querySelector('input[name="csrf_token"]');
        if (input) {
            return input.value;
        }
        return '';
    }

    /**
     * Make a fetch request with CSRF token included
     * @param {string} url - Request URL
     * @param {Object} options - Fetch options
     * @returns {Promise} Fetch promise
     */
    function secureFetch(url, options) {
        options = options || {};
        options.headers = options.headers || {};

        // Add CSRF token for non-GET requests
        if (options.method && options.method.toUpperCase() !== 'GET') {
            options.headers['X-CSRF-Token'] = getCsrfToken();
        }

        return fetch(url, options);
    }

    // Export to global namespace
    window.DOMUtils = {
        escapeHtml: escapeHtml,
        createTextNode: createTextNode,
        setText: setText,
        createElement: createElement,
        createTableRow: createTableRow,
        createBadge: createBadge,
        createLink: createLink,
        updateText: updateText,
        appendChildren: appendChildren,
        clearElement: clearElement,
        formatNumber: formatNumber,
        formatPercent: formatPercent,
        getCsrfToken: getCsrfToken,
        secureFetch: secureFetch
    };

    // Also export escapeHtml directly for convenience
    window.escapeHtml = escapeHtml;

})(window);
