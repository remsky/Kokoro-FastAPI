export default class TextEditor {
    constructor(container, options = {}) {
        this.options = {
            charsPerPage: 500,  // Default to 500 chars per page
            onTextChange: null,
            ...options
        };
        
        this.container = container;
        this.currentPage = 1;
        this.pages = [''];
        this.charCount = 0;
        this.fullText = '';
        this.findFrom = 0;
        
        this.setupDOM();
        this.bindEvents();
        // sync nav button disabled state so CSS can hide the row while single-page
        this.updatePageDisplay();
    }

    setupDOM() {
        this.container.innerHTML = `
            <div class="text-editor">
                <div class="editor-view">
                    <div class="page-navigation">
                        <button type="button" id="read-along-btn" class="read-along-btn" aria-pressed="false" title="Available when generation completes" disabled>Read along</button>
                        <div class="pagination">
                            <button class="prev-btn" aria-label="Previous page">←</button>
                            <span class="page-info">Page <input type="number" class="page-jump" min="1" value="1"> of <span class="page-total">1</span></span>
                            <button class="next-btn" aria-label="Next page">→</button>
                        </div>
                        <details class="find-menu">
                            <summary class="find-toggle" title="Find and replace" aria-label="Find and replace">
                                <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="11" cy="11" r="7"></circle><path d="m16.5 16.5 4.5 4.5"></path></svg>
                            </summary>
                            <div class="find-panel">
                                <input type="text" class="find-input" placeholder="Find">
                                <input type="text" class="replace-input" placeholder="Replace with">
                                <span class="find-count"></span>
                                <div class="find-actions">
                                    <button type="button" class="find-next-btn">Next</button>
                                    <button type="button" class="replace-one-btn">Replace</button>
                                    <button type="button" class="replace-all-btn">All</button>
                                </div>
                            </div>
                        </details>
                    </div>
                    <textarea
                        class="page-content"
                        placeholder="Enter text to convert to speech..."
                    ></textarea>
                    <div class="editor-footer">
                        <div class="file-controls">
                            <input type="file" class="file-input" accept=".txt" style="display: none;">
                            <button class="upload-btn">Upload Text</button>
                            <button class="clear-btn">Clear Text</button>
                        </div>
                        <div class="chars-per-page">
                            <input
                                type="number"
                                class="chars-input"
                                value="500"
                                min="100"
                                max="2000"
                                title="Characters per page"
                            >
                            <span class="chars-label">/page</span>
                            <button class="format-btn">Format</button>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Cache DOM elements
        this.elements = {
            pageContent: this.container.querySelector('.page-content'),
            prevBtn: this.container.querySelector('.prev-btn'),
            nextBtn: this.container.querySelector('.next-btn'),
            pageJump: this.container.querySelector('.page-jump'),
            pageTotal: this.container.querySelector('.page-total'),
            findMenu: this.container.querySelector('.find-menu'),
            findInput: this.container.querySelector('.find-input'),
            replaceInput: this.container.querySelector('.replace-input'),
            findCount: this.container.querySelector('.find-count'),
            findNextBtn: this.container.querySelector('.find-next-btn'),
            replaceOneBtn: this.container.querySelector('.replace-one-btn'),
            replaceAllBtn: this.container.querySelector('.replace-all-btn'),
            fileInput: this.container.querySelector('.file-input'),
            uploadBtn: this.container.querySelector('.upload-btn'),
            clearBtn: this.container.querySelector('.clear-btn'),
            charsPerPage: this.container.querySelector('.chars-input'),
            formatBtn: this.container.querySelector('.format-btn')
        };

        // Set initial chars per page value
        this.elements.charsPerPage.value = this.options.charsPerPage;
    }

    bindEvents() {
        // Handle page content changes
        this.elements.pageContent.addEventListener('input', (e) => {
            const newContent = e.target.value;
            this.pages[this.currentPage - 1] = newContent;
            
            // Only handle empty pages, otherwise just update the text
            if (!newContent.trim() && this.pages.length > 1) {
                // Remove the empty page and adjust
                this.pages.splice(this.currentPage - 1, 1);
                this.currentPage = Math.min(this.currentPage, this.pages.length);
                this.updatePageDisplay();
            }
            
            // pages are only a display split, join back into the full text
            this.fullText = this.pages.join(' ');

            if (this.options.onTextChange) {
                this.options.onTextChange(this.fullText);
            }
        });

        // Navigation
        this.elements.prevBtn.addEventListener('click', () => this.prevPage());
        this.elements.nextBtn.addEventListener('click', () => this.nextPage());
        this.elements.pageJump.addEventListener('change', (e) => {
            this.goToPage(parseInt(e.target.value, 10));
        });

        this.elements.findInput.addEventListener('input', () => {
            this.findFrom = 0;
            const matches = this.countMatches(this.elements.findInput.value);
            this.setFindCount(this.elements.findInput.value ? `${matches} match${matches === 1 ? '' : 'es'}` : '');
        });
        this.elements.findInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                this.findNext();
            }
        });
        this.elements.replaceInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                this.replaceOne();
            }
        });
        this.elements.findNextBtn.addEventListener('click', () => this.findNext());
        this.elements.replaceOneBtn.addEventListener('click', () => this.replaceOne());
        this.elements.replaceAllBtn.addEventListener('click', () => this.replaceAll());
        this.elements.findMenu.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.elements.findMenu.open = false;
            }
        });
        this.elements.findMenu.addEventListener('toggle', () => {
            if (this.elements.findMenu.open) {
                this.elements.findInput.focus();
            }
        });

        // File upload
        this.elements.uploadBtn.addEventListener('click', () => {
            this.elements.fileInput.click();
        });
        
        this.elements.fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (e) => {
                    this.setText(e.target.result);
                    if (this.options.onTextChange) {
                        this.options.onTextChange(this.getText());
                    }
                };
                reader.readAsText(file);
            }
        });

        // Clear text
        this.elements.clearBtn.addEventListener('click', () => {
            this.clear();
            if (this.options.onTextChange) {
                this.options.onTextChange('');
            }
        });

        // Cache format button
        this.elements.formatBtn = this.container.querySelector('.format-btn');

        // Characters per page control - just update the value
        this.elements.charsPerPage.addEventListener('change', (e) => {
            const value = parseInt(e.target.value);
            if (value >= 100 && value <= 2000) {
                this.options.charsPerPage = value;
            }
        });

        // Format pages button - trigger the split
        this.elements.formatBtn.addEventListener('click', () => {
            const value = parseInt(this.elements.charsPerPage.value);
            if (value >= 100 && value <= 2000) {
                this.options.charsPerPage = value;
                this.splitIntoPages(this.fullText);
                if (this.options.onTextChange) {
                    this.options.onTextChange(this.fullText);
                }
            }
        });
    }

    splitIntoPages(text) {
        if (!text || !text.trim()) {
            this.pages = [''];
            this.fullText = '';
            this.currentPage = 1;
            this.updatePageDisplay();
            return;
        }

        const tokens = text.trim().split(/(\s+)/);
        this.pages = [];
        let currentPage = '';

        for (const token of tokens) {
            const isWhitespace = /^\s+$/.test(token);
            const potentialPage = currentPage + token;

            if (potentialPage.length >= this.options.charsPerPage && currentPage && !isWhitespace) {
                this.pages.push(currentPage.trimEnd());
                currentPage = token;
            } else {
                currentPage = potentialPage;
            }
        }

        if (currentPage.trim()) {
            this.pages.push(currentPage.trimEnd());
        }
        
        if (this.pages.length === 0) {
            this.pages = [''];
            this.currentPage = 1;
        } else {
            // Keep current page in bounds
            this.currentPage = Math.min(this.currentPage, this.pages.length);
        }

        // fullText is always the pages joined with single spaces, so find offsets map exactly
        this.fullText = this.pages.join(' ');
        this.updatePageDisplay();
    }

    setText(text) {
        // Just set the text without splitting into pages
        this.fullText = text;
        this.pages = [text];
        this.currentPage = 1;
        this.updatePageDisplay();
    }

    getPageText() {
        return this.pages[this.currentPage - 1] || '';
    }

    getCursor() {
        return this.elements.pageContent.selectionStart ?? this.getPageText().length;
    }

    /** Writes the visible page back, keeping pages, fullText and the caret in step. */
    setPageText(text, cursor = null) {
        this.pages[this.currentPage - 1] = text;
        this.fullText = this.pages.join(' ');
        this.elements.pageContent.value = text;

        if (cursor !== null) {
            this.elements.pageContent.focus();
            this.elements.pageContent.setSelectionRange(cursor, cursor);
        }
        this.options.onTextChange?.(this.fullText);
    }

    /** Rewrites the whole document, keeping whichever page split is in use. */
    replaceText(text) {
        if (this.pages.length > 1) {
            this.splitIntoPages(text);
        } else {
            this.setText(text);
        }
        this.options.onTextChange?.(this.fullText);
    }

    updatePageDisplay() {
        this.elements.pageContent.value = this.pages[this.currentPage - 1] || '';
        this.elements.pageJump.value = this.currentPage;
        this.elements.pageJump.max = this.pages.length;
        this.elements.pageTotal.textContent = this.pages.length;

        // Update button states
        this.elements.prevBtn.disabled = this.currentPage === 1;
        this.elements.nextBtn.disabled = this.currentPage === this.pages.length;
    }

    prevPage() {
        if (this.currentPage > 1) {
            this.currentPage--;
            this.updatePageDisplay();
        }
    }

    nextPage() {
        if (this.currentPage < this.pages.length) {
            this.currentPage++;
            this.updatePageDisplay();
        }
    }

    goToPage(page) {
        if (Number.isInteger(page)) {
            this.currentPage = Math.max(1, Math.min(page, this.pages.length));
        }
        this.updatePageDisplay();
    }

    pageStart(pageIndex) {
        let start = 0;
        for (let i = 0; i < pageIndex; i++) {
            start += this.pages[i].length + 1;
        }
        return start;
    }

    setFindCount(message) {
        this.elements.findCount.textContent = message;
    }

    countMatches(term) {
        if (!term) {
            return 0;
        }
        const haystack = this.fullText.toLowerCase();
        const needle = term.toLowerCase();
        let count = 0;
        let index = haystack.indexOf(needle);
        while (index !== -1) {
            count++;
            index = haystack.indexOf(needle, index + needle.length);
        }
        return count;
    }

    findNext() {
        const term = this.elements.findInput.value;
        if (!term) {
            return;
        }

        const haystack = this.fullText.toLowerCase();
        const needle = term.toLowerCase();
        let index = haystack.indexOf(needle, this.findFrom);
        if (index === -1 && this.findFrom > 0) {
            index = haystack.indexOf(needle);
        }
        if (index === -1) {
            this.setFindCount('0 matches');
            return;
        }
        this.findFrom = index + needle.length;

        let position = 0;
        for (let scan = haystack.indexOf(needle); scan !== -1 && scan <= index; scan = haystack.indexOf(needle, scan + needle.length)) {
            position++;
        }
        this.setFindCount(`${position} of ${this.countMatches(term)}`);
        this.revealOffset(index, needle.length);
    }

    revealOffset(index, length = 0) {
        let page = 0;
        let offset = index;
        while (page < this.pages.length - 1 && offset > this.pages[page].length) {
            offset -= this.pages[page].length + 1;
            page++;
        }
        this.currentPage = page + 1;
        this.updatePageDisplay();
        const box = this.elements.pageContent;
        box.focus({ preventScroll: true });
        box.setSelectionRange(offset, Math.min(offset + length, this.pages[page].length));
        box.scrollTop = (box.scrollHeight * offset) / Math.max(1, box.value.length) - box.clientHeight / 2;
    }

    replaceOne() {
        const term = this.elements.findInput.value;
        if (!term) {
            return;
        }

        const { selectionStart, selectionEnd } = this.elements.pageContent;
        const selected = this.getPageText().slice(selectionStart, selectionEnd);
        // nothing selected yet means find first, replace on the next press
        if (selected.toLowerCase() !== term.toLowerCase()) {
            this.findNext();
            return;
        }

        const replacement = this.elements.replaceInput.value;
        const pageText = this.getPageText();
        this.setPageText(pageText.slice(0, selectionStart) + replacement + pageText.slice(selectionEnd));
        this.findFrom = this.pageStart(this.currentPage - 1) + selectionStart + replacement.length;
        this.findNext();
    }

    replaceAll() {
        const term = this.elements.findInput.value;
        if (!term) {
            return;
        }
        const count = this.countMatches(term);
        if (!count) {
            this.setFindCount('0 matches');
            return;
        }

        const replacement = this.elements.replaceInput.value;
        const pattern = new RegExp(term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi');
        this.replaceText(this.fullText.replace(pattern, () => replacement));
        this.findFrom = 0;
        this.setFindCount(`${count} replaced`);
    }

    getText() {
        return this.fullText;
    }

    clear() {
        this.setText('');
    }
}