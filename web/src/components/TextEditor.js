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
        this.isTyping = false;
        
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
                        <div class="pagination">
                            <button class="prev-btn">← Previous</button>
                            <span class="page-info">Page 1 of 1</span>
                            <button class="next-btn">Next →</button>
                        </div>
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
            pageInfo: this.container.querySelector('.page-info'),
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
            
            // Update full text - join with space since pages are just for UI
            this.fullText = this.pages.join(' ');

            if (this.options.onTextChange) {
                this.options.onTextChange(this.fullText);
            }
        });

        // Navigation
        this.elements.prevBtn.addEventListener('click', () => {
            if (this.currentPage > 1) {
                this.currentPage--;
                this.updatePageDisplay();
            }
        });

        this.elements.nextBtn.addEventListener('click', () => {
            if (this.currentPage < this.pages.length) {
                this.currentPage++;
                this.updatePageDisplay();
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

        // Store original text to preserve natural line breaks
        this.fullText = text.trim();
        const words = text.trim().split(/\s+/);
        this.pages = [];
        let currentPage = '';
        
        for (let i = 0; i < words.length; i++) {
            const word = words[i];
            const potentialPage = currentPage + (currentPage ? ' ' : '') + word;
            
            if (potentialPage.length >= this.options.charsPerPage && currentPage) {
                this.pages.push(currentPage);
                currentPage = word;
            } else {
                currentPage = potentialPage;
            }
        }
        
        if (currentPage) {
            this.pages.push(currentPage);
        }
        
        if (this.pages.length === 0) {
            this.pages = [''];
            this.currentPage = 1;
        } else {
            // Keep current page in bounds
            this.currentPage = Math.min(this.currentPage, this.pages.length);
        }

        this.updatePageDisplay();
    }

    setText(text) {
        // Just set the text without splitting into pages
        this.fullText = text;
        this.pages = [text];
        this.currentPage = 1;
        this.updatePageDisplay();
    }

    updatePageDisplay() {
        this.elements.pageContent.value = this.pages[this.currentPage - 1] || '';
        this.elements.pageInfo.textContent = `Page ${this.currentPage} of ${this.pages.length}`;
        
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

    getText() {
        return this.fullText;
    }

    clear() {
        this.setText('');
    }
}