export class VoiceSelector {
    constructor(voiceService) {
        this.voiceService = voiceService;
        this.tagMode = false;
        this.onInsertTag = null;
        this.elements = {
            voiceSearch: document.getElementById('voice-search'),
            voiceDropdown: document.getElementById('voice-dropdown'),
            voiceOptions: document.getElementById('voice-options'),
            selectedVoices: document.getElementById('selected-voices')
        };

        this.setupEventListeners();
    }

    /**
     * In tag mode the list becomes a cast to insert from, so a click writes a tag
     * instead of changing the selection. The selection stays live either way: it is
     * still the voice for anything ahead of the first tag.
     */
    setTagMode(enabled, onInsertTag = null) {
        this.tagMode = enabled;
        this.onInsertTag = onInsertTag;
        this.elements.selectedVoices.classList.toggle('as-cast', enabled);
        this.updateSearchPlaceholder();
    }

    setupEventListeners() {
        // Voice search focus
        this.elements.voiceSearch.addEventListener('focus', () => {
            this.elements.voiceDropdown.classList.add('show');
        });

        // Voice search
        this.elements.voiceSearch.addEventListener('input', (e) => {
            const filteredVoices = this.voiceService.filterVoices(e.target.value);
            this.renderVoiceOptions(filteredVoices);
        });

        // Voice selection - handle clicks on the entire voice option
        this.elements.voiceOptions.addEventListener('mousedown', (e) => {
            e.preventDefault(); // Prevent blur on search input
            
            const voiceOption = e.target.closest('.voice-option');
            if (!voiceOption) return;
            
            const voice = voiceOption.dataset.voice;
            if (!voice) return;

            if (this.tagMode) {
                // hand focus to the editor where the tag just landed
                this.elements.voiceDropdown.classList.remove('show');
                this.onInsertTag?.(voice);
                return;
            }

            const isSelected = voiceOption.classList.contains('selected');
            
            if (!isSelected) {
                this.voiceService.addVoice(voice);
            } else {
                this.voiceService.removeVoice(voice);
            }
            
            voiceOption.classList.toggle('selected');
            this.updateSelectedVoicesDisplay();
            
            // Keep focus on search input
            requestAnimationFrame(() => {
                this.elements.voiceSearch.focus();
            });
        });

        // Weight adjustment
        this.elements.selectedVoices.addEventListener('input', (e) => {
            if (e.target.type === 'number') {
                const voice = e.target.dataset.voice;
                let weight = parseFloat(e.target.value);
                
                // Ensure weight is between 0.1 and 10
                weight = Math.max(0.1, Math.min(10, weight));
                e.target.value = weight;
                
                this.voiceService.updateWeight(voice, weight);
            }
        });

        // Remove selected voice
        this.elements.selectedVoices.addEventListener('click', (e) => {
            if (e.target.classList.contains('remove-voice')) {
                e.preventDefault();
                e.stopPropagation();
                const voice = e.target.dataset.voice;
                this.voiceService.removeVoice(voice);
                this.updateVoiceOptionState(voice, false);
                this.updateSelectedVoicesDisplay();
            }
        });

        // Handle clicks outside to close dropdown
        document.addEventListener('mousedown', (e) => {
            // Don't handle clicks in selected voices area
            if (this.elements.selectedVoices.contains(e.target)) {
                return;
            }
            
            // Don't close if clicking in search or dropdown
            if (this.elements.voiceSearch.contains(e.target) || 
                this.elements.voiceDropdown.contains(e.target)) {
                return;
            }
            
            this.elements.voiceDropdown.classList.remove('show');
            this.elements.voiceSearch.blur();
        });

        this.elements.voiceSearch.addEventListener('blur', () => {
            if (!this.elements.voiceSearch.value) {
                this.updateSearchPlaceholder();
            }
        });
    }

    renderVoiceOptions(voices) {
        this.elements.voiceOptions.innerHTML = voices
            .map(voice => `
                <div class="voice-option ${this.voiceService.getSelectedVoices().includes(voice) ? 'selected' : ''}" 
                     data-voice="${voice}">
                    ${voice}
                </div>
            `)
            .join('');
    }

    updateSelectedVoicesDisplay() {
        const selectedVoices = this.voiceService.getSelectedVoiceWeights();
        this.elements.selectedVoices.innerHTML = selectedVoices
            .map(({voice, weight}) => `
                <span class="selected-voice-tag">
                    <span class="voice-name">${voice}</span>
                    <span class="voice-weight">
                        <input type="number" 
                               value="${weight}" 
                               min="0.1" 
                               max="10" 
                               step="0.1" 
                               data-voice="${voice}"
                               class="weight-input"
                               title="Voice weight (0.1 to 10)">
                    </span>
                    <span class="remove-voice" data-voice="${voice}" title="Remove voice">×</span>
                </span>
            `)
            .join('');
        
        this.updateSearchPlaceholder();
    }

    updateSearchPlaceholder() {
        if (this.tagMode) {
            this.elements.voiceSearch.placeholder = 'Search voices to insert...';
            return;
        }

        const hasSelected = this.voiceService.hasSelectedVoices();
        this.elements.voiceSearch.placeholder = hasSelected ?
            'Search voices...' :
            'Search and select voices...';
    }

    updateVoiceOptionState(voice, selected) {
        const voiceOption = this.elements.voiceOptions
            .querySelector(`[data-voice="${voice}"]`);
        if (voiceOption) {
            voiceOption.classList.toggle('selected', selected);
        }
    }

    async initialize() {
        try {
            await this.voiceService.loadVoices();
            this.renderVoiceOptions(this.voiceService.getAvailableVoices());
            this.updateSelectedVoicesDisplay();
            return true;
        } catch (error) {
            console.error('Failed to initialize voice selector:', error);
            return false;
        }
    }
}

export default VoiceSelector;