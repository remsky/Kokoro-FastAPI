import { config } from '../config.js';

export class VoiceService {
    constructor() {
        this.availableVoices = [];
        this.selectedVoices = new Map(); // Changed to Map to store voice:weight pairs
    }

    async loadVoices() {
        try {
            const apiUrl = await config.getApiUrl('/v1/audio/voices');
            const response = await fetch(apiUrl);
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail?.message || 'Failed to load voices');
            }
            
            const data = await response.json();
            if (!data.voices?.length) {
                throw new Error('No voices available');
            }

            this.availableVoices = data.voices;
            
            // Select first voice if none selected
            if (this.selectedVoices.size === 0) {
                const firstVoice = this.availableVoices.find(voice => voice && voice.trim());
                if (firstVoice) {
                    this.addVoice(firstVoice);
                }
            }

            return this.availableVoices;
        } catch (error) {
            console.error('Failed to load voices:', error);
            throw error;
        }
    }

    getAvailableVoices() {
        return this.availableVoices;
    }

    getSelectedVoices() {
        return Array.from(this.selectedVoices.keys());
    }

    getSelectedVoiceWeights() {
        return Array.from(this.selectedVoices.entries()).map(([voice, weight]) => ({
            voice,
            weight
        }));
    }

    getSelectedVoiceString() {
        const entries = Array.from(this.selectedVoices.entries());
        
        // If only one voice with weight 1, return just the voice name
        if (entries.length === 1 && entries[0][1] === 1) {
            return entries[0][0];
        }
        
        // Otherwise return voice(weight) format
        return entries
            .map(([voice, weight]) => `${voice}(${weight})`)
            .join('+');
    }

    addVoice(voice, weight = 1) {
        if (this.availableVoices.includes(voice)) {
            this.selectedVoices.set(voice, parseFloat(weight) || 1);
            return true;
        }
        return false;
    }

    updateWeight(voice, weight) {
        if (this.selectedVoices.has(voice)) {
            this.selectedVoices.set(voice, parseFloat(weight) || 1);
            return true;
        }
        return false;
    }

    removeVoice(voice) {
        return this.selectedVoices.delete(voice);
    }

    clearSelectedVoices() {
        this.selectedVoices.clear();
    }

    filterVoices(searchTerm) {
        if (!searchTerm) {
            return this.availableVoices;
        }
        
        const term = searchTerm.toLowerCase();
        return this.availableVoices.filter(voice => 
            voice.toLowerCase().includes(term)
        );
    }

    hasSelectedVoices() {
        return this.selectedVoices.size > 0;
    }
}

export default VoiceService;