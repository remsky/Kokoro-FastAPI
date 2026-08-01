import AudioService from './services/AudioService.js';
import VoiceService from './services/VoiceService.js';
import PlayerState from './state/PlayerState.js';
import PlayerControls from './components/PlayerControls.js';
import VoiceSelector from './components/VoiceSelector.js';
import WaveVisualizer from './components/WaveVisualizer.js';
import TextEditor from './components/TextEditor.js';
import config from './config.js';

export class App {
    constructor() {
        this.elements = {
            generateBtn: document.getElementById('generate-btn'),
            generateBtnText: document.querySelector('#generate-btn .btn-text'),
            generateBtnLoader: document.querySelector('#generate-btn .loader'),
            downloadBtn: document.getElementById('download-btn'),
            autoplayToggle: document.getElementById('autoplay-toggle'),
            formatSelect: document.getElementById('format-select'),
            status: document.getElementById('status'),
            cancelBtn: document.getElementById('cancel-btn'),
            streamingNotice: document.getElementById('streaming-notice'),
            charCount: document.getElementById('char-count'),
            cup: document.querySelector('.logo-container .cup')
        };

        this.initialize();
    }

    async initialize() {
        // Initialize services and state
        this.playerState = new PlayerState();
        this.audioService = new AudioService();
        this.voiceService = new VoiceService();

        this.renderVersionBadge();
        this.renderStarBadge();

        // Initialize components
        this.playerControls = new PlayerControls(this.audioService, this.playerState);
        this.voiceSelector = new VoiceSelector(this.voiceService);
        this.waveVisualizer = new WaveVisualizer(this.playerState);
        
        // counter lives outside the component, in the editor status row
        const editorContainer = document.getElementById('text-editor');
        this.textEditor = new TextEditor(editorContainer, {
            linesPerPage: 20,
            onTextChange: (text) => {
                this.elements.charCount.textContent = `${text.length} characters`;
            }
        });

        // Initialize voice selector
        const voicesLoaded = await this.voiceSelector.initialize();
        if (!voicesLoaded) {
            this.showStatus('Failed to load voices', 'error');
            this.elements.generateBtn.disabled = true;
            return;
        }

        this.setupEventListeners();
        this.setupAudioEvents();
        this.applyBrowserStreamingNotice();
    }

    async renderStarBadge() {
        const count = document.getElementById('gh-star-count');
        if (!count) return;
        try {
            const response = await fetch('https://api.github.com/repos/remsky/Kokoro-FastAPI');
            if (!response.ok) return;
            const stars = (await response.json()).stargazers_count;
            if (typeof stars !== 'number') return;
            count.textContent = stars >= 1000 ? `${(stars / 1000).toFixed(1).replace(/\.0$/, '')}k` : `${stars}`;
            count.hidden = false;
        } catch (_) {
            // leave hidden on failure
        }
    }

    async renderVersionBadge() {
        const badge = document.getElementById('version-badge');
        if (!badge) return;
        try {
            await config.ensureInitialized();
            if (config.version) {
                badge.textContent = `v${config.version}`;
                badge.hidden = false;
            }
        } catch (_) {
            // leave hidden on failure
        }
    }

    applyBrowserStreamingNotice() {
        const notice = this.elements.streamingNotice;
        if (!notice) {
            return;
        }
        const format = this.elements.formatSelect?.value || 'mp3';
        const formatLabel = format.toUpperCase();
        const isFirefox = /Firefox\//.test(navigator.userAgent);
        let message = '';

        if (format === 'pcm') {
            message = 'PCM may not play in-browser; download still works.';
        } else if (format !== 'mp3') {
            message = `${formatLabel} plays/downloads once generation finishes.`;
        } else if (!this.audioService.supportsMSEMp3()) {
            message = isFirefox
                ? 'No streaming in Firefox; playback/download ready once generation finishes.'
                : 'Streaming may be unsupported here; playback/download ready once generation finishes.';
        } else if (this.elements.autoplayToggle?.checked) {
            message = 'Auto-play on; pause when done for full seek.';
        }

        notice.textContent = message;
        notice.hidden = !message;
    }

    setupEventListeners() {
        // Generate button
        this.elements.generateBtn.addEventListener('click', () => this.generateSpeech());

        // Download button (div with role=button, so handle keyboard activation too)
        this.elements.downloadBtn.addEventListener('click', () => this.downloadAudio());
        this.elements.downloadBtn.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                this.downloadAudio();
            }
        });

        // Keep browser/output warning aligned with the selected format and autoplay state
        this.elements.formatSelect.addEventListener('change', () => this.applyBrowserStreamingNotice());
        this.elements.autoplayToggle.addEventListener('change', () => this.applyBrowserStreamingNotice());

        // Cancel button
        this.elements.cancelBtn.addEventListener('click', () => {
            this.audioService.cancel();
            this.setGenerating(false);
            this.elements.downloadBtn.classList.remove('ready');
            this.showStatus('Generation cancelled', 'info');
        });

        // Handle page unload
        window.addEventListener('beforeunload', () => {
            this.audioService.cleanup();
            this.playerControls.cleanup();
            this.waveVisualizer.cleanup();
        });
    }

    setupAudioEvents() {
        // Handle download button visibility
        this.audioService.addEventListener('downloadReady', () => {
            this.elements.downloadBtn.classList.add('ready');
        });

        // Handle buffer errors
        this.audioService.addEventListener('bufferError', () => {
            this.showStatus('Processing... (Download will be available when complete)', 'info');
        });

        // Handle completion
        this.audioService.addEventListener('complete', () => {
            this.setGenerating(false);
            
            // Show preparing status
            this.showStatus('Preparing file...', 'info');

            // Flash the coffee cup
            this.elements.cup.classList.add('done');
        });

        // Handle download ready
        this.audioService.addEventListener('downloadReady', () => {
            setTimeout(() => {
                if (!this._playbackFailed) {
                    this.showStatus('Generation complete', 'success');
                }
            }, 500); // Small delay to ensure "Preparing file..." is visible
        });

        // Handle audio end
        this.audioService.addEventListener('ended', () => {
            this.playerState.setPlaying(false);
        });

        // Handle errors
        this.audioService.addEventListener('error', (error) => {
            this.showStatus('Error: ' + error.message, 'error');
            this.setGenerating(false);
            this.elements.downloadBtn.style.display = 'none';
        });

        // Block-mode playback failure: file is still available for download
        this.audioService.addEventListener('playbackUnavailable', () => {
            this._playbackFailed = true;
            this.showStatus(
                'Playback unavailable in this browser. Use the download below.',
                'info'
            );
        });
    }

    showStatus(message, type = 'info') {
        this.elements.status.textContent = message;
        this.elements.status.className = 'status ' + type;
        // an uncleared timer from an earlier status would blank this one early
        clearTimeout(this._statusTimer);
        this._statusTimer = setTimeout(() => {
            this.elements.status.className = 'status';
        }, 5000);
    }

    setGenerating(isGenerating) {
        this.playerState.setGenerating(isGenerating);
        this.elements.generateBtn.disabled = isGenerating;
        this.elements.generateBtn.classList.toggle('loading', isGenerating);
        this.elements.generateBtnLoader.style.display = isGenerating ? 'block' : 'none';
        this.elements.generateBtnText.style.visibility = isGenerating ? 'hidden' : 'visible';
        this.elements.cancelBtn.style.display = isGenerating ? 'block' : 'none';
        this.elements.cup.classList.toggle('brewing', isGenerating);
        if (isGenerating) {
            this.elements.cup.classList.remove('done');
        }
    }

    validateInput() {
        const text = this.textEditor.getText().trim();
        if (!text) {
            this.showStatus('Please enter some text', 'error');
            return false;
        }
        
        if (!this.voiceService.hasSelectedVoices()) {
            this.showStatus('Please select a voice', 'error');
            return false;
        }
        
        return true;
    }

    async generateSpeech() {
        // Don't check isGenerating state since we want to allow generation after cancel
        if (!this.validateInput()) {
            return;
        }

        const text = this.textEditor.getText().trim();
        const voice = this.voiceService.getSelectedVoiceString();
        const speed = this.playerState.getState().speed;

        this.playerState.setReady(false);
        this.playerState.setPlaying(false);
        this.playerState.setTime(0, 0);
        this.setGenerating(true);
        this._playbackFailed = false;
        this.elements.downloadBtn.classList.remove('ready');

        // Just reset progress bar, don't do full cleanup
        this.waveVisualizer.updateProgress(0, 1);
        
        try {
            console.log('Starting audio generation...', { chars: text.length, voice, speed });

            if (!text || !voice) {
                console.error('Invalid input:', { text, voice, speed });
                throw new Error('Invalid input parameters');
            }
            
            await this.audioService.streamAudio(
                text,
                voice,
                speed,
                (loaded, total) => this.waveVisualizer.updateProgress(loaded, total)
            );
        } catch (error) {
            console.error('Generation error:', error);
            if (error.name !== 'AbortError') {
                this.showStatus('Error generating speech: ' + error.message, 'error');
                this.setGenerating(false);
            }
        }
    }

    downloadAudio() {
        const downloadUrl = this.audioService.getDownloadUrl();
        if (!downloadUrl) {
            console.warn('No download URL available');
            return;
        }

        console.log('Starting download from:', downloadUrl);
        
        const format = this.elements.formatSelect.value;
        const voice = this.voiceService.getSelectedVoiceString();
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        
        const a = document.createElement('a');
        a.href = downloadUrl;
        a.download = `${voice}_${timestamp}.${format}`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new App();
});
