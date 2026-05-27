export class PlayerControls {
    constructor(audioService, playerState) {
        this.audioService = audioService;
        this.playerState = playerState;
        this.elements = {
            playPauseBtn: document.getElementById('play-pause-btn'),
            seekSlider: document.getElementById('seek-slider'),
            volumeSlider: document.getElementById('volume-slider'),
            speedSlider: document.getElementById('speed-slider'),
            speedValue: document.getElementById('speed-value'),
            timeDisplay: document.getElementById('time-display'),
            cancelBtn: document.getElementById('cancel-btn')
        };
        
        this.setupEventListeners();
        this.setupAudioEvents();
        this.setupStateSubscription();
        this.timeUpdateInterval = null;
    }

    formatTime(secs) {
        if (!Number.isFinite(secs) || secs < 0) {
            return '0:00';
        }

        const minutes = Math.floor(secs / 60);
        const seconds = Math.floor(secs % 60);
        return `${minutes}:${seconds.toString().padStart(2, '0')}`;
    }

    startTimeUpdate() {
        this.stopTimeUpdate(); // Clear any existing interval
        this.timeUpdateInterval = setInterval(() => {
            this.updateTimeDisplay();
        }, 100); // Update every 100ms for smooth tracking
    }

    stopTimeUpdate() {
        if (this.timeUpdateInterval) {
            clearInterval(this.timeUpdateInterval);
            this.timeUpdateInterval = null;
        }
    }

    updateTimeDisplay() {
        const currentTime = this.audioService.getCurrentTime();
        const duration = this.audioService.getDuration();
        
        // Update time display
        this.elements.timeDisplay.textContent = 
            `${this.formatTime(currentTime)} / ${this.formatTime(duration || 0)}`;
        
        // Update seek slider
        if (Number.isFinite(duration) && duration > 0 && !this.elements.seekSlider.dragging) {
            this.elements.seekSlider.value = (currentTime / duration) * 100;
        }
        
        // Update state
        this.playerState.setTime(currentTime, duration);
    }

    setupEventListeners() {
        // Play/Pause button
        this.elements.playPauseBtn.addEventListener('click', () => {
            if (this.audioService.isPlaying()) {
                this.audioService.pause();
            } else {
                this.audioService.play();
            }
        });

        // Seek slider
        this.elements.seekSlider.addEventListener('mousedown', () => {
            this.elements.seekSlider.dragging = true;
        });

        this.elements.seekSlider.addEventListener('mouseup', () => {
            this.elements.seekSlider.dragging = false;
        });

        this.elements.seekSlider.addEventListener('input', (e) => {
            const duration = this.audioService.getDuration();
            const seekTime = (duration * e.target.value) / 100;
            this.audioService.seek(seekTime);
            this.updateTimeDisplay();
        });

        // Volume slider
        this.elements.volumeSlider.addEventListener('input', (e) => {
            const volume = e.target.value / 100;
            this.audioService.setVolume(volume);
            this.playerState.setVolume(volume);
        });

        // Speed slider
        this.elements.speedSlider.addEventListener('input', (e) => {
            const speed = parseFloat(e.target.value);
            this.elements.speedValue.textContent = speed.toFixed(1);
            this.playerState.setSpeed(speed);
        });

        // Cancel button
        this.elements.cancelBtn.addEventListener('click', () => {
            this.audioService.cancel();
            this.playerState.reset();
            this.updateControls({ isGenerating: false });
            this.stopTimeUpdate();
        });
    }

    setupAudioEvents() {
        this.audioService.addEventListener('play', () => {
            this.elements.playPauseBtn.textContent = 'Pause';
            this.playerState.setPlaying(true);
            this.startTimeUpdate();
        });

        this.audioService.addEventListener('pause', () => {
            this.elements.playPauseBtn.textContent = 'Play';
            this.playerState.setPlaying(false);
            this.stopTimeUpdate();
        });

        this.audioService.addEventListener('ended', () => {
            this.elements.playPauseBtn.textContent = 'Play';
            this.playerState.setPlaying(false);
            this.stopTimeUpdate();
        });

        this.audioService.addEventListener('ready', () => {
            this.playerState.setReady(true);
            this.updateTimeDisplay();
        });

        // Initial time display
        this.updateTimeDisplay();
    }

    setupStateSubscription() {
        this.playerState.subscribe(state => this.updateControls(state));
    }

    updateControls(state) {
        // Update button states
        this.elements.playPauseBtn.disabled = !state.isReady && !state.isGenerating;
        this.elements.seekSlider.disabled = !state.isReady || !Number.isFinite(state.duration) || state.duration <= 0;
        this.elements.cancelBtn.style.display = state.isGenerating ? 'block' : 'none';
        
        // Update volume and speed if changed externally
        if (this.elements.volumeSlider.value !== state.volume * 100) {
            this.elements.volumeSlider.value = state.volume * 100;
        }
        
        if (this.elements.speedSlider.value !== state.speed.toString()) {
            this.elements.speedSlider.value = state.speed;
            this.elements.speedValue.textContent = state.speed.toFixed(1);
        }
    }

    cleanup() {
        this.stopTimeUpdate();
        if (this.audioService) {
            this.audioService.pause();
        }
        if (this.playerState) {
            this.playerState.reset();
        }
        // Reset UI elements
        this.elements.playPauseBtn.textContent = 'Play';
        this.elements.playPauseBtn.disabled = true;
        this.elements.seekSlider.value = 0;
        this.elements.seekSlider.disabled = true;
        this.elements.timeDisplay.textContent = '0:00 / 0:00';
    }
}

export default PlayerControls;
