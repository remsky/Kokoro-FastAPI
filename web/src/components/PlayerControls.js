const SPEED_MIN = 0.1;
const SPEED_MAX = 4;

export class PlayerControls {
    constructor(audioService, playerState) {
        this.audioService = audioService;
        this.playerState = playerState;
        this.elements = {
            playPauseBtn: document.getElementById('play-pause-btn'),
            seekSlider: document.getElementById('seek-slider'),
            volumeSlider: document.getElementById('volume-slider'),
            speedInput: document.getElementById('speed-input'),
            timeDisplay: document.getElementById('time-display'),
            cancelBtn: document.getElementById('cancel-btn')
        };
        
        this.setupEventListeners();
        this.setupAudioEvents();
        this.setupStateSubscription();
        this.timeUpdateInterval = null;
        this.updateControls(this.playerState.getState());
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

        // Seek slider (pointer events cover mouse and touch drags)
        this.elements.seekSlider.addEventListener('pointerdown', () => {
            this.elements.seekSlider.dragging = true;
        });

        this.elements.seekSlider.addEventListener('pointerup', () => {
            this.elements.seekSlider.dragging = false;
        });

        this.elements.seekSlider.addEventListener('pointercancel', () => {
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

        this.elements.speedInput.addEventListener('input', (e) => {
            const speed = parseFloat(e.target.value);
            if (speed >= SPEED_MIN && speed <= SPEED_MAX) {
                this.playerState.setSpeed(speed);
            }
        });

        this.elements.speedInput.addEventListener('change', (e) => {
            const parsed = parseFloat(e.target.value);
            const speed = Number.isFinite(parsed)
                ? Math.min(SPEED_MAX, Math.max(SPEED_MIN, parsed))
                : this.playerState.getState().speed;
            this.playerState.setSpeed(speed);
            e.target.value = speed.toFixed(1);
        });

        // Cancel button
        this.elements.cancelBtn.addEventListener('click', () => {
            this.audioService.cancel();
            this.playerState.reset();
            this.updateControls({ isGenerating: false });
            this.stopTimeUpdate();
        });
    }

    // css picks the glyph off .playing, the label is all that changes here
    setPlayIcon(playing) {
        const btn = this.elements.playPauseBtn;
        btn.classList.toggle('playing', playing);
        btn.setAttribute('aria-label', playing ? 'Pause' : 'Play');
        btn.setAttribute('title', playing ? 'Pause' : 'Play');
    }

    setupAudioEvents() {
        this.audioService.addEventListener('play', () => {
            this.setPlayIcon(true);
            this.playerState.setPlaying(true);
            this.startTimeUpdate();
        });

        this.audioService.addEventListener('pause', () => {
            this.setPlayIcon(false);
            this.playerState.setPlaying(false);
            this.stopTimeUpdate();
        });

        this.audioService.addEventListener('ended', () => {
            this.setPlayIcon(false);
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
        
        if (document.activeElement !== this.elements.speedInput
            && parseFloat(this.elements.speedInput.value) !== state.speed) {
            this.elements.speedInput.value = state.speed.toFixed(1);
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
        this.setPlayIcon(false);
        this.elements.playPauseBtn.disabled = true;
        this.elements.seekSlider.value = 0;
        this.elements.seekSlider.disabled = true;
        this.elements.timeDisplay.textContent = '0:00 / 0:00';
    }
}

export default PlayerControls;
