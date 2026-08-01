export class WaveVisualizer {
    constructor(playerState) {
        this.playerState = playerState;
        this.wave = null;
        this.progressBar = null;
        this.onResize = null;
        this.resizeFrame = null;
        this.container = document.getElementById('wave-container');

        this.setupWave();
        this.setupProgressBar();
        this.setupStateSubscription();
    }

    setupWave() {
        this.wave = new SiriWave({
            container: this.container,
            style: 'ios9',
            width: this.container.clientWidth,
            height: 100,
            autostart: false,
            amplitude: 1,
            speed: 0.03
        });

        // setWidth keeps the canvas and the clear rect in sync, a stale width leaves the uncovered strip un-erased
        this.onResize = () => {
            if (this.resizeFrame !== null) return;
            this.resizeFrame = requestAnimationFrame(() => {
                this.resizeFrame = null;
                if (this.wave && typeof this.wave.setWidth === 'function') {
                    this.wave.setWidth(this.container.clientWidth);
                }
            });
        };
        window.addEventListener('resize', this.onResize);
    }

    setupProgressBar() {
        this.progressBar = document.createElement('progress');
        this.progressBar.max = 100;
        this.progressBar.value = 0;
        this.progressBar.className = 'generation-progress';
        this.container.appendChild(this.progressBar);
        this.progressBar.style.display = 'none';
    }

    setupStateSubscription() {
        this.wasPlaying = false;
        this.playerState.subscribe(state => {
            if (this.progressBar) {
                if (state.isGenerating) {
                    this.progressBar.style.display = 'block';
                    this.progressBar.value = state.progress;
                } else if (state.progress >= 100) {
                    setTimeout(() => {
                        if (!this.progressBar) return;
                        this.progressBar.style.display = 'none';
                        this.progressBar.value = 0;
                    }, 500);
                }
            }

            // start/stop only on transitions, a repeat start would reset the wave phase
            if (state.isPlaying && !this.wasPlaying) {
                this.wave?.start();
            } else if (!state.isPlaying && this.wasPlaying) {
                this.wave?.stop();
            }
            this.wasPlaying = state.isPlaying;
        });
    }

    updateProgress(receivedChunks, totalChunks) {
        if (!totalChunks || !this.progressBar) return;

        const progress = Math.min((receivedChunks / totalChunks) * 100, 99);

        if (receivedChunks === 0 || progress > this.progressBar.value) {
            this.progressBar.style.display = 'block';
            this.progressBar.value = progress;
            this.playerState.setProgress(receivedChunks, totalChunks);
        }
    }

    teardownWave() {
        if (this.onResize) {
            window.removeEventListener('resize', this.onResize);
            this.onResize = null;
        }

        if (this.resizeFrame !== null) {
            cancelAnimationFrame(this.resizeFrame);
            this.resizeFrame = null;
        }

        if (this.wave) {
            if (typeof this.wave.stop === 'function') this.wave.stop();
            // dispose detaches the canvas, without it every cleanup stacks another one
            if (typeof this.wave.dispose === 'function') this.wave.dispose();
            else if (this.wave.canvas && this.wave.canvas.parentNode) {
                this.wave.canvas.parentNode.removeChild(this.wave.canvas);
            }
            this.wave = null;
        }
    }

    teardownProgressBar() {
        if (!this.progressBar) return;
        this.progressBar.style.display = 'none';
        this.progressBar.value = 0;
        if (this.progressBar.parentNode) {
            this.progressBar.parentNode.removeChild(this.progressBar);
        }
        this.progressBar = null;
    }

    cleanup() {
        this.teardownWave();
        this.teardownProgressBar();

        this.setupWave();
        this.setupProgressBar();
        this.wasPlaying = false;

        if (this.playerState) {
            this.playerState.setProgress(0, 1);
        }
    }
}

export default WaveVisualizer;
