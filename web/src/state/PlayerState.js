export class PlayerState {
    constructor() {
        this.state = {
            isPlaying: false,
            isGenerating: false,
            currentTime: 0,
            duration: 0,
            volume: 1,
            speed: 1,
            progress: 0,
            isReady: false,
            error: null
        };
        this.listeners = new Set();
    }

    subscribe(listener) {
        this.listeners.add(listener);
        return () => this.listeners.delete(listener);
    }

    notify() {
        this.listeners.forEach(listener => listener(this.state));
    }

    setState(updates) {
        let changed = false;
        for (const key in updates) {
            if (updates[key] !== this.state[key]) {
                changed = true;
                break;
            }
        }
        if (!changed) return;
        this.state = {
            ...this.state,
            ...updates
        };
        this.notify();
    }

    setPlaying(isPlaying) {
        this.setState({ isPlaying });
    }

    setGenerating(isGenerating) {
        this.setState({ isGenerating });
    }

    setProgress(loaded, total) {
        const progress = total > 0 ? (loaded / total) * 100 : 0;
        this.setState({ progress });
    }

    setReady(isReady) {
        this.setState({ isReady });
    }

    setTime(currentTime, duration) {
        this.setState({ currentTime, duration });
    }

    setVolume(volume) {
        this.setState({ volume });
    }

    setSpeed(speed) {
        this.setState({ speed });
    }

    setError(error) {
        this.setState({ error });
    }

    clearError() {
        this.setState({ error: null });
    }

    reset() {
        // Keep current speed setting but reset everything else
        const currentSpeed = this.state.speed;
        const currentVolume = this.state.volume;
        
        this.setState({
            isPlaying: false,
            isGenerating: false,
            currentTime: 0,
            duration: 0,
            progress: 0,
            isReady: false,
            error: null,
            speed: currentSpeed,
            volume: currentVolume
        });
    }

    getState() {
        return { ...this.state };
    }
}

export default PlayerState;
