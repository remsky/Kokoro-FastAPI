export function samplePeaks(samples, columns) {
    const per = Math.max(1, Math.ceil(samples.length / columns));
    const peaks = [];
    for (let start = 0; start < samples.length; start += per) {
        let peak = 0;
        for (let i = start; i < Math.min(samples.length, start + per); i++) {
            peak = Math.max(peak, Math.abs(samples[i]));
        }
        peaks.push(peak);
    }
    return peaks;
}

function drawPeaks(canvas, peaks) {
    const context = canvas.getContext('2d');
    const { width, height } = canvas;
    context.clearRect(0, 0, width, height);
    context.fillStyle = getComputedStyle(canvas).color;
    for (let x = 0; x < peaks.length; x++) {
        const bar = Math.max(1, peaks[x] * height);
        context.fillRect(x, (height - bar) / 2, 1, bar);
    }
}

export function drawSamples(canvas, samples) {
    canvas.width = canvas.clientWidth;
    drawPeaks(canvas, samples ? samplePeaks(samples, canvas.width) : []);
}

export function liveLevels(canvas, stream) {
    const context = new AudioContext();
    const analyser = context.createAnalyser();
    analyser.fftSize = 2048;
    context.createMediaStreamSource(stream).connect(analyser);
    const buffer = new Float32Array(analyser.fftSize);
    const peaks = [];
    canvas.width = canvas.clientWidth;
    let frame;
    const tick = () => {
        analyser.getFloatTimeDomainData(buffer);
        peaks.push(samplePeaks(buffer, 1)[0]);
        if (peaks.length > canvas.width) {
            peaks.shift();
        }
        drawPeaks(canvas, peaks);
        frame = requestAnimationFrame(tick);
    };
    tick();
    return () => {
        cancelAnimationFrame(frame);
        context.close();
    };
}
