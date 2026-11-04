"""Streaming DSP contracts, independent of model downloads."""

import numpy as np
import pytest

from api.src.services import audio as audio_service


def tone(samples, rate=24000):
    return (0.2 * np.sin(2 * np.pi * 220 * np.arange(samples) / rate)).astype(
        np.float32
    )


def filtered(audio, speed, splits, rate=24000):
    processor = audio_service.AudioFilterStream(speed, sample_rate=rate)
    output = []
    for chunk in np.array_split(audio, splits):
        output.append(processor.push(chunk))
    output.append(processor.flush())
    processor.close()
    return np.concatenate(output)


@pytest.mark.parametrize("speed", [0.25, 0.7, 1.0, 1.5, 2.0, 4.0])
def test_stream_is_independent_of_input_chunk_boundaries(speed):
    assert hasattr(audio_service, "AudioFilterStream"), (
        "A request-scoped filter is required"
    )
    audio = tone(48000)
    whole = filtered(audio, speed, 1)
    split = filtered(audio, speed, 73)
    assert len(split) == round(len(audio) / speed)
    np.testing.assert_allclose(split, whole, atol=1e-6)
    freq = np.fft.rfftfreq(len(split), 1 / 24000)[np.argmax(abs(np.fft.rfft(split)))]
    assert abs(freq - 220) < 5


def test_stream_emits_before_flush_and_reuses_graph():
    assert hasattr(audio_service, "AudioFilterStream"), (
        "A request-scoped filter is required"
    )
    processor = audio_service.AudioFilterStream(1.5)
    first = processor.push(tone(12000))
    graph = processor.graph
    second = processor.push(tone(12000))
    assert processor.graph is graph
    assert len(first) > 0 and len(second) > 0
    tail = processor.flush()
    assert len(first) + len(second) + len(tail) == 16000
    assert len(processor.flush()) == 0
    processor.close()
    assert processor.graph is None


@pytest.mark.parametrize("samples", [0, 1, 200])
def test_stream_short_and_empty_tail(samples):
    assert hasattr(audio_service, "AudioFilterStream"), (
        "A request-scoped filter is required"
    )
    audio = tone(samples)
    output = filtered(audio, 0.7, 1)
    assert len(output) == round(samples / 0.7)
    if samples == 200:
        assert np.max(abs(output)) > 0.05


@pytest.mark.parametrize("rate", [8000, 24000, 48000])
@pytest.mark.parametrize("normalize", [False, True])
def test_stream_preserves_sample_rate_and_stereo_layout(rate, normalize):
    assert hasattr(audio_service, "AudioFilterStream"), (
        "A request-scoped filter is required"
    )
    audio = np.column_stack([tone(rate, rate), tone(rate, rate) * 0.5])
    processor = audio_service.AudioFilterStream(
        1.5, sample_rate=rate, channels=2, normalize=normalize
    )
    result = np.concatenate([processor.push(audio), processor.flush()])
    assert result.shape == (round(rate / 1.5), 2)
    np.testing.assert_allclose(result[:, 1], result[:, 0] * 0.5, atol=1e-6)


@pytest.mark.parametrize("speed", [0, -1, float("nan"), float("inf"), 5])
def test_invalid_speed_rejected(speed):
    with pytest.raises(ValueError):
        audio_service.AudioFilterStream(speed)


def test_concurrent_streams_do_not_share_filter_history():
    from concurrent.futures import ThreadPoolExecutor

    quiet = tone(48000) * 0.01
    loud = tone(48000)
    with ThreadPoolExecutor(2) as pool:
        a = pool.submit(filtered, quiet, 0.7, 19)
        b = pool.submit(filtered, loud, 2.0, 23)
    np.testing.assert_array_equal(a.result(), filtered(quiet, 0.7, 1))
    np.testing.assert_array_equal(b.result(), filtered(loud, 2.0, 1))


def test_closed_stream_rejects_more_samples():
    processor = audio_service.AudioFilterStream(1.5)
    processor.push(tone(200))
    processor.close()
    with pytest.raises(RuntimeError):
        processor.push(tone(200))


def normalized(audio, speed=1.0, splits=20, gain=1.0):
    processor = audio_service.AudioFilterStream(speed, normalize=True, gain=gain)
    try:
        return np.concatenate(
            [
                *[processor.push(x) for x in np.array_split(audio, splits)],
                processor.flush(),
            ]
        )
    finally:
        processor.close()


@pytest.mark.parametrize("speed", [0.7, 1.0, 1.5, 2.0])
def test_normalization_reduces_level_difference_without_clipping(speed):
    quiet = normalized(tone(96000) * 0.1, speed)
    loud = normalized(tone(96000) * 4.5, speed)
    rms = lambda x: np.sqrt(np.mean(x**2))
    assert rms(quiet) > 0.04
    assert rms(loud) / rms(quiet) < 3
    assert max(abs(quiet).max(), abs(loud).max()) <= 0.951
    assert len(quiet) == round(96000 / speed)


def test_normalization_ignores_chunk_boundaries_and_preserves_silence():
    audio = tone(96000) * 0.1
    np.testing.assert_allclose(
        normalized(audio, splits=1), normalized(audio, splits=127), atol=1e-6
    )
    np.testing.assert_array_equal(
        normalized(np.zeros(24000, np.float32)), np.zeros(24000)
    )
    noise = normalized(tone(96000) * 0.00005)
    assert np.sqrt(np.mean(noise**2)) < 0.00001


@pytest.mark.parametrize("samples", [0, 1, 200, 2400])
def test_normalization_short_utterance_flush(samples):
    output = normalized(tone(samples) * 0.1, splits=1)
    assert len(output) == samples
    assert np.all(np.isfinite(output))
    if samples >= 200:
        assert abs(output).max() >= 0.019


def test_normalization_streams_and_respects_explicit_gain():
    processor = audio_service.AudioFilterStream(normalize=True)
    first = processor.push(tone(12000) * 0.1)
    assert len(first) > 0
    processor.close()
    audio = tone(24000) * 0.1
    np.testing.assert_allclose(
        normalized(audio, gain=0.5), normalized(audio) * 0.5, atol=1e-6
    )
    np.testing.assert_array_equal(normalized(audio, gain=0), np.zeros(len(audio)))


@pytest.mark.parametrize("speed", [1.0, 1.5, 2.0])
def test_normalization_does_not_fade_end_of_clip(speed):
    audio = np.full(24000, 0.5, dtype=np.float32)
    output = normalized(audio, speed=speed, splits=20)
    window = round(2400 / speed)
    middle = np.mean(np.abs(output[len(output) // 2 : len(output) // 2 + window]))
    tail = np.mean(np.abs(output[-window:]))
    assert tail >= middle * 0.9
