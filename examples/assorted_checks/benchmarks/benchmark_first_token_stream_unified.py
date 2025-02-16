#!/usr/bin/env python3
import os
import time

import requests
from openai import OpenAI
from lib.stream_utils import run_benchmark

OPENAI_CLIENT = OpenAI(
    base_url="http://localhost:8880/v1", api_key="not-needed-for-local"
)


def measure_first_token_requests(
    text: str, output_dir: str, tokens: int, run_number: int
) -> dict:
    """Measure time to audio via direct API calls and save the audio output"""
    results = {
        "text_length": len(text),
        "token_count": None,  # Will be set by run_benchmark
        "total_time": None,
        "time_to_first_chunk": None,
        "error": None,
        "audio_path": None,
        "audio_length": None,
    }

    try:
        start_time = time.time()

        # Make request with streaming enabled
        response = requests.post(
            "http://localhost:8880/v1/audio/speech",
            json={
                "model": "kokoro",
                "input": text,
                "voice": "af_heart",
                "response_format": "pcm",
                "stream": True,
            },
            stream=True,
            timeout=1800,
        )
        response.raise_for_status()

        # Save complete audio
        audio_filename = f"benchmark_tokens{tokens}_run{run_number}_stream.wav"
        audio_path = os.path.join(output_dir, audio_filename)
        results["audio_path"] = audio_path

        first_chunk_time = None
        chunks = []
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                    results["time_to_first_chunk"] = first_chunk_time - start_time
                chunks.append(chunk)

        # Concatenate all PCM chunks
        if not chunks:
            raise ValueError("No audio chunks received")

        all_audio_data = b"".join(chunks)

        # Write as WAV file
        import wave

        with wave.open(audio_path, "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
            wav_file.setframerate(24000)  # Known sample rate for Kokoro
            wav_file.writeframes(all_audio_data)

        # Calculate audio length using scipy
        import scipy.io.wavfile as wavfile

        sample_rate, audio_data = wavfile.read(audio_path)
        results["audio_length"] = len(audio_data) / sample_rate  # Length in seconds

        results["total_time"] = time.time() - start_time

        # Print debug info
        print(f"Complete audio size: {len(all_audio_data)} bytes")
        print(f"Number of chunks received: {len(chunks)}")
        print(f"Audio length: {results['audio_length']:.3f}s")

        return results

    except Exception as e:
        results["error"] = str(e)
        return results


def measure_first_token_openai(
    text: str, output_dir: str, tokens: int, run_number: int
) -> dict:
    """Measure time to audio via OpenAI API calls and save the audio output"""
    results = {
        "text_length": len(text),
        "token_count": None,  # Will be set by run_benchmark
        "total_time": None,
        "time_to_first_chunk": None,
        "error": None,
        "audio_path": None,
        "audio_length": None,
    }

    try:
        start_time = time.time()

        # Initialize OpenAI client

        # Save complete audio
        audio_filename = f"benchmark_tokens{tokens}_run{run_number}_stream_openai.wav"
        audio_path = os.path.join(output_dir, audio_filename)
        results["audio_path"] = audio_path

        first_chunk_time = None
        all_audio_data = bytearray()
        chunk_count = 0

        # Make streaming request using OpenAI client
        with OPENAI_CLIENT.audio.speech.with_streaming_response.create(
            model="kokoro",
            voice="af_heart",
            response_format="pcm",
            input=text,
        ) as response:
            for chunk in response.iter_bytes(chunk_size=1024):
                if chunk:
                    chunk_count += 1
                    if first_chunk_time is None:
                        first_chunk_time = time.time()
                        results["time_to_first_chunk"] = first_chunk_time - start_time
                    all_audio_data.extend(chunk)

        # Write as WAV file
        import wave

        with wave.open(audio_path, "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
            wav_file.setframerate(24000)  # Known sample rate for Kokoro
            wav_file.writeframes(all_audio_data)

        # Calculate audio length using scipy
        import scipy.io.wavfile as wavfile

        sample_rate, audio_data = wavfile.read(audio_path)
        results["audio_length"] = len(audio_data) / sample_rate  # Length in seconds

        results["total_time"] = time.time() - start_time

        # Print debug info
        print(f"Complete audio size: {len(all_audio_data)} bytes")
        print(f"Number of chunks received: {chunk_count}")
        print(f"Audio length: {results['audio_length']:.3f}s")

        return results

    except Exception as e:
        results["error"] = str(e)
        return results


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prefix = "cpu"
    # Run requests benchmark
    print("\n=== Running Direct Requests Benchmark ===")
    run_benchmark(
        measure_first_token_requests,
        output_dir=os.path.join(script_dir, "output_audio_stream"),
        output_data_dir=os.path.join(script_dir, "output_data"),
        output_plots_dir=os.path.join(script_dir, "output_plots"),
        suffix="_stream",
        plot_title_suffix="(Streaming)",
        prefix=prefix,
    )
    # Run OpenAI benchmark
    print("\n=== Running OpenAI Library Benchmark ===")
    run_benchmark(
        measure_first_token_openai,
        output_dir=os.path.join(script_dir, "output_audio_stream_openai"),
        output_data_dir=os.path.join(script_dir, "output_data"),
        output_plots_dir=os.path.join(script_dir, "output_plots"),
        suffix="_stream_openai",
        plot_title_suffix="(OpenAI Streaming)",
        prefix=prefix,
    )


if __name__ == "__main__":
    main()
