#!/usr/bin/env python3
# Compatible with both Windows and Linux
"""
Kokoro TTS Race Condition Test

This script creates multiple concurrent requests to a Kokoro TTS service
to reproduce a race condition where audio outputs don't match the requested text.
Each thread generates a simple numbered sentence, which should make mismatches
easy to identify through listening.

To run:
python kokoro_race_condition_test.py --threads 8 --iterations 5 --url http://localhost:8880
"""

import argparse
import base64
import concurrent.futures
import json
import os
import sys
import time
import wave
from pathlib import Path

import requests


def setup_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Test Kokoro TTS for race conditions")
    parser.add_argument(
        "--url",
        default="http://localhost:8880",
        help="Base URL of the Kokoro TTS service",
    )
    parser.add_argument(
        "--threads", type=int, default=8, help="Number of concurrent threads to use"
    )
    parser.add_argument(
        "--iterations", type=int, default=5, help="Number of iterations per thread"
    )
    parser.add_argument("--voice", default="af_heart", help="Voice to use for TTS")
    parser.add_argument(
        "--output-dir",
        default="./tts_test_output",
        help="Directory to save output files",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def generate_test_sentence(thread_id, iteration):
    """Generate a simple test sentence with numbers to make mismatches easily identifiable"""
    return (
        f"This is test sentence number {thread_id}-{iteration}. "
        f"If you hear this sentence, you should hear the numbers {thread_id}-{iteration}."
    )


def log_message(message, debug=False, is_error=False):
    """Log messages with timestamps"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    prefix = "[ERROR]" if is_error else "[INFO]"
    if is_error or debug:
        print(f"{prefix} {timestamp} - {message}")
    sys.stdout.flush()  # Ensure logs are visible in Docker output


def request_tts(url, test_id, text, voice, output_dir, debug=False):
    """Request TTS from the Kokoro API and save the WAV output"""
    start_time = time.time()
    output_file = os.path.join(output_dir, f"test_{test_id}.wav")
    text_file = os.path.join(output_dir, f"test_{test_id}.txt")

    # Log output paths for debugging
    log_message(f"Thread {test_id}: Text will be saved to: {text_file}", debug)
    log_message(f"Thread {test_id}: Audio will be saved to: {output_file}", debug)

    # Save the text for later comparison
    try:
        with open(text_file, "w") as f:
            f.write(text)
        log_message(f"Thread {test_id}: Successfully saved text file", debug)
    except Exception as e:
        log_message(
            f"Thread {test_id}: Error saving text file: {str(e)}", debug, is_error=True
        )

    # Make the TTS request
    try:
        log_message(f"Thread {test_id}: Requesting TTS for: '{text}'", debug)

        response = requests.post(
            f"{url}/v1/audio/speech",
            json={
                "model": "kokoro",
                "input": text,
                "voice": voice,
                "response_format": "wav",
            },
            headers={"Accept": "audio/wav"},
            timeout=60,  # Increase timeout to 60 seconds
        )

        log_message(
            f"Thread {test_id}: Response status code: {response.status_code}", debug
        )
        log_message(
            f"Thread {test_id}: Response content type: {response.headers.get('Content-Type', 'None')}",
            debug,
        )
        log_message(
            f"Thread {test_id}: Response content length: {len(response.content)} bytes",
            debug,
        )

        if response.status_code != 200:
            log_message(
                f"Thread {test_id}: API error: {response.status_code} - {response.text}",
                debug,
                is_error=True,
            )
            return False

        # Check if we got valid audio data
        if (
            len(response.content) < 100
        ):  # Sanity check - WAV files should be larger than this
            log_message(
                f"Thread {test_id}: Received suspiciously small audio data: {len(response.content)} bytes",
                debug,
                is_error=True,
            )
            log_message(
                f"Thread {test_id}: Content (base64): {base64.b64encode(response.content).decode('utf-8')}",
                debug,
                is_error=True,
            )
            return False

        # Save the audio output with explicit error handling
        try:
            with open(output_file, "wb") as f:
                bytes_written = f.write(response.content)
                log_message(
                    f"Thread {test_id}: Wrote {bytes_written} bytes to {output_file}",
                    debug,
                )

            # Verify the WAV file exists and has content
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                log_message(
                    f"Thread {test_id}: Verified file exists with size: {file_size} bytes",
                    debug,
                )

                # Validate WAV file by reading its headers
                try:
                    with wave.open(output_file, "rb") as wav_file:
                        channels = wav_file.getnchannels()
                        sample_width = wav_file.getsampwidth()
                        framerate = wav_file.getframerate()
                        frames = wav_file.getnframes()
                        log_message(
                            f"Thread {test_id}: Valid WAV file - channels: {channels}, "
                            f"sample width: {sample_width}, framerate: {framerate}, frames: {frames}",
                            debug,
                        )
                except Exception as wav_error:
                    log_message(
                        f"Thread {test_id}: Invalid WAV file: {str(wav_error)}",
                        debug,
                        is_error=True,
                    )
            else:
                log_message(
                    f"Thread {test_id}: File was not created: {output_file}",
                    debug,
                    is_error=True,
                )
        except Exception as save_error:
            log_message(
                f"Thread {test_id}: Error saving audio file: {str(save_error)}",
                debug,
                is_error=True,
            )
            return False

        end_time = time.time()
        log_message(
            f"Thread {test_id}: Saved output to {output_file} (time: {end_time - start_time:.2f}s)",
            debug,
        )
        return True

    except requests.exceptions.Timeout:
        log_message(f"Thread {test_id}: Request timed out", debug, is_error=True)
        return False
    except Exception as e:
        log_message(f"Thread {test_id}: Exception: {str(e)}", debug, is_error=True)
        return False


def worker_task(thread_id, args):
    """Worker task for each thread"""
    for i in range(args.iterations):
        iteration = i + 1
        test_id = f"{thread_id:02d}_{iteration:02d}"
        text = generate_test_sentence(thread_id, iteration)
        success = request_tts(
            args.url, test_id, text, args.voice, args.output_dir, args.debug
        )

        if not success:
            log_message(
                f"Thread {thread_id}: Iteration {iteration} failed",
                args.debug,
                is_error=True,
            )

        # Small delay between iterations to avoid overwhelming the API
        time.sleep(0.1)


def run_test(args):
    """Run the test with the specified parameters"""
    # Ensure output directory exists and check permissions
    os.makedirs(args.output_dir, exist_ok=True)

    # Test write access to the output directory
    test_file = os.path.join(args.output_dir, "write_test.txt")
    try:
        with open(test_file, "w") as f:
            f.write("Testing write access\n")
        os.remove(test_file)
        log_message(
            f"Successfully verified write access to output directory: {args.output_dir}"
        )
    except Exception as e:
        log_message(
            f"Warning: Cannot write to output directory {args.output_dir}: {str(e)}",
            is_error=True,
        )
        log_message(f"Current directory: {os.getcwd()}", is_error=True)
        log_message(f"Directory contents: {os.listdir('.')}", is_error=True)

    # Test connection to Kokoro TTS service
    try:
        response = requests.get(f"{args.url}/health", timeout=5)
        if response.status_code == 200:
            log_message(f"Successfully connected to Kokoro TTS service at {args.url}")
        else:
            log_message(
                f"Warning: Kokoro TTS service health check returned status {response.status_code}",
                is_error=True,
            )
    except Exception as e:
        log_message(
            f"Warning: Cannot connect to Kokoro TTS service at {args.url}: {str(e)}",
            is_error=True,
        )

    # Record start time
    start_time = time.time()
    log_message(
        f"Starting test with {args.threads} threads, {args.iterations} iterations per thread"
    )

    # Create and start worker threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = []
        for thread_id in range(1, args.threads + 1):
            futures.append(executor.submit(worker_task, thread_id, args))

        # Wait for all tasks to complete
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                log_message(
                    f"Thread execution failed: {str(e)}", args.debug, is_error=True
                )

    # Record end time and print summary
    end_time = time.time()
    total_time = end_time - start_time
    total_requests = args.threads * args.iterations
    log_message(f"Test completed in {total_time:.2f} seconds")
    log_message(f"Total requests: {total_requests}")
    log_message(f"Average time per request: {total_time / total_requests:.2f} seconds")
    log_message(f"Requests per second: {total_requests / total_time:.2f}")
    log_message(f"Output files saved to: {os.path.abspath(args.output_dir)}")
    log_message(
        "To verify, listen to the audio files and check if they match the text files"
    )
    log_message(
        "If you hear audio describing a different test number than the filename, you've found a race condition"
    )


def analyze_audio_files(output_dir):
    """Provide summary of the generated audio files"""
    # Look for both WAV and TXT files
    wav_files = list(Path(output_dir).glob("*.wav"))
    txt_files = list(Path(output_dir).glob("*.txt"))

    log_message(f"Found {len(wav_files)} WAV files and {len(txt_files)} TXT files")

    if len(wav_files) == 0:
        log_message(
            "No WAV files found! This indicates the TTS service requests may be failing.",
            is_error=True,
        )
        log_message(
            "Check the connection to the TTS service and the response status codes above.",
            is_error=True,
        )

    file_stats = []
    for wav_path in wav_files:
        try:
            with wave.open(str(wav_path), "rb") as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                duration = frames / rate

                # Get corresponding text
                text_path = wav_path.with_suffix(".txt")
                if text_path.exists():
                    with open(text_path, "r") as text_file:
                        text = text_file.read().strip()
                else:
                    text = "N/A"

                file_stats.append(
                    {"filename": wav_path.name, "duration": duration, "text": text}
                )
        except Exception as e:
            log_message(f"Error analyzing {wav_path}: {str(e)}", False, is_error=True)

    # Print summary table
    if file_stats:
        log_message("\nAudio File Summary:")
        log_message(f"{'Filename':<20}{'Duration':<12}{'Text':<60}")
        log_message("-" * 92)
        for stat in file_stats:
            log_message(
                f"{stat['filename']:<20}{stat['duration']:<12.2f}{stat['text'][:57] + '...' if len(stat['text']) > 60 else stat['text']:<60}"
            )

    # List missing WAV files where text files exist
    missing_wavs = set(p.stem for p in txt_files) - set(p.stem for p in wav_files)
    if missing_wavs:
        log_message(
            f"\nFound {len(missing_wavs)} text files without corresponding WAV files:",
            is_error=True,
        )
        for stem in sorted(list(missing_wavs))[:10]:  # Limit to 10 for readability
            log_message(f"  - {stem}.txt (no WAV file)", is_error=True)
        if len(missing_wavs) > 10:
            log_message(f"  ... and {len(missing_wavs) - 10} more", is_error=True)


if __name__ == "__main__":
    args = setup_args()
    run_test(args)
    analyze_audio_files(args.output_dir)

    log_message("\nNext Steps:")
    log_message("1. Listen to the generated audio files")
    log_message("2. Verify if each audio correctly says its ID number")
    log_message(
        "3. Check for any mismatches between the audio content and the text files"
    )
    log_message(
        "4. If mismatches are found, you've successfully reproduced the race condition"
    )
