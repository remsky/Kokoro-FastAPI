"""
ZipVoice Integration Examples

This script demonstrates various ways to use the ZipVoice integration
for zero-shot voice cloning.
"""

import asyncio
import base64
from pathlib import Path

import httpx


BASE_URL = "http://localhost:8880"


async def example_1_register_and_use_voice():
    """Example 1: Register a voice and use it for generation"""
    print("\n" + "="*60)
    print("Example 1: Register and Use Voice")
    print("="*60)

    async with httpx.AsyncClient() as client:
        # Step 1: Register a voice
        print("\n[1/2] Registering voice...")

        # For this example, create a dummy WAV file
        # In production, use a real voice sample
        voice_file_path = "my_voice_sample.wav"

        files = {'audio_file': open(voice_file_path, 'rb')}
        data = {
            'name': 'my_custom_voice',
            'transcription': 'This is a sample of my voice speaking clearly.'
        }

        response = await client.post(
            f'{BASE_URL}/v1/zipvoice/voices/register',
            files=files,
            data=data
        )

        if response.status_code == 200:
            print("✓ Voice registered successfully!")
            print(f"  Response: {response.json()}")
        else:
            print(f"✗ Failed: {response.text}")
            return

        # Step 2: Generate speech using the registered voice
        print("\n[2/2] Generating speech...")

        response = await client.post(
            f'{BASE_URL}/v1/zipvoice/audio/speech',
            json={
                'model': 'zipvoice',
                'input': 'Hello! This is a demonstration of zero-shot voice cloning using ZipVoice.',
                'voice': 'my_custom_voice',
                'prompt_text': 'This is a sample of my voice speaking clearly.',
                'response_format': 'mp3',
                'stream': False
            }
        )

        if response.status_code == 200:
            output_file = 'output_example1.mp3'
            with open(output_file, 'wb') as f:
                f.write(response.content)
            print(f"✓ Speech generated successfully!")
            print(f"  Saved to: {output_file}")
        else:
            print(f"✗ Failed: {response.text}")


async def example_2_url_based_voice():
    """Example 2: Use voice from URL"""
    print("\n" + "="*60)
    print("Example 2: URL-Based Voice Prompt")
    print("="*60)

    async with httpx.AsyncClient() as client:
        print("\nGenerating speech from URL-based voice prompt...")

        # Use a voice sample from a URL
        voice_url = "https://example.com/voice_samples/speaker1.wav"

        response = await client.post(
            f'{BASE_URL}/v1/zipvoice/audio/speech',
            json={
                'model': 'zipvoice',
                'input': 'This speech was generated using a voice sample downloaded from a URL.',
                'voice': f'url+{voice_url}',
                'prompt_text': 'The transcription of the voice sample.',
                'response_format': 'wav',
                'num_steps': 8
            }
        )

        if response.status_code == 200:
            output_file = 'output_example2.wav'
            with open(output_file, 'wb') as f:
                f.write(response.content)
            print(f"✓ Speech generated successfully!")
            print(f"  Saved to: {output_file}")
        else:
            print(f"✗ Failed: {response.text}")


async def example_3_base64_voice():
    """Example 3: Use base64 encoded voice"""
    print("\n" + "="*60)
    print("Example 3: Base64 Encoded Voice")
    print("="*60)

    # Read and encode audio file
    voice_file = "my_voice_sample.wav"
    with open(voice_file, 'rb') as f:
        audio_bytes = f.read()
        audio_b64 = base64.b64encode(audio_bytes).decode()

    async with httpx.AsyncClient() as client:
        print("\nGenerating speech from base64 encoded voice...")

        response = await client.post(
            f'{BASE_URL}/v1/zipvoice/audio/speech',
            json={
                'model': 'zipvoice',
                'input': 'This demonstrates using a base64 encoded voice prompt.',
                'voice': f'base64+{audio_b64}',
                'prompt_text': 'This is the transcription of my voice.',
                'response_format': 'mp3'
            }
        )

        if response.status_code == 200:
            output_file = 'output_example3.mp3'
            with open(output_file, 'wb') as f:
                f.write(response.content)
            print(f"✓ Speech generated successfully!")
            print(f"  Saved to: {output_file}")
        else:
            print(f"✗ Failed: {response.text}")


async def example_4_streaming():
    """Example 4: Streaming response"""
    print("\n" + "="*60)
    print("Example 4: Streaming Generation")
    print("="*60)

    async with httpx.AsyncClient() as client:
        print("\nStreaming speech generation...")

        # Long text for streaming
        long_text = """
        This is a longer piece of text that will be streamed back to the client.
        Streaming allows you to start playing audio before the entire generation
        is complete. This is particularly useful for real-time applications where
        latency matters. ZipVoice supports pseudo-streaming by generating chunks
        and yielding them progressively.
        """

        async with client.stream(
            'POST',
            f'{BASE_URL}/v1/zipvoice/audio/speech',
            json={
                'model': 'zipvoice',
                'input': long_text,
                'voice': 'my_custom_voice',
                'prompt_text': 'Sample transcription.',
                'response_format': 'mp3',
                'stream': True
            }
        ) as response:
            output_file = 'output_example4.mp3'
            bytes_received = 0

            with open(output_file, 'wb') as f:
                async for chunk in response.aiter_bytes():
                    f.write(chunk)
                    bytes_received += len(chunk)
                    print(f"  Received: {bytes_received} bytes", end='\r')

            print(f"\n✓ Streaming complete!")
            print(f"  Saved to: {output_file}")


async def example_5_voice_management():
    """Example 5: Voice management operations"""
    print("\n" + "="*60)
    print("Example 5: Voice Management")
    print("="*60)

    async with httpx.AsyncClient() as client:
        # List all voices
        print("\n[1/4] Listing registered voices...")
        response = await client.get(f'{BASE_URL}/v1/zipvoice/voices')
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Found {data['count']} registered voice(s):")
            for name in data['voices'].keys():
                print(f"  - {name}")
        else:
            print(f"✗ Failed: {response.text}")

        # Get info for specific voice
        print("\n[2/4] Getting voice info...")
        voice_name = "my_custom_voice"
        response = await client.get(f'{BASE_URL}/v1/zipvoice/voices/{voice_name}')
        if response.status_code == 200:
            info = response.json()
            print(f"✓ Voice info for '{voice_name}':")
            print(f"  Transcription: {info['transcription']}")
            print(f"  Duration: {info['audio_info']['duration']:.2f}s")
            print(f"  Sample rate: {info['audio_info']['samplerate']}Hz")
        else:
            print(f"✗ Failed: {response.text}")

        # Clear cache
        print("\n[3/4] Clearing URL cache...")
        response = await client.post(
            f'{BASE_URL}/v1/zipvoice/voices/cache/clear',
            params={'cache_type': 'url'}
        )
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Cleared {result['files_deleted']} cached file(s)")
        else:
            print(f"✗ Failed: {response.text}")

        # Delete voice (commented out to preserve the example voice)
        # print("\n[4/4] Deleting voice...")
        # response = await client.delete(f'{BASE_URL}/v1/zipvoice/voices/{voice_name}')
        # if response.status_code == 200:
        #     print(f"✓ Voice '{voice_name}' deleted")
        # else:
        #     print(f"✗ Failed: {response.text}")

        print("\n[4/4] Skipping deletion to preserve example voice")


async def example_6_fast_generation():
    """Example 6: Fast generation with optimizations"""
    print("\n" + "="*60)
    print("Example 6: Fast Generation (Optimized)")
    print("="*60)

    async with httpx.AsyncClient() as client:
        print("\nGenerating with speed optimizations...")

        import time
        start = time.time()

        response = await client.post(
            f'{BASE_URL}/v1/zipvoice/audio/speech',
            json={
                'model': 'zipvoice_distill',  # Use distilled model
                'input': 'This is a fast generation example.',
                'voice': 'my_custom_voice',
                'prompt_text': 'Sample.',
                'response_format': 'mp3',
                'num_steps': 4,  # Reduce inference steps
                'remove_long_silence': True,
                'stream': False
            }
        )

        elapsed = time.time() - start

        if response.status_code == 200:
            output_file = 'output_example6.mp3'
            with open(output_file, 'wb') as f:
                f.write(response.content)
            print(f"✓ Fast generation complete in {elapsed:.2f}s!")
            print(f"  Saved to: {output_file}")
            print("\nOptimizations used:")
            print("  - zipvoice_distill model")
            print("  - num_steps=4 (vs default 8)")
            print("  - remove_long_silence=True")
        else:
            print(f"✗ Failed: {response.text}")


async def example_7_parameter_comparison():
    """Example 7: Compare different parameters"""
    print("\n" + "="*60)
    print("Example 7: Parameter Comparison")
    print("="*60)

    configs = [
        {
            'name': 'High Quality',
            'model': 'zipvoice',
            'num_steps': 16,
            'speed': 1.0
        },
        {
            'name': 'Balanced',
            'model': 'zipvoice',
            'num_steps': 8,
            'speed': 1.0
        },
        {
            'name': 'Fast',
            'model': 'zipvoice_distill',
            'num_steps': 4,
            'speed': 1.0
        },
        {
            'name': 'Fast + Slow Speech',
            'model': 'zipvoice_distill',
            'num_steps': 4,
            'speed': 0.8
        }
    ]

    test_text = "This is a test of different generation parameters."

    async with httpx.AsyncClient() as client:
        for i, config in enumerate(configs, 1):
            print(f"\n[{i}/{len(configs)}] Testing: {config['name']}")

            import time
            start = time.time()

            response = await client.post(
                f'{BASE_URL}/v1/zipvoice/audio/speech',
                json={
                    'model': config['model'],
                    'input': test_text,
                    'voice': 'my_custom_voice',
                    'prompt_text': 'Sample.',
                    'num_steps': config['num_steps'],
                    'speed': config['speed'],
                    'response_format': 'mp3',
                    'stream': False
                }
            )

            elapsed = time.time() - start

            if response.status_code == 200:
                output_file = f'output_example7_{i}.mp3'
                with open(output_file, 'wb') as f:
                    f.write(response.content)
                print(f"  ✓ Generated in {elapsed:.2f}s ({len(response.content)} bytes)")
                print(f"    Saved to: {output_file}")
            else:
                print(f"  ✗ Failed: {response.text}")


async def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("ZipVoice Integration Examples")
    print("="*60)
    print("\nPrerequisites:")
    print("1. FastAPI server running on http://localhost:8880")
    print("2. ZipVoice backend enabled")
    print("3. Voice sample file 'my_voice_sample.wav' in current directory")

    input("\nPress Enter to continue...")

    # Run examples
    examples = [
        ("Register and Use Voice", example_1_register_and_use_voice),
        # ("URL-Based Voice", example_2_url_based_voice),  # Requires real URL
        # ("Base64 Voice", example_3_base64_voice),  # Requires voice file
        ("Streaming", example_4_streaming),
        ("Voice Management", example_5_voice_management),
        ("Fast Generation", example_6_fast_generation),
        ("Parameter Comparison", example_7_parameter_comparison),
    ]

    for name, func in examples:
        try:
            await func()
        except Exception as e:
            print(f"\n✗ Example '{name}' failed: {e}")

    print("\n" + "="*60)
    print("Examples Complete!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
