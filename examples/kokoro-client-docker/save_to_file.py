import sys
from pathlib import Path
import os
import argparse
import requests
from urllib.parse import urljoin
from openai import OpenAI


def _parse_arg():
    parser = argparse.ArgumentParser(
        description="Save generated audio to file",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--base_url",
        default=None,
        type=str,
        required=True,
        help="The base URL to TTS server, ex: http://host.docker.internal:8880/v1"
    )
    parser.add_argument(
        "--api_key",
        default="not-needed-for-local",
        type=str,
        help="The API Key for TTS server"
    )
    parser.add_argument(
        "--model",
        default="kokoro",
        type=str,
        help="The model to use, current support [kokoro, tts-1, tts-1-hd]"
    )
    parser.add_argument(
        "--voice",
        default="af_bella",
        type=str,
        help="The voice to use. Can use multiple voices, ex: af_alloy+zm_yunyang"
    )
    parser.add_argument(
        "--list_voices",
        action="store_true",
        help="List the voices."
    )
    parser.add_argument(
        "--text_input",
        default="Hi, I am a text-to-speech assistant.",
        type=str,
        help="The input text"
    )
    parser.add_argument(
        "--output",
        default=None,
        type=str,
        help="The output audio file"
    )
    args = parser.parse_args()    
    return args


def _list_available_voices(base_url: str):
    url=urljoin(base_url, '/v1/audio/voices')
    response = requests.get(url=url)
    return response.json()


def _check_params(
    base_url: str,
    model: str,
    voices: str,
    ):
    # Check server
    url=urljoin(base_url, '/health')
    response = requests.get(url=url)
    if not response.json()['status'] == 'healthy':
        print(f"Server unhealthy: {base_url}")
        sys.exit(1)

    # Check model
    url=urljoin(base_url, '/v1/models')
    response = requests.get(url=url)
    model_names = [d['id'] for d in response.json()['data']]
    if not model in model_names:
        print(f"Incorrect model name: {model}")
        sys.exit(1)

    # Check voices
    available_voices = _list_available_voices(base_url=base_url)['voices']
    vs = voices.split('+')
    for v in vs:
        if not v in available_voices:
            print(f"Incorrect model name: {v}")


def main(
    base_url: str,
    api_key: str,
    model: str,
    voice: str,
    text_input: str,
    output: str
    ) -> None:

    client = OpenAI(base_url=base_url, api_key=api_key)

    # Create text-to-speech audio file
    with client.audio.speech.with_streaming_response.create(
        model=model,
        voice=voice,
        input=text_input,
    ) as response:
        response.stream_to_file(output)


if __name__ == "__main__":
    args = _parse_arg()
    
    if args.list_voices:
        print(_list_available_voices())
        sys.exit(0)

    _check_params(
        base_url=args.base_url,
        model=args.model,
        voices=args.voices
    )

    base_url = args.base_url
    api_key=args.api_key
    model=args.model
    voice=args.voice
    text_input=args.text_input
    if not args.output:
        output = Path(__file__).parent / "speech.mp3"
    elif os.path.isdir(args.output):
        output = os.path.join(args.output, 'speech.mp3')
    else:
        output = args.output

    main(
        base_url=base_url,
        api_key=api_key,
        model=model,
        voices=voice,
        text_input=text_input,
        output=output
    )