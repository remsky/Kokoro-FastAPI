"""Build speaker-labelled subtitles from a multi-speaker render.

Sends inline [voice:...] tags to /dev/captioned_speech with allow_voice_tags on,
so every word timestamp comes back carrying the voice that spoke it. Groups the
words into cues on speaker changes and writes an SRT next to the audio.

Also renders the same text with the opt in off to show the difference: the
timestamps are identical, the voice field is null, and the tags get spoken.

Run from the repo root:
    uv run --project examples python examples/assorted_checks/test_dialogue/caption_dialogue.py
"""

import base64
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import requests

BASE_URL = os.getenv("KOKORO_BASE_URL", "http://localhost:8880").rstrip("/")
TIMEOUT = float(os.getenv("KOKORO_TIMEOUT", "600"))
MAX_CUE_SECONDS = 5.0
MAX_CUE_CHARS = 80
PUNCTUATION = set(".,!?;:")

OUTPUT_DIR = Path(__file__).parent / "output"

SCRIPT = (
    "[voice:af_bella] So the reconciliation job caught it before anyone noticed? "
    "[voice:am_michael] It did. Fifteen minutes after the drift started, right on schedule. "
    "[voice:af_bella] And nobody had to be paged for it. "
    "[voice:bm_george] That is the whole point of the thing, really."
)


def render(text: str, allow_voice_tags: bool) -> dict:
    response = requests.post(
        f"{BASE_URL}/dev/captioned_speech",
        json={
            "model": "kokoro",
            "input": text,
            "voice": "af_heart",
            "response_format": "mp3",
            "stream": False,
            "return_timestamps": True,
            "allow_voice_tags": allow_voice_tags,
        },
        timeout=TIMEOUT,
    )
    response.raise_for_status()
    return json.loads(response.content)


def join_word(text: str, word: str) -> str:
    """Words come back tokenized, so punctuation lands as its own timestamp."""
    separator = "" if word in PUNCTUATION else " "
    return f"{text}{separator}{word}"


def group_into_cues(timestamps: List[dict]) -> List[dict]:
    """Merge consecutive words into cues, breaking on speaker change, length or duration."""
    cues: List[dict] = []

    for stamp in timestamps:
        current = cues[-1] if cues else None
        too_long = current and (
            stamp["end_time"] - current["start_time"] > MAX_CUE_SECONDS
            or len(current["text"]) + len(stamp["word"]) + 1 > MAX_CUE_CHARS
        )
        if current and current["voice"] == stamp["voice"] and not too_long:
            current["text"] = join_word(current["text"], stamp["word"])
            current["end_time"] = stamp["end_time"]
            continue
        cues.append(
            {
                "voice": stamp["voice"],
                "text": stamp["word"],
                "start_time": stamp["start_time"],
                "end_time": stamp["end_time"],
            }
        )

    # a chunk's last word can run past the next chunk's first, so cues would overlap
    for cue, following in zip(cues, cues[1:]):
        cue["end_time"] = min(cue["end_time"], following["start_time"])

    return cues


def srt_time(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def to_srt(cues: List[dict]) -> str:
    blocks = []
    for index, cue in enumerate(cues, start=1):
        label = f"{cue['voice']}: " if cue["voice"] else ""
        blocks.append(
            f"{index}\n"
            f"{srt_time(cue['start_time'])} --> {srt_time(cue['end_time'])}\n"
            f"{label}{cue['text']}\n"
        )
    return "\n".join(blocks)


def speakers(timestamps: List[dict]) -> List[Optional[str]]:
    """Distinct voices in order of first appearance, preserving None."""
    return list(dict.fromkeys(stamp["voice"] for stamp in timestamps))


def main() -> int:
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"server: {BASE_URL}")
    print(f"output: {OUTPUT_DIR}\n")

    labelled = render(SCRIPT, allow_voice_tags=True)
    timestamps = labelled["timestamps"] or []
    if not timestamps:
        print("no timestamps returned, nothing to caption")
        return 1

    (OUTPUT_DIR / "dialogue_captions.mp3").write_bytes(
        base64.b64decode(labelled["audio"].encode("utf-8"))
    )
    cues = group_into_cues(timestamps)
    (OUTPUT_DIR / "dialogue_captions.srt").write_text(to_srt(cues), encoding="utf-8")

    print(f"{len(timestamps)} words, {len(cues)} cues, speakers {speakers(timestamps)}")
    for cue in cues[:6]:
        print(
            f"  {srt_time(cue['start_time'])} {cue['voice'] or '(unlabelled)':<12} "
            f"{cue['text'][:50]}"
        )
    if len(cues) > 6:
        print(f"  ... {len(cues) - 6} more")

    plain = render(SCRIPT, allow_voice_tags=False)
    plain_timestamps = plain["timestamps"] or []
    print(
        f"\nwithout the opt in: {len(plain_timestamps)} words, "
        f"speakers {speakers(plain_timestamps)}, tags spoken as written"
    )

    print(f"\nwrote {OUTPUT_DIR / 'dialogue_captions.srt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
