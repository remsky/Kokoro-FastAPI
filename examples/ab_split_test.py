"""A/B listen test for the sentence splitter change (PR #415 adaptation).

Run once against the old image with --label old, rebuild, run again with
--label new. Files land in examples/ab_split/<label>/ with matching names,
so any player sorted by name gives direct A/B pairs.

    uv run ab_split_test.py --label old
"""

import argparse
from pathlib import Path

import requests

URL = "http://localhost:8880/v1/audio/speech"

CASES = [
    (
        "01_abbrev_mid",
        "af_heart",
        "This, that, the other thing, etc. was on the list. We need bread, milk, eggs, etc. before the store closes.",
    ),
    (
        "02_abbrev_end",
        "af_heart",
        "Bring pencils, paper, erasers, etc. Then we can start the exam.",
    ),
    (
        "03_ie_eg",
        "af_heart",
        "Use the primary color, i.e. red, for warnings. Some fruits, e.g. apples and pears, keep well.",
    ),
    (
        "04_decimals",
        "af_heart",
        "You have 4.2 messages on average. The rate rose 3.5 percent, reaching 12.8 million this year.",
    ),
    (
        "05_semicolons",
        "af_heart",
        "The plan was simple; the execution was not. We packed three things: rope, water, and a map.",
    ),
    (
        "06_titles",
        "af_heart",
        "Dr. Smith met Mr. Jones at 5 p.m. on Main St. to discuss the results.",
    ),
    (
        "07_dash_continuation",
        "af_heart",
        "The Marbles. --- In the heart of a vibrant city in Spain stood an ancient temple, a masterpiece of design.",
    ),
    (
        "08_runon_clauses",
        "af_heart",
        "When the committee finally assembled after weeks of delays, cancellations, rescheduled flights, "
        "missing paperwork, and two separate hotel mixups, the chairman, visibly exhausted but determined "
        "to proceed, opened the meeting with a lengthy statement about procedure, precedent, patience, "
        "and the pressing need to conclude before the holidays, which everyone privately doubted was possible.",
    ),
    (
        "09_tags",
        "af_heart",
        "First a normal sentence. [pause:1.5s] Then after a pause, [voice:am_adam] a different voice finishes the thought.",
    ),
    (
        "10_zh_commas",
        "zf_xiaobei",
        "他走进房间，放下行李，看了看窗外，然后叹了口气。天色渐暗，远处的灯光一盏盏亮起，街道安静得出奇。",
    ),
    (
        "11_zh_terminators",
        "zf_xiaobei",
        "这是第一句。这真的可能吗？当然可能！我们明天就出发。",
    ),
]

NEWLINE_CASE = (
    "12_newlines_nonorm",
    "af_heart",
    "Chapter One\n\nThe rain had not stopped for three days. Rivers crept up their banks.",
)


def synth(label_dir: Path, name: str, voice: str, text: str, normalize: bool = True):
    payload = {
        "model": "kokoro",
        "input": text,
        "voice": voice,
        "response_format": "mp3",
        "stream": False,
    }
    if not normalize:
        payload["normalization_options"] = {"normalize": False}
    r = requests.post(URL, json=payload, timeout=120)
    r.raise_for_status()
    out = label_dir / f"{name}.mp3"
    out.write_bytes(r.content)
    print(f"  {out.name}  ({len(r.content) / 1024:.0f} KB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, choices=["old", "new"])
    args = parser.parse_args()

    label_dir = Path(__file__).parent / "ab_split" / args.label
    label_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing {label_dir}")

    for name, voice, text in CASES:
        synth(label_dir, name, voice, text)
    name, voice, text = NEWLINE_CASE
    synth(label_dir, name, voice, text, normalize=False)


if __name__ == "__main__":
    main()
