"""Bounds on the pause-tag path."""

import pytest

from api.src.core.config import settings
from api.src.services.text_processing.text_processor import (
    check_pause_budget,
    smart_split,
)


async def _durations(text):
    return [pause async for _t, _tok, pause in smart_split(text) if pause]


@pytest.mark.asyncio
async def test_pause_tags_still_split_into_pause_chunks():
    assert await _durations("hello [pause:2s] there [pause:3s] friend") == [2.0, 3.0]


def test_pause_budget_allows_normal_dialogue():
    check_pause_budget("hello [pause:2s] there [pause:3s] friend")


def test_pause_budget_rejects_many_tags():
    text = " ".join(["[pause:60s]"] * 4545)  # the reported ~50 KB OOM payload
    with pytest.raises(ValueError, match="exceeds"):
        check_pause_budget(text)


def test_pause_budget_counts_total_not_per_tag():
    """Every tag is under max_pause_duration_s, the sum is not."""
    tags = int(settings.max_total_pause_s // 10) + 2
    with pytest.raises(ValueError, match="exceeds"):
        check_pause_budget(" ".join(["[pause:10s]"] * tags))


def test_pause_budget_counts_across_voice_tags():
    """Voice/rate segmentation must not reset the budget."""
    with pytest.raises(ValueError, match="exceeds"):
        check_pause_budget("[voice:af_bella] [pause:60s] " * 6)


def test_pause_budget_exact_boundary_passes():
    """1000 x 0.3s is exactly the 300s default; float error must not reject it."""
    check_pause_budget(" ".join(["[pause:0.3s]"] * 1000))


def test_pause_free_text_unaffected():
    check_pause_budget("just some ordinary text")
