"""Ties spoken words back to the input words they were normalized from (#368)."""

from bisect import bisect_left, bisect_right
from typing import Callable, Dict, Iterable, List, Optional

from ...structures.schemas import WordTimestamp

RESYNC_WINDOW = 32
RESYNC_CONFIRM = 2
LOOKAHEAD_CHARS = 64


def _key(word: str) -> str:
    return "".join(ch for ch in word.casefold() if ch.isalnum()) or word


def align_words(
    original: str, normalized: str, spoken_form: Callable[[str], str]
) -> List[Optional[str]]:
    """Return the input word behind each whitespace word of normalized.

    Each input word is normalized alone, which reproduces the real output for
    context-free rewrites (105 -> one hundred and five). Where context changes
    the output ($5 million -> five million dollars) the walk resyncs, and the
    words in between get every input word they came from.
    """
    originals = original.split()
    spoken = [_key(word) for word in normalized.split()]
    if not originals:
        return [None] * len(spoken)

    cache: Dict[str, List[str]] = {}
    hint: List[str] = []
    source: List[int] = []
    for index, word in enumerate(originals):
        if word not in cache:
            cache[word] = [_key(piece) for piece in spoken_form(word).split()]
        hint.extend(cache[word])
        source.extend([index] * len(cache[word]))

    def matches(i: int, j: int) -> bool:
        for step in range(RESYNC_CONFIRM):
            if i + step == len(hint) and j + step == len(spoken):
                return True
            if i + step >= len(hint) or j + step >= len(spoken):
                return False
            if hint[i + step] != spoken[j + step]:
                return False
        return True

    positions: Dict[str, List[int]] = {}
    for index, key in enumerate(spoken):
        positions.setdefault(key, []).append(index)

    def resync(i: int, j: int) -> tuple[int, int]:
        # indexed so a long run of mismatches stays linear on 1M character inputs
        best: Optional[tuple[int, int, int]] = None
        if len(hint) - i <= RESYNC_WINDOW and len(spoken) - j <= RESYNC_WINDOW:
            best = (len(hint) - i + len(spoken) - j, len(hint), len(spoken))
        for di in range(min(RESYNC_WINDOW, len(hint) - i - 1) + 1):
            if best is not None and di >= best[0]:
                break
            spots = positions.get(hint[i + di], [])
            for spot in spots[bisect_left(spots, j) :]:
                dj = spot - j
                if dj > RESYNC_WINDOW or (best is not None and di + dj >= best[0]):
                    break
                if matches(i + di, spot):
                    best = (di + dj, i + di, spot)
                    break
        if best is None:
            return min(i + 1, len(hint)), j + 1
        return best[1], best[2]

    result: List[Optional[str]] = [None] * len(spoken)
    i = j = 0
    while j < len(spoken):
        if i < len(hint) and hint[i] == spoken[j]:
            result[j] = originals[source[i]]
            i += 1
            j += 1
            continue
        next_i, next_j = resync(i, j)
        if next_i > i:
            words = originals[source[i] : source[next_i - 1] + 1]
        else:
            # spoken words the per-word pass never produced belong to their neighbour
            words = [originals[source[max(i - 1, 0)]]]
        for gap in range(j, min(next_j, len(spoken))):
            result[gap] = " ".join(words)
        i, j = next_i, next_j
    return result


class OriginalWords:
    """One request's spoken words in order, each tied to its input word."""

    def __init__(self) -> None:
        self._text = ""
        self._starts: List[int] = []
        self._sources: List[Optional[str]] = []
        self._cursor = 0

    def add(
        self, original: str, normalized: str, spoken_form: Callable[[str], str]
    ) -> None:
        words = normalized.split()
        offset = len(self._text)
        for word in words:
            self._starts.append(offset)
            offset += len(word)
        self._sources.extend(align_words(original, normalized, spoken_form))
        self._text += "".join(words)

    def annotate(self, timestamps: Iterable[WordTimestamp]) -> None:
        for timestamp in timestamps:
            word = "".join(timestamp.word.split())
            if not word:
                continue
            # bounded so a word the model reworded cannot drag the cursor far ahead
            found = self._text.find(
                word, self._cursor, self._cursor + len(word) + LOOKAHEAD_CHARS
            )
            if found < 0:
                continue
            timestamp.original_word = self._sources[
                bisect_right(self._starts, found) - 1
            ]
            self._cursor = found + len(word)
