"""
Long-document support.

RedactX is trained on texts of at most `max_chars` characters (600 by default). Production documents
(discharge summaries, transcripts, logs) are much longer, so they are split into overlapping windows,
each window is scored independently, and window-local spans are mapped back to global offsets and
merged.

Guarantee: any span no longer than `overlap` characters lies completely inside at least one window,
so it can be detected in full regardless of where the window boundaries fall.
"""

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class Window:
    start: int
    end: int

    def slice(self, text: str) -> str:
        return text[self.start:self.end]


def chunk_text(text: str, max_chars: int = 600, overlap: int = 150) -> List[Window]:
    """
    Splits `text` into windows of at most `max_chars` characters. Consecutive windows overlap by at
    least `overlap` characters. Window ends are moved back to the nearest whitespace when one exists in
    the last quarter of the window; window starts are moved *backwards* (never forwards) to a word
    boundary, which can only increase the overlap.

    Proof of the guarantee: window i+1 starts at s_{i+1} <= e_i - overlap. For a span [a, b) with
    b - a <= overlap, take the last window i with s_i <= a. If b > e_i then a > e_i - overlap >= s_{i+1},
    contradicting the choice of i, unless i is the final window (which ends at len(text) >= b).
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if not 0 <= overlap <= max_chars // 2:
        raise ValueError("overlap must be in [0, max_chars // 2]")
    n = len(text)
    if n <= max_chars:
        return [Window(0, n)]

    windows: List[Window] = []
    start = 0
    min_window = max_chars - max_chars // 4
    while True:
        end = min(start + max_chars, n)
        if end < n:
            cut = max(text.rfind(" ", start + min_window, end), text.rfind("\n", start + min_window, end))
            if cut > start:
                end = cut
        windows.append(Window(start, end))
        if end >= n:
            return windows
        next_start = end - overlap
        lo = max(start + 1, next_start - overlap // 2)
        ws = max(text.rfind(" ", lo, next_start), text.rfind("\n", lo, next_start))
        if ws != -1:
            next_start = ws + 1
        start = max(next_start, start + 1)


Interval = Tuple[int, int]


def merge_intervals(intervals: Iterable[Interval], join_gap: int = 0) -> List[Interval]:
    """Union of half-open intervals; intervals separated by <= join_gap characters are joined."""
    out: List[List[int]] = []
    for s, e in sorted(intervals):
        if e <= s:
            continue
        if out and s <= out[-1][1] + join_gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def to_global(window: Window, local_spans: Sequence[Tuple[int, int]]) -> List[Interval]:
    return [(window.start + s, window.start + e) for s, e in local_spans]
