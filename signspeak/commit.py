"""
Debounce layer for static letters — the piece INTEGRATION_TODO.md deferred
until a real gesture source existed. The camera source is that real source.

Problem it solves: the static classifier fires every frame, so holding "H"
for a second would append "hhhhhhhh" to the word buffer. Raw appends corrupt
words; naively dropping repeats breaks genuine doubles ("hello").

Approach (commit-and-rearm):
  * A letter is COMMITTED once when it wins a majority vote over a sliding
    window of frames (smooths single-frame flickers).
  * After a commit the same letter is suppressed until the committer RE-ARMS,
    which happens when the winning letter changes or the hand leaves the
    frame. To sign a genuine double letter, briefly drop or move the hand
    between the two — the natural "bounce" fingerspellers already do.

This module deliberately has no OpenCV/MediaPipe dependencies so the logic
can be unit-tested headlessly.
"""

from collections import Counter, deque
from typing import Optional, Tuple

from . import config


class StaticCommitter:
    """Turns a per-frame stream of (letter, confidence) predictions into
    discrete, debounced commit events."""

    def __init__(self,
                 window: int = None,
                 min_votes: int = None,
                 control_labels: set = None,
                 control_min_votes: int = None):
        self.window = window or config.STATIC_SMOOTH_WINDOW
        self.min_votes = min_votes or config.STATIC_MIN_VOTES
        # Control signs trigger actions, so they must clear a higher bar.
        self.control_labels = control_labels or set()
        self.control_min_votes = min(
            control_min_votes or config.CONTROL_MIN_VOTES, self.window)
        self._recent: deque = deque(maxlen=self.window)
        self._confs: dict = {}
        self._last_committed: Optional[str] = None

    def _required_votes(self, label: str) -> int:
        return self.control_min_votes if label in self.control_labels else self.min_votes

    # -- events from the camera loop ------------------------------------

    def hand_lost(self) -> None:
        """Hand left the frame: clear votes and re-arm for repeats."""
        self._recent.clear()
        self._confs.clear()
        self._last_committed = None

    def interrupt(self) -> None:
        """A motion gesture started/ended: drop votes so the settling pose
        doesn't leak into static recognition (mirrors old recognize.py)."""
        self._recent.clear()
        self._confs.clear()

    def update(self, letter: str, confidence: float) -> Optional[Tuple[str, float]]:
        """Feed one frame's static prediction.

        Returns (letter, mean_confidence) exactly once when a letter newly
        wins the vote; otherwise None.
        """
        self._recent.append(letter)
        self._confs.setdefault(letter, deque(maxlen=self.window)).append(confidence)

        top, votes = Counter(self._recent).most_common(1)[0]
        if votes < self._required_votes(top):
            return None
        if top != letter:            # this frame isn't the winner; wait
            return None
        if top == self._last_committed:
            return None              # already committed; suppressed until re-arm

        # A different letter took over the vote -> re-armed implicitly.
        self._last_committed = top
        confs = self._confs.get(top, [confidence])
        return top, float(sum(confs) / len(confs))

    # -- display helpers -------------------------------------------------

    @property
    def current_winner(self) -> str:
        """Best current guess for on-screen display ('' if unstable)."""
        if not self._recent:
            return ""
        top, votes = Counter(self._recent).most_common(1)[0]
        return top if votes >= self._required_votes(top) else ""