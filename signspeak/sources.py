"""
Gesture sources for the pipeline.

A *source* is any iterable that yields ``(symbol, confidence)`` tuples, where
``symbol`` is a letter or a control gesture ('neutral', 'space', 'speak') and
``confidence`` is a float in [0, 1]. This duck-typed contract lets the fake
source used for development be swapped for a real ESP32 reader without changing
the pipeline.
"""

from typing import Iterator, Tuple

Gesture = Tuple[str, float]


def fake_source() -> Iterator[Gesture]:
    """Scripted stream spelling 'iamhungrypleasegivemefood', then 'speak'.

    One letter is given low confidence to exercise the pipeline's gate.
    """
    letters = "iamhungrypleasegivemefood"
    for index, letter in enumerate(letters):
        yield letter, 0.60 if index == 3 else 0.95
    yield "neutral", 0.99
    yield "speak", 0.98


# The real source now lives in signspeak/camera_source.py: webcam frames ->
# MediaPipe landmarks -> the trained static/motion models -> (symbol,
# confidence). It fulfils this exact contract, so pipeline.py is unchanged.
