"""
Two-hand word signs: everything the word model needs apart from camera I/O.

Like commit.py, this module has no OpenCV or MediaPipe imports, so all of it
can be unit-tested headlessly. The camera loop and the collector feed it
plain (21, 3) landmark arrays.

  HandPairer      decides which detected hand is the dominant one and keeps
                  that assignment stable from frame to frame
  ModeArbiter     letter mode with one hand in view, word mode with two.
                  Measured, not trained, so it needs no data and can never
                  be confused with a real sign (the webcam equivalent of the
                  glove's "left hand raised" switch)
  WordSegmenter   the two-hand counterpart of tracking.GestureSegmenter: cuts
                  one sign out of the stream, from motion onset (plus a short
                  pre-roll) to the moment both hands settle
  word_features   one recorded sign -> a fixed 256-d vector

The collector and the live camera loop both run signs through the same
WordSegmenter and the same word_features, so the model is trained on exactly
the kind of segment it will be shown live.
"""

from collections import deque
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import config

# MediaPipe's 21-point layout (same as tracking.py).
WRIST = 0
INDEX_TIP = 8
MIDDLE_MCP = 9
PALM_POINTS = (0, 5, 9, 13, 17)   # wrist + the four knuckles

DOM, NONDOM = 0, 1                 # slot order used everywhere below

Hand = Optional[np.ndarray]        # (21, 3) landmarks, or None if not seen
Frame = Tuple[Hand, Hand]          # (dominant, non-dominant)


# --------------------------------------------------------------------------
# Which hand is which
# --------------------------------------------------------------------------

class HandPairer:
    """Assign detected hands to (dominant, non-dominant) slots.

    MediaPipe returns hands in no fixed order, so two hands are assigned by:

      1. continuity: if both slots were seen last frame, keep each hand in
         the slot whose previous wrist position is nearest. This holds the
         identities through signs where the hands cross.
      2. image position: the frame is mirrored before detection, so the
         signer's right hand appears on the right of the image.

    MediaPipe's own handedness label is deliberately NOT used to order two
    hands. Whether it assumes mirrored input differs between MediaPipe
    versions, and an inverted label would lock in the wrong assignment that
    continuity would then preserve. Image position is plain geometry.

    A single visible hand goes to the slot it is nearest to (falling back to
    its label, then to dominant), which keeps the slots stable when one hand
    briefly drops out mid-sign.
    """

    def __init__(self, dominant: str = None):
        self.dominant = (dominant or config.DOMINANT_HAND).capitalize()
        self._prev: List[Optional[np.ndarray]] = [None, None]   # wrist xy per slot

    def reset(self) -> None:
        self._prev = [None, None]

    def _slot_from_label(self, label: Optional[str]) -> Optional[int]:
        if not label:
            return None
        return DOM if label.capitalize() == self.dominant else NONDOM

    def _dom_is_right_of_image(self) -> bool:
        # Mirrored view: the signer's right hand has the larger x.
        return self.dominant == "Right"

    def pair(self, hands: Sequence[Tuple[np.ndarray, Optional[str]]]) -> Frame:
        """hands: [(landmarks (21, 3), handedness label or None), ...].

        Returns (dominant, non_dominant); either may be None.
        """
        if not hands:
            self.reset()
            return None, None

        hands = list(hands)[:2]
        arrs = [np.asarray(h[0], dtype=np.float32) for h in hands]
        wrists = [a[WRIST, :2] for a in arrs]

        if len(hands) == 1:
            slot = self._nearest_slot(wrists[0])
            if slot is None:
                slot = self._slot_from_label(hands[0][1])
            if slot is None:
                slot = DOM
            out: List[Hand] = [None, None]
            out[slot] = arrs[0]
            self._prev[slot] = wrists[0]
            return out[DOM], out[NONDOM]

        # Two hands.
        order = None
        if self._prev[DOM] is not None and self._prev[NONDOM] is not None:
            keep = (np.linalg.norm(wrists[0] - self._prev[DOM])
                    + np.linalg.norm(wrists[1] - self._prev[NONDOM]))
            swap = (np.linalg.norm(wrists[1] - self._prev[DOM])
                    + np.linalg.norm(wrists[0] - self._prev[NONDOM]))
            order = (0, 1) if keep <= swap else (1, 0)
        if order is None:
            right_first = wrists[0][0] >= wrists[1][0]
            dom_first = right_first == self._dom_is_right_of_image()
            order = (0, 1) if dom_first else (1, 0)

        dom, nondom = arrs[order[0]], arrs[order[1]]
        self._prev = [dom[WRIST, :2], nondom[WRIST, :2]]
        return dom, nondom

    def _nearest_slot(self, wrist: np.ndarray) -> Optional[int]:
        best, best_d = None, np.inf
        for slot in (DOM, NONDOM):
            if self._prev[slot] is not None:
                d = float(np.linalg.norm(wrist - self._prev[slot]))
                if d < best_d:
                    best, best_d = slot, d
        return best


# --------------------------------------------------------------------------
# Letter mode or word mode
# --------------------------------------------------------------------------

class ModeArbiter:
    """One hand in view: letter mode. Two hands: word mode.

    Hysteresis in both directions, so MediaPipe dropping a hand for a frame
    or two (fast motion, partial occlusion) does not flip the mode in the
    middle of a sign.
    """

    LETTER = "letter"
    WORD = "word"

    def __init__(self, enter_frames: int = None, exit_frames: int = None):
        self.enter_frames = enter_frames or config.WORD_MODE_ENTER_FRAMES
        self.exit_frames = exit_frames or config.WORD_MODE_EXIT_FRAMES
        self.mode = self.LETTER
        self._run = 0

    def update(self, n_hands: int) -> bool:
        """Feed this frame's hand count. Returns True if the mode changed."""
        two = n_hands >= 2
        if self.mode == self.LETTER:
            self._run = self._run + 1 if two else 0
            if self._run >= self.enter_frames:
                self.mode, self._run = self.WORD, 0
                return True
        else:
            self._run = self._run + 1 if not two else 0
            if self._run >= self.exit_frames:
                self.mode, self._run = self.LETTER, 0
                return True
        return False

    @property
    def is_word(self) -> bool:
        return self.mode == self.WORD


# --------------------------------------------------------------------------
# Cutting one sign out of the stream
# --------------------------------------------------------------------------

class WordSegmenter:
    """Two-hand gesture segmentation with a settle-first start.

    States:
      SETTLING  after a reset or a lost hand: wait until the hands have been
                still for `still_frames_to_end` frames. This stops the act of
                raising the second hand into view (which is what switches to
                word mode) from being captured as a sign.
      IDLE      ready. A rolling pre-roll of recent frames is kept so the
                start of the motion is not lost.
      MOVING    buffering. Ends when the hands settle, when both hands are
                lost, or when `max_frames` is reached.

    Start and stop use the faster of the two wrists (image units per frame),
    so a sign where one hand stays still and the other moves is still
    caught. The `min_path` check uses the movement of BOTH hands added
    together, so a symmetric sign where each hand travels only a short way
    (hands closing together, or opening apart) still counts as a real sign.
    A hand that was missing on the previous frame contributes nothing, so a
    hand re-appearing in a new place is not mistaken for a fast movement.
    """

    SETTLING, IDLE, MOVING = "SETTLING", "IDLE", "MOVING"

    def __init__(self, move_on=0.018, move_off=0.007, still_frames_to_end=8,
                 min_frames=10, min_path=0.15, max_frames=150, preroll=4):
        self.move_on = move_on
        self.move_off = move_off
        self.still_frames_to_end = still_frames_to_end
        self.min_frames = min_frames
        self.min_path = min_path
        self.max_frames = max_frames
        self.preroll = preroll
        self.reset()

    def reset(self) -> None:
        self.state = self.SETTLING
        # Why the last segment ended: "settled" (a clean sign), "max_frames"
        # (the hands never came to rest) or "hands_lost". The collector
        # keeps only "settled" takes.
        self.end_reason: Optional[str] = None
        self.last_speed = 0.0          # this frame's start/stop speed, for tuning
        self.buffer: List[Frame] = []
        self._pre: deque = deque(maxlen=max(0, self.preroll))
        self._pre_moved: deque = deque(maxlen=max(0, self.preroll))  # their movement
        self._prev: List[Optional[np.ndarray]] = [None, None]
        self._still = 0
        self._slow = 0
        self.path_len = 0.0

    @property
    def is_moving(self) -> bool:
        return self.state == self.MOVING

    @property
    def is_ready(self) -> bool:
        return self.state == self.IDLE

    def _speed(self, frame: Frame) -> Tuple[float, float]:
        """(faster wrist's step, both wrists' steps added) for this frame."""
        fastest, total = 0.0, 0.0
        for slot in (DOM, NONDOM):
            hand = frame[slot]
            if hand is None:
                self._prev[slot] = None
                continue
            wrist = np.asarray(hand, dtype=np.float32)[WRIST, :2]
            if self._prev[slot] is not None:
                step = float(np.linalg.norm(wrist - self._prev[slot]))
                fastest = max(fastest, step)
                total += step
            self._prev[slot] = wrist
        return fastest, total

    def update(self, dom: Hand, nondom: Hand) -> Optional[List[Frame]]:
        """Process one frame. Returns the finished sign (a list of
        (dom, nondom) frames) the moment one ends, otherwise None."""
        if dom is None and nondom is None:
            finished = self._finish(next_state=self.SETTLING, reason="hands_lost")
            self._prev = [None, None]
            self._still = 0
            return finished

        frame = (None if dom is None else np.asarray(dom, dtype=np.float32),
                 None if nondom is None else np.asarray(nondom, dtype=np.float32))
        speed, moved = self._speed(frame)
        self.last_speed = speed

        if self.state == self.SETTLING:
            self._still = self._still + 1 if speed < self.move_off else 0
            self._pre.append(frame)
            self._pre_moved.append(moved)
            if self._still >= self.still_frames_to_end:
                self.state = self.IDLE
                self._still = 0
            return None

        if self.state == self.IDLE:
            if speed > self.move_on:
                self.state = self.MOVING
                self.buffer = list(self._pre) + [frame]
                # The pre-roll is part of the sign, so its movement counts
                # toward min_path too. A smooth stroke does much of its
                # accelerating before it crosses move_on.
                self.path_len = sum(self._pre_moved) + moved
                self._pre.clear()
                self._pre_moved.clear()
                self._slow = 0
            else:
                self._pre.append(frame)
                self._pre_moved.append(moved)
            return None

        # MOVING
        self.buffer.append(frame)
        self.path_len += moved
        if len(self.buffer) >= self.max_frames:
            return self._finish(next_state=self.SETTLING, reason="max_frames")
        if speed < self.move_off:
            self._slow += 1
            if self._slow >= self.still_frames_to_end:
                return self._finish(next_state=self.IDLE, reason="settled")
        else:
            self._slow = 0
        return None

    def _finish(self, next_state: str, reason: str) -> Optional[List[Frame]]:
        gesture = None
        if (self.state == self.MOVING
                and len(self.buffer) >= self.min_frames
                and self.path_len >= self.min_path):
            gesture = self.buffer
            self.end_reason = reason
        self.buffer = []
        self._slow = 0
        self.path_len = 0.0
        self._pre.clear()
        self._pre_moved.clear()
        # A sign that ended by settling leaves the hands already still, so
        # the next one can start straight away (next_state IDLE). Hand loss
        # or the length cap means settle first. SETTLING stays SETTLING.
        if self.state != self.SETTLING:
            self.state = next_state
        return gesture


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------

TRAJ_LEN = config.WORD_TRAJ_LEN
N_FEATURES = config.WORD_FEATURES

# Layout of the feature vector, in order. Kept as data so tests and the
# report can name every block.
FEATURE_BLOCKS = (
    ("dom_trajectory", 2 * TRAJ_LEN),      # wrist path, relative to both-wrist centroid
    ("nondom_trajectory", 2 * TRAJ_LEN),
    ("palm_gap_curve", TRAJ_LEN),          # palm-to-palm distance over time
    ("dom_handshape_mean", 63),            # wrist-centred, hand-size-scaled
    ("nondom_handshape_mean", 63),
    ("dom_finger_motion", 21),             # per-landmark spread over time
    ("nondom_finger_motion", 21),
    ("motion_summary", 4),                 # mean speed and path length per hand
    ("contact_and_repetition", 4),         # closest approach, direction reversals
)
assert sum(n for _, n in FEATURE_BLOCKS) == N_FEATURES, "feature layout mismatch"


def stack_sequence(seq: Sequence[Frame]) -> np.ndarray:
    """List of (dom, nondom) frames -> (T, 2, 21, 3) float32, NaN where a
    hand was not seen. This is the format raw takes are saved in."""
    out = np.full((len(seq), 2, 21, 3), np.nan, dtype=np.float32)
    for t, frame in enumerate(seq):
        for slot in (DOM, NONDOM):
            if frame[slot] is not None:
                out[t, slot] = np.asarray(frame[slot], dtype=np.float32)
    return out


def _fill_track(track: np.ndarray, max_missing: float) -> Optional[np.ndarray]:
    """One hand's (T, 21, 3) track with NaN gaps -> gaps filled from the
    nearest seen frame, or None if it is missing too often."""
    track = np.array(track, dtype=np.float32, copy=True)
    T = track.shape[0]
    if T < 2:
        return None
    seen = ~np.isnan(track[:, WRIST, 0])
    if seen.sum() == 0 or (1.0 - seen.mean()) > max_missing:
        return None
    if not seen.all():
        idx = np.arange(T)
        seen_idx = idx[seen]
        nearest = seen_idx[np.abs(idx[:, None] - seen_idx[None, :]).argmin(axis=1)]
        track = track[nearest]
    return track


def fill_gaps(arr: np.ndarray, max_missing: float = None) -> Optional[np.ndarray]:
    """Fill frames where a hand was missing with its nearest seen frame.

    Returns None if either hand is missing from more than `max_missing` of
    the frames, since that sign cannot be described reliably.
    """
    max_missing = config.WORD_MAX_MISSING if max_missing is None else max_missing
    arr = np.asarray(arr, dtype=np.float32)
    tracks = [_fill_track(arr[:, slot], max_missing) for slot in (DOM, NONDOM)]
    if any(t is None for t in tracks):
        return None
    return np.stack(tracks, axis=1)


def _resample(seq: np.ndarray, target_len: int) -> np.ndarray:
    """(n, d) -> (target_len, d) by linear interpolation along time."""
    seq = np.asarray(seq, dtype=np.float32)
    n = len(seq)
    if n == target_len:
        return seq
    old_t = np.linspace(0.0, 1.0, n)
    new_t = np.linspace(0.0, 1.0, target_len)
    return np.stack([np.interp(new_t, old_t, seq[:, i]) for i in range(seq.shape[1])],
                    axis=1)


def _normalize_hands(arr: np.ndarray) -> np.ndarray:
    """Vectorised tracking.normalize_landmarks over (T, 2, 21, 3).

    Same maths: wrist to origin, divide by wrist-to-middle-knuckle length.
    Duplicated here (and checked equal in the tests) so this module stays
    free of the OpenCV/MediaPipe imports that tracking.py pulls in.
    """
    coords = arr - arr[:, :, WRIST:WRIST + 1, :]
    scale = np.linalg.norm(coords[:, :, MIDDLE_MCP, :], axis=-1)        # (T, 2)
    safe = np.where(scale > 1e-6, scale, 1.0)
    return coords / safe[:, :, None, None]


def _reversals(x: np.ndarray, min_step: float) -> int:
    """Direction changes in a 1-D track, ignoring steps smaller than
    `min_step` (tracking jitter). Counts taps and shakes."""
    v = np.diff(np.asarray(x, dtype=np.float64))
    v = v[np.abs(v) > min_step]
    if len(v) < 2:
        return 0
    s = np.sign(v)
    return int(np.sum(s[1:] != s[:-1]))


def word_features_from_array(raw: np.ndarray) -> Optional[np.ndarray]:
    """(T, 2, 21, 3) with NaN gaps -> 256-d feature vector, or None.

    Every block is invariant to where the signer sits in the frame and to
    their distance from the camera: positions are taken relative to the
    centroid of the two wrists and measured in hand lengths (median
    wrist-to-middle-knuckle distance), not image units. Absolute screen
    position is deliberately left out, since it changes every time you sit
    down and would teach the model where you sat during collection.
    """
    arr = fill_gaps(raw)
    if arr is None:
        return None
    T = arr.shape[0]
    xy = arr[..., :2]                                                    # (T, 2, 21, 2)

    hand_len = np.linalg.norm(xy[:, :, MIDDLE_MCP] - xy[:, :, WRIST], axis=-1)
    s = float(np.median(hand_len))
    if not np.isfinite(s) or s < 1e-6:
        return None

    wrists = xy[:, :, WRIST]                                             # (T, 2, 2)
    origin = wrists.reshape(-1, 2).mean(axis=0)
    rel = (wrists - origin) / s
    traj_dom = _resample(rel[:, DOM], TRAJ_LEN).flatten()
    traj_nondom = _resample(rel[:, NONDOM], TRAJ_LEN).flatten()

    palms = xy[:, :, PALM_POINTS].mean(axis=2)                           # (T, 2, 2)
    gap = np.linalg.norm(palms[:, DOM] - palms[:, NONDOM], axis=-1) / s  # (T,)
    gap_curve = _resample(gap[:, None], TRAJ_LEN).flatten()

    shapes = _normalize_hands(arr)                                       # (T, 2, 21, 3)
    shape_mean = shapes.mean(axis=0).reshape(2, 63)
    finger_motion = shapes.std(axis=0).mean(axis=-1)                     # (2, 21)

    steps = np.linalg.norm(np.diff(wrists, axis=0), axis=-1) / s         # (T-1, 2)
    motion = np.concatenate([steps.mean(axis=0), steps.sum(axis=0)])     # 4

    tip_to_palm = np.linalg.norm(xy[:, DOM, INDEX_TIP] - palms[:, NONDOM], axis=-1) / s
    contact = np.array([
        gap.min(),
        tip_to_palm.min(),
        _reversals(rel[:, DOM, 1], config.WORD_REVERSAL_MIN_STEP),
        _reversals(rel[:, NONDOM, 1], config.WORD_REVERSAL_MIN_STEP),
    ], dtype=np.float32)

    feats = np.concatenate([
        traj_dom, traj_nondom, gap_curve,
        shape_mean[DOM], shape_mean[NONDOM],
        finger_motion[DOM], finger_motion[NONDOM],
        motion, contact,
    ]).astype(np.float32)
    if feats.shape[0] != N_FEATURES or not np.all(np.isfinite(feats)):
        return None
    return feats


def word_features(seq: Sequence[Frame]) -> Optional[np.ndarray]:
    """A segmented sign (list of (dom, nondom) frames) -> 256-d vector, or
    None if a hand was missing for too much of it."""
    return word_features_from_array(stack_sequence(seq))


# --------------------------------------------------------------------------
# One-hand signs (one-hand words, J and Z)
# --------------------------------------------------------------------------
# Cut by a WordSegmenter fed only the fingerspelling hand, so they are
# segmented exactly like the two-hand words, in collection and live.

ONEHAND_N_FEATURES = config.ONEHAND_FEATURES

ONEHAND_FEATURE_BLOCKS = (
    ("trajectory", 2 * TRAJ_LEN),          # wrist path around its centre, in hand lengths
    ("toward_camera_curve", TRAJ_LEN),     # apparent hand size over time
    ("handshape_start", 63),               # before and as the movement begins
    ("handshape_middle", 63),
    ("handshape_end", 63),
    ("finger_motion", 21),                 # per-landmark spread over time
    ("motion_summary", 2),                 # mean speed, path length
    ("repetition", 2),                     # vertical and horizontal reversals
)
assert sum(n for _, n in ONEHAND_FEATURE_BLOCKS) == ONEHAND_N_FEATURES, \
    "one-hand feature layout mismatch"


def stack_onehand(seq: Sequence) -> np.ndarray:
    """Frames from a one-hand segment -> (T, 21, 3), NaN where unseen.

    Accepts the (hand, None) tuples WordSegmenter returns, or bare arrays.
    This is the format one-hand raw takes are saved in.
    """
    out = np.full((len(seq), 21, 3), np.nan, dtype=np.float32)
    for t, frame in enumerate(seq):
        hand = frame[0] if isinstance(frame, tuple) else frame
        if hand is not None:
            out[t] = np.asarray(hand, dtype=np.float32)
    return out


def onehand_features_from_array(raw: np.ndarray) -> Optional[np.ndarray]:
    """(T, 21, 3) with NaN gaps -> 262-d one-hand feature vector, or None.

    Like the two-hand features: position-free and measured in hand lengths,
    so independent of where you sit and how far from the camera. MediaPipe
    gives the wrist no real depth, so movement toward or away from the
    camera (tapping the chest, a forward flick) is captured instead through
    the hand's apparent size over time.
    """
    track = _fill_track(raw, config.WORD_MAX_MISSING)
    if track is None:
        return None
    T = track.shape[0]
    xy = track[..., :2]

    hand_len = np.linalg.norm(xy[:, MIDDLE_MCP] - xy[:, WRIST], axis=-1)   # (T,)
    s = float(np.median(hand_len))
    if not np.isfinite(s) or s < 1e-6:
        return None

    wrist = xy[:, WRIST]
    rel = (wrist - wrist.mean(axis=0)) / s
    traj = _resample(rel, TRAJ_LEN).flatten()
    size_curve = _resample((hand_len / s)[:, None], TRAJ_LEN).flatten()

    shapes = _normalize_hands(track[:, None])[:, 0]                         # (T, 21, 3)
    k = max(1, T // 6)
    mid0 = max(0, T // 2 - k // 2)
    start = shapes[:k].mean(axis=0).flatten()
    middle = shapes[mid0:mid0 + k].mean(axis=0).flatten()
    end = shapes[-k:].mean(axis=0).flatten()
    finger_motion = shapes.std(axis=0).mean(axis=-1)                        # (21,)

    steps = np.linalg.norm(np.diff(wrist, axis=0), axis=-1) / s
    motion = np.array([steps.mean(), steps.sum()], dtype=np.float32)
    repetition = np.array([
        _reversals(rel[:, 1], config.WORD_REVERSAL_MIN_STEP),
        _reversals(rel[:, 0], config.WORD_REVERSAL_MIN_STEP),
    ], dtype=np.float32)

    feats = np.concatenate([traj, size_curve, start, middle, end,
                            finger_motion, motion, repetition]).astype(np.float32)
    if feats.shape[0] != ONEHAND_N_FEATURES or not np.all(np.isfinite(feats)):
        return None
    return feats


def onehand_features(seq: Sequence) -> Optional[np.ndarray]:
    """A segmented one-hand sign -> 262-d vector, or None."""
    return onehand_features_from_array(stack_onehand(seq))
