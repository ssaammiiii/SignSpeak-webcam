"""
Shared building blocks for the fingerspelling pipeline.

Covers three things every other script needs:
  1. MediaPipe HandLandmarker setup + model download.
  2. Turning a detected hand into feature vectors:
       - static letters  -> normalize_landmarks() (one frame, 63-d)
       - motion letters   -> motion_features()     (a sequence, 95-d)
  3. GestureSegmenter: a small state machine that decides, at runtime,
     whether the hand is holding still (static letter) or drawing a
     motion (J / Z), and hands back a completed gesture when one ends.
"""

import os
import urllib.request

# Silence MediaPipe/absl telemetry noise (the "Failed to send to clearcut"
# errors are Google analytics pings failing - harmless, but spammy).
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "3")

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions

from . import config

MODEL_PATH = str(config.LANDMARKER_PATH)
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)

# MediaPipe's 21-point layout: 0 wrist, 1-4 thumb, 5-8 index, 9-12 middle,
# 13-16 ring, 17-20 pinky (each finger knuckle -> tip).
WRIST = 0
MIDDLE_MCP = 9   # middle-finger knuckle: the scale reference for normalization

# How many frames every motion gesture is resampled to before feature
# extraction, so gestures drawn fast or slow become directly comparable.
TRAJ_LEN = config.TRAJ_LEN


# --------------------------------------------------------------------------
# MediaPipe setup
# --------------------------------------------------------------------------

def ensure_model():
    """Download the hand_landmarker.task bundle if it isn't already present."""
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        print(f"Downloading hand landmark model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")
    return MODEL_PATH


def create_landmarker(num_hands=1):
    """Create a HandLandmarker configured for VIDEO (webcam) mode."""
    ensure_model()
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return vision.HandLandmarker.create_from_options(options)


def bgr_frame_to_mp_image(frame_bgr):
    """Convert an OpenCV BGR frame to a mediapipe.Image (SRGB)."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)


def landmarks_to_array(hand_landmarks):
    """Convert MediaPipe's 21 landmark objects into a (21, 3) float array."""
    return np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)


def hands_with_labels(result):
    """Every detected hand as [((21, 3) array, "Left"/"Right" or None), ...].

    Used by the two-hand word path. The label is informational only: the
    word code orders hands by image position (see words.HandPairer).
    """
    hands = []
    for i, hand in enumerate(result.hand_landmarks):
        label = None
        if i < len(result.handedness) and result.handedness[i]:
            label = result.handedness[i][0].category_name
        hands.append((landmarks_to_array(hand), label))
    return hands


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------

def normalize_landmarks(landmarks_xyz):
    """
    Make one frame's landmarks translation- and scale-invariant, so the
    classifier learns hand SHAPE, not where the hand is or how close it is.

    landmarks_xyz: (21, 3) array. Returns a flat 63-d vector.
    """
    coords = np.asarray(landmarks_xyz, dtype=np.float32).copy()
    coords -= coords[WRIST]                      # wrist -> origin
    scale = np.linalg.norm(coords[MIDDLE_MCP])   # wrist -> middle knuckle
    if scale > 1e-6:
        coords /= scale
    return coords.flatten()


def resample_sequence(seq, target_len=TRAJ_LEN):
    """
    Resample a variable-length (n, d) sequence to (target_len, d) by linear
    interpolation along time. Lets gestures of different durations line up.
    """
    seq = np.asarray(seq, dtype=np.float32)
    n = len(seq)
    if n == target_len:
        return seq
    old_t = np.linspace(0.0, 1.0, n)
    new_t = np.linspace(0.0, 1.0, target_len)
    return np.stack(
        [np.interp(new_t, old_t, seq[:, i]) for i in range(seq.shape[1])],
        axis=1,
    )


def motion_features(landmark_seq, traj_len=TRAJ_LEN):
    """
    Turn a whole gesture (a list of per-frame (21,3) arrays) into one feature
    vector describing BOTH the drawn shape and the handshape:

      - trajectory (32-d): the wrist's (x, y) path, made relative to its
        start point and scaled by its largest excursion, then resampled to
        `traj_len` points. This is the SHAPE you draw in the air -- a hook
        for J, a zigzag for Z -- independent of where on screen you draw it
        or how big/fast you draw it.
      - handshape (63-d): normalized landmarks at the gesture's midpoint.
        Distinguishes J's pinky-out shape from Z's index-out shape and
        rejects random hand waving.

    Returns a 95-d vector (32 + 63).
    """
    seq = [np.asarray(f, dtype=np.float32) for f in landmark_seq]

    wrist_xy = np.array([f[WRIST, :2] for f in seq], dtype=np.float32)  # (n, 2)
    wrist_xy = wrist_xy - wrist_xy[0]                                    # start at origin
    extent = float(np.max(np.linalg.norm(wrist_xy, axis=1)))
    if extent > 1e-6:
        wrist_xy = wrist_xy / extent                                    # scale-invariant
    traj = resample_sequence(wrist_xy, traj_len).flatten()              # 32-d

    shape = normalize_landmarks(seq[len(seq) // 2])                      # 63-d
    return np.concatenate([traj, shape]).astype(np.float32)


# --------------------------------------------------------------------------
# Runtime gesture segmentation (static vs motion)
# --------------------------------------------------------------------------

class GestureSegmenter:
    """
    Watches the wrist frame by frame and separates "holding a static letter"
    from "drawing a motion letter".

    Feed each frame's raw (21, 3) landmarks (or None when no hand is seen) to
    update(). It stays IDLE until the wrist speeds up past `move_on`, then
    buffers frames while MOVING until the wrist settles below `move_off` for
    `still_frames_to_end` frames. When a gesture ends, update() returns the
    buffered frames -- but only if the gesture was long enough and travelled
    far enough (`min_frames`, `min_path`) to be a real drawn letter rather
    than a jitter. Otherwise it returns None.

    Hysteresis (move_on > move_off) stops it flickering between states.
    All thresholds are in MediaPipe normalized image units (0..1) per frame,
    so they're resolution-independent but framerate-sensitive -- tune them in
    recognize.py if your camera runs much faster or slower than ~20-30 fps.
    """

    def __init__(self, move_on=0.018, move_off=0.007, still_frames_to_end=4,
                 min_frames=6, min_path=0.12):
        self.move_on = move_on
        self.move_off = move_off
        self.still_frames_to_end = still_frames_to_end
        self.min_frames = min_frames
        self.min_path = min_path
        self._reset()

    def _reset(self):
        self.state = "IDLE"
        self.buffer = []
        self.prev_wrist = None
        self.slow_count = 0
        self.path_len = 0.0

    @property
    def is_moving(self):
        return self.state == "MOVING"

    def update(self, landmarks_xyz):
        """Process one frame. Returns a completed gesture (list of frames)
        the moment one ends, otherwise None."""
        if landmarks_xyz is None:
            # Hand lost: end any gesture in progress, then reset tracking.
            finished = self._finish()
            self.prev_wrist = None
            return finished

        wrist = np.asarray(landmarks_xyz, dtype=np.float32)[WRIST, :2]
        speed = 0.0 if self.prev_wrist is None else float(np.linalg.norm(wrist - self.prev_wrist))
        self.prev_wrist = wrist

        if self.state == "IDLE":
            if speed > self.move_on:
                self.state = "MOVING"
                self.buffer = [np.asarray(landmarks_xyz, dtype=np.float32)]
                self.slow_count = 0
                self.path_len = 0.0
            return None

        # MOVING
        self.buffer.append(np.asarray(landmarks_xyz, dtype=np.float32))
        self.path_len += speed
        if speed < self.move_off:
            self.slow_count += 1
            if self.slow_count >= self.still_frames_to_end:
                return self._finish()
        else:
            self.slow_count = 0
        return None

    def _finish(self):
        gesture = None
        if (self.state == "MOVING"
                and len(self.buffer) >= self.min_frames
                and self.path_len >= self.min_path):
            gesture = self.buffer
        # Reset for the next gesture (keep prev_wrist; caller resets on hand loss).
        self.state = "IDLE"
        self.buffer = []
        self.slow_count = 0
        self.path_len = 0.0
        return gesture

# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

# The 21-landmark hand skeleton: which landmark indices are connected by a
# "bone". Purely for on-screen display — recognition uses only the points.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                 # palm base
)


def draw_hand(frame, hand_landmarks,
              point_color=(0, 255, 0), line_color=(255, 255, 255)):
    """Draw the full MediaPipe-style hand skeleton (bones + joints) onto a
    BGR frame in place. Display-only; has no effect on recognition."""
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], line_color, 2)
    for p in pts:
        cv2.circle(frame, p, 4, point_color, -1)