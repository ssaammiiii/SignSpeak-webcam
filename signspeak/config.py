"""
Central configuration for the unified SignSpeak system.

Every tunable parameter lives here — recognition, training, and translation —
so nothing is hardcoded inside pipeline logic or scripts.

Paths are overridable via environment variables, so the recognition engine
always picks up whatever the trainer last produced without code edits:

    SIGNSPEAK_MODEL_DIR   where trained models are saved / loaded (default: models/)
    SIGNSPEAK_DATA_DIR    where collected datasets are saved / read (default: data/)
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (training output == recognition input, by construction)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = Path(os.environ.get("SIGNSPEAK_MODEL_DIR", PROJECT_ROOT / "models"))
DATA_DIR = Path(os.environ.get("SIGNSPEAK_DATA_DIR", PROJECT_ROOT / "data"))

STATIC_MODEL_PATH = MODEL_DIR / "static_model.joblib"
MOTION_MODEL_PATH = MODEL_DIR / "motion_model.joblib"
LANDMARKER_PATH = MODEL_DIR / "hand_landmarker.task"   # MediaPipe bundle (auto-downloaded)

STATIC_DATASET_CSV = DATA_DIR / "static_dataset.csv"
MOTION_DATASET_CSV = DATA_DIR / "motion_dataset.csv"

# Two-hand word model. word_raw/ keeps every take as a (T, 2, 21, 3) landmark
# array, one folder per signer, so features can be recomputed later with
# `collect_words.py --rebuild` instead of re-recording.
WORD_MODEL_PATH = MODEL_DIR / "word_model.joblib"
WORD_DATASET_CSV = DATA_DIR / "word_dataset.csv"
WORD_RAW_DIR = DATA_DIR / "word_raw"

# One-hand model: one-hand words plus J and Z, cut by the same kind of
# segmenter as the two-hand words. When onehand_model.joblib exists it takes
# over J and Z from motion_model.joblib; until then nothing changes.
ONEHAND_MODEL_PATH = MODEL_DIR / "onehand_model.joblib"
ONEHAND_DATASET_CSV = DATA_DIR / "onehand_dataset.csv"
ONEHAND_RAW_DIR = DATA_DIR / "onehand_raw"

# ---------------------------------------------------------------------------
# Recognition (camera source)
# ---------------------------------------------------------------------------
CAMERA_INDEX: int = 0

# Static-letter smoothing: majority vote over a sliding window of frames.
STATIC_SMOOTH_WINDOW: int = 8   # frames considered
STATIC_MIN_VOTES: int = 5       # agreeing frames required to commit a letter
# Control signs trigger actions (word break / speak the sentence), so they
# need a stricter bar: this many agreeing frames (default: the full window,
# i.e. unanimous) before a control sign commits.
CONTROL_MIN_VOTES: int = 8
MOTION_HOLD_FRAMES: int = 20    # how long a recognized J/Z stays on screen

# GestureSegmenter thresholds (see signspeak/tracking.py for meaning).
# Normalized image units per frame — framerate-sensitive; tune if your camera
# runs much faster or slower than ~20-30 fps.
SEGMENTER = dict(
    move_on=0.018,
    move_off=0.007,
    still_frames_to_end=4,
    min_frames=6,
    min_path=0.12,
)

# ---------------------------------------------------------------------------
# Word mode (two-hand signs)
# ---------------------------------------------------------------------------
# Only active when models/word_model.joblib exists. Without it the camera
# runs one-hand detection exactly as before.

# Which hand fingerspells. In word mode it fills the "dominant" slot.
DOMINANT_HAND: str = "Right"

# Mode switch: two hands in view for this many consecutive frames enters word
# mode; fewer than two for this many leaves it. The exit is slower so a hand
# blurring out of detection mid-sign does not drop you back to letters.
WORD_MODE_ENTER_FRAMES: int = 6     # ~0.2 s at 30 fps
WORD_MODE_EXIT_FRAMES: int = 12     # ~0.4 s at 30 fps

# WordSegmenter (signspeak/words.py). Speeds are wrist movement in normalized
# image units per frame, like SEGMENTER above, using the faster of the two
# hands. Slightly more permissive than J/Z because some word signs are slow,
# single strokes; the NONE class and the settle-first rule absorb the extra
# triggers. still_frames_to_end is longer than for J/Z so the short pause at
# the turning point of a repeated tap does not split one sign into two.
# Tuning: if "..." never appears while you sign, lower move_on. If it
# appears while you are only resting your hands, raise it.
WORD_SEGMENTER = dict(
    move_on=0.014,
    move_off=0.006,
    still_frames_to_end=8,
    min_frames=10,
    min_path=0.12,
    max_frames=150,     # hard cap (~5 s) so continuous motion cannot run on
    preroll=4,          # frames kept from just before motion onset
)

# Word feature geometry (must match between collection, training, inference).
WORD_TRAJ_LEN: int = 16
WORD_FEATURES: int = 4 * WORD_TRAJ_LEN + WORD_TRAJ_LEN + 2 * 63 + 2 * 21 + 8  # 256
WORD_MAX_MISSING: float = 0.35      # drop a take if a hand is unseen this often
WORD_REVERSAL_MIN_STEP: float = 0.05  # hand lengths/frame; below this is jitter

# One-hand segmenter (letter mode, used once a one-hand model exists). Same
# design as WORD_SEGMENTER, on the fingerspelling hand alone. move_on matches
# the J/Z segmenter so ordinary fingerspelling transitions trigger it no more
# often than they do now; still_frames_to_end is shorter than for two-hand
# words so letters resume quickly after a sign.
ONEHAND_SEGMENTER = dict(
    move_on=0.018,
    move_off=0.007,
    still_frames_to_end=6,
    min_frames=8,
    min_path=0.12,
    max_frames=120,     # ~4 s
    preroll=4,
)
ONEHAND_FEATURES: int = 2 * WORD_TRAJ_LEN + WORD_TRAJ_LEN + 3 * 63 + 21 + 4  # 262

# Collection: after arming a take, how long to wait for the sign to start.
WORD_ARM_TIMEOUT: float = 5.0

# Runtime gating for word tokens.
WORD_CONFIDENCE_THRESHOLD: float = 0.55
# One-hand model outputs: words, and J/Z (which keep their own, lower gate,
# MOTION_CONFIDENCE_THRESHOLD, as before).
ONEHAND_WORD_CONFIDENCE_THRESHOLD: float = 0.55
WORD_REFRACTORY_S: float = 0.8      # ignore a second word this soon after one
WORD_HOLD_FRAMES: int = 30          # how long a recognized word stays on screen

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
MIN_SAMPLES_PER_CLASS: int = 15   # warn below this
RF_ESTIMATORS: int = 200
RANDOM_STATE: int = 42

# Motion feature geometry (must match between collection, training, inference).
TRAJ_LEN: int = 16                 # resampled trajectory points -> 32-d
STATIC_FEATURES: int = 63          # 21 landmarks x 3
MOTION_FEATURES: int = TRAJ_LEN * 2 + STATIC_FEATURES  # 95-d

# Motion collection
CAPTURE_SECONDS: float = 2.0
MIN_HAND_FRAMES: int = 6

# ---------------------------------------------------------------------------
# Translation pipeline
# ---------------------------------------------------------------------------
# Per-stream confidence gates, applied in camera_source where we know which
# classifier produced each token. Different model types speak different
# confidence "dialects" (an ensemble's averaged probabilities and a tree
# model's vote fractions run lower than a lone MLP's softmax), so each
# stream gets its own gate, calibrated from recognize.py logs:
STATIC_CONFIDENCE_THRESHOLD: float = 0.60   # letters from the static model
MOTION_CONFIDENCE_THRESHOLD: float = 0.45   # J/Z from the motion model
CONTROL_CONFIDENCE_THRESHOLD: float = 0.60  # SPACE/SPEAK signs (never fire weakly)

# Pipeline-level gate. The camera source already gates per stream, so this
# is 0 for live use; it still applies to other sources (e.g. fake_source).
CONFIDENCE_THRESHOLD: float = 0.0

# LLM used to turn a raw letter stream into a grammatical sentence.
LLM_MODEL: str = "gemini-3.5-flash"
LLM_MAX_TOKENS: int = 200