"""
Configurable loading of the trained classifiers.

The trainer writes to config.MODEL_DIR and the recognition engine reads from
the same place, so retraining automatically updates what recognition uses —
no paths to edit. Override the directory with SIGNSPEAK_MODEL_DIR if needed.

Both classifiers are scikit-learn RandomForests, so `predict_proba` gives the
per-class probabilities we surface as the pipeline's confidence value.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np

from . import config


@dataclass
class SignModels:
    """The classifiers produced by training."""
    static: object                       # required: 24 still letters
    motion: Optional[object] = None      # optional: J and Z
    word: Optional[object] = None        # optional: two-hand word signs
    onehand: Optional[object] = None     # optional: one-hand words + J/Z

    @classmethod
    def load(cls,
             static_path: Path = None,
             motion_path: Path = None,
             word_path: Path = None,
             onehand_path: Path = None) -> "SignModels":
        static_path = Path(static_path or config.STATIC_MODEL_PATH)
        motion_path = Path(motion_path or config.MOTION_MODEL_PATH)
        word_path = Path(word_path or config.WORD_MODEL_PATH)
        onehand_path = Path(onehand_path or config.ONEHAND_MODEL_PATH)

        if not static_path.exists():
            raise FileNotFoundError(
                f"No static model at {static_path}. Run training first "
                f"(collect data, then `python train.py`), or point "
                f"SIGNSPEAK_MODEL_DIR at a folder containing trained models."
            )
        static = joblib.load(static_path)

        motion = None
        if motion_path.exists():
            motion = joblib.load(motion_path)
        else:
            print(f"Note: no motion model at {motion_path} - J and Z won't be "
                  f"recognized. Collect motion data and retrain to enable them.")

        word = None
        if word_path.exists():
            word = joblib.load(word_path)
        # No message when absent: word mode is an optional extension, and
        # letter-only use is a normal configuration.

        onehand = None
        if onehand_path.exists():
            onehand = joblib.load(onehand_path)
            classes = {str(c) for c in getattr(onehand, "classes_", [])}
            missing = [m for m in "JZ" if m not in classes]
            if missing:
                print(f"Note: the one-hand model has no {'/'.join(missing)} class. "
                      f"It replaces the J/Z model, so collect J and Z with "
                      f"`collect_words.py --one-hand` and retrain.")
        return cls(static=static, motion=motion, word=word, onehand=onehand)

    # ------------------------------------------------------------------
    # Prediction helpers: (label, confidence) instead of bare labels, so
    # the translation pipeline's confidence gate has something to gate on.
    # ------------------------------------------------------------------

    @staticmethod
    def _predict_with_confidence(clf, features: np.ndarray) -> Tuple[str, float]:
        probs = clf.predict_proba(features.reshape(1, -1))[0]
        idx = int(np.argmax(probs))
        return str(clf.classes_[idx]), float(probs[idx])

    def predict_static(self, features: np.ndarray) -> Tuple[str, float]:
        """features: 63-d normalized landmark vector -> (letter, confidence)."""
        return self._predict_with_confidence(self.static, features)

    def predict_motion(self, features: np.ndarray) -> Optional[Tuple[str, float]]:
        """features: 95-d motion vector -> (letter, confidence), or None if
        no motion model is available."""
        if self.motion is None:
            return None
        return self._predict_with_confidence(self.motion, features)

    def predict_word(self, features: np.ndarray) -> Optional[Tuple[str, float]]:
        """features: 256-d word vector -> (word label, confidence), or None
        if no word model is available."""
        if self.word is None:
            return None
        return self._predict_with_confidence(self.word, features)

    def predict_onehand(self, features: np.ndarray) -> Optional[Tuple[str, float]]:
        """features: 262-d one-hand vector -> (label, confidence), where the
        label is a one-hand word, J, Z or NONE; None if no one-hand model."""
        if self.onehand is None:
            return None
        return self._predict_with_confidence(self.onehand, features)
