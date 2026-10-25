"""
The camera gesture source — the seam that joins the two projects.

This is the old Sign_ASL `recognize.py` runtime loop, restructured as a
*gesture source*: an iterable yielding ``(symbol, confidence)`` tuples, the
exact duck-typed contract `signspeak.pipeline` was designed around (see
sources.py). The pipeline neither knows nor cares that predictions now come
from a webcam and the trainer's models instead of `fake_source()`.

Per frame:
  hand still   -> static model votes over a window; a letter is committed
                  once (debounced by StaticCommitter) and yielded lowercase.
  hand drawing -> GestureSegmenter buffers frames; when the motion settles
                  the motion model classifies J/Z and it is yielded.

Control is SIGNED, not typed. Two extra static classes — SPACE and SPEAK
(collect them in collect_static.py; suggested handshapes: open "5" hand and
thumbs-up) — commit like letters but require a stricter, unanimous vote
window (config.CONTROL_MIN_VOTES) because they trigger actions:

  SPACE sign -> yields ("space", conf)   word break
  SPEAK sign -> yields ("speak", conf)   flush buffer -> sentence -> speech

Fallback: if the loaded static model was trained WITHOUT control-sign
classes, the keyboard takes over (SPACE key / ENTER key) so older models
keep working — with a printed hint to collect the control signs.

Word mode (only when models/word_model.joblib exists): detection runs with
two hands. One hand in view is letter mode, exactly as above. Bring the
other hand into view and, after WORD_MODE_ENTER_FRAMES, the loop switches
to word mode: the static and J/Z paths pause, the WordSegmenter cuts each
two-hand sign out of the stream, and the word model classifies it.

  word sign  -> yields ("doctor", conf)  a whole word for the buffer

Drop the second hand to return to letters. Nothing half-finished carries
across a mode switch. Without a word model the landmarker is created with
one hand and none of the word machinery runs.

One-hand signs (only when models/onehand_model.joblib exists): in letter
mode, movements are cut by a WordSegmenter on the fingerspelling hand and
classified by the one-hand model as a one-hand word, J, Z or NONE. This
replaces the GestureSegmenter + motion model J/Z path, which runs exactly
as before whenever there is no one-hand model.

Q or ESC always quits (that's a window control, not a language control).
"""

import time
from typing import Iterator

import cv2

from . import config, labels
from .commit import StaticCommitter
from .models import SignModels
from .sources import Gesture
from .tracking import (
    draw_hand,
    GestureSegmenter,
    bgr_frame_to_mp_image,
    create_landmarker,
    hands_with_labels,
    landmarks_to_array,
    motion_features,
    normalize_landmarks,
)
from .words import (HandPairer, ModeArbiter, WordSegmenter, onehand_features,
                    word_features)

_KEY_SPACE = 32
_KEY_ENTER = 13
_KEY_ESC = 27


def _has_control_signs(models: SignModels) -> bool:
    """True if the static model was trained with the control-sign classes."""
    classes = set(str(c) for c in getattr(models.static, "classes_", []))
    return labels.SPACE_SIGN in classes and labels.SPEAK_SIGN in classes


def camera_source(models: SignModels = None,
                  camera_index: int = None,
                  window_title: str = "SignSpeak",
                  ) -> Iterator[Gesture]:
    """Yield (symbol, confidence) gestures from live webcam recognition."""
    models = models or SignModels.load()
    camera_index = config.CAMERA_INDEX if camera_index is None else camera_index

    sign_control = _has_control_signs(models)
    if sign_control:
        hint = "SPACE sign=word  SPEAK sign=say it  Q=quit"
        print("Camera source running with signed controls "
              "(SPACE/SPEAK handshapes). Q/ESC = quit.")
    else:
        hint = "SPACE=word  ENTER=speak  Q=quit"
        print("Note: this static model has no trained control signs, so the "
              "keyboard controls word breaks and speaking (SPACE / ENTER).")
        print("Collect SPACE and SPEAK signs in collect_static.py and retrain "
              "to control everything by signing.")

    use_words = models.word is not None
    word_hint = "WORD MODE: sign with both hands   drop one hand: letters   Q=quit"
    if use_words:
        print("Word model loaded: bring your second hand into view for word "
              "signs, drop it to fingerspell.")
    use_onehand = models.onehand is not None
    if use_onehand:
        print("One-hand model loaded: one-hand words, J and Z are signed with "
              "one hand (motion_model.joblib is not used).")

    landmarker = create_landmarker(num_hands=2 if use_words else 1)
    segmenter = GestureSegmenter(**config.SEGMENTER)
    committer = StaticCommitter(
        control_labels={labels.SPACE_SIGN, labels.SPEAK_SIGN})

    pairer = HandPairer(config.DOMINANT_HAND)
    arbiter = ModeArbiter()
    word_segmenter = WordSegmenter(**config.WORD_SEGMENTER)
    onehand_segmenter = WordSegmenter(**config.ONEHAND_SEGMENTER)
    word_display, word_hold = "", 0
    last_word_at = float("-inf")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam (index {camera_index}). "
            f"Try a different config.CAMERA_INDEX."
        )

    motion_letter = ""
    motion_hold = 0
    committed_tail = ""          # last few committed tokens, for the overlay
    start = time.time()

    def register(token: str) -> str:
        """Update the on-screen tail for a committed token."""
        nonlocal committed_tail
        if token == labels.SPEAK:
            committed_tail = ""
        elif token == labels.SPACE:
            committed_tail = (committed_tail + " ")[-24:]
        elif labels.is_word_token(token):
            sep = "" if not committed_tail or committed_tail.endswith(" ") else " "
            committed_tail = (committed_tail + sep + token.upper() + " ")[-24:]
        else:
            committed_tail = (committed_tail + token)[-24:]
        return token

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror -> feels natural

            result = landmarker.detect_for_video(
                bgr_frame_to_mp_image(frame), int((time.time() - start) * 1000)
            )
            # ---- which hands, which mode ---------------------------------
            if use_words:
                for hand_lms in result.hand_landmarks:
                    draw_hand(frame, hand_lms)
                hands = hands_with_labels(result)
                dom, nondom = pairer.pair(hands)
                if arbiter.update(len(hands)):
                    # Mode switched: nothing half-finished may carry over.
                    segmenter = GestureSegmenter(**config.SEGMENTER)
                    committer.hand_lost()
                    word_segmenter.reset()
                    onehand_segmenter.reset()
                    motion_hold = word_hold = 0
                word_mode = arbiter.is_word
                # Letter mode reads the fingerspelling hand: the only one in
                # view, or the dominant one while a second is coming or going.
                raw = dom if dom is not None else nondom
            else:
                word_mode = False
                hand = result.hand_landmarks[0] if result.hand_landmarks else None
                raw = landmarks_to_array(hand) if hand is not None else None
                if hand is not None:
                    draw_hand(frame, hand)

            if word_mode:
                # ---- whole-word signs (two hands) ----------------------
                sign = word_segmenter.update(dom, nondom)
                if sign is not None:
                    feats = word_features(sign)
                    prediction = models.predict_word(feats) if feats is not None else None
                    if prediction is not None:
                        word, confidence = prediction
                        token = labels.to_token(word)
                        now = time.time()
                        if (token != labels.NEUTRAL   # NONE class = not a sign
                                and confidence >= config.WORD_CONFIDENCE_THRESHOLD
                                and now - last_word_at >= config.WORD_REFRACTORY_S):
                            last_word_at = now
                            word_display, word_hold = word, config.WORD_HOLD_FRAMES
                            yield register(token), confidence
                if word_hold > 0:
                    display = word_display
                    word_hold -= 1
                elif word_segmenter.is_moving:
                    display = "..."             # a sign is being made
                elif word_segmenter.is_ready:
                    display = ""
                else:
                    display = "hold"            # settling before the next sign
            else:
                if raw is None:
                    committer.hand_lost()

                if use_onehand:
                    # ---- one-hand signs: words, J and Z ----------------
                    sign = onehand_segmenter.update(raw, None)
                    if sign is not None:
                        committer.interrupt()   # settling pose must not leak into static
                        feats = onehand_features(sign)
                        prediction = (models.predict_onehand(feats)
                                      if feats is not None else None)
                        if prediction is not None:
                            label, confidence = prediction
                            token = labels.to_token(label)
                            gate = (config.MOTION_CONFIDENCE_THRESHOLD
                                    if label in labels.MOTION_LETTERS
                                    else config.ONEHAND_WORD_CONFIDENCE_THRESHOLD)
                            if token != labels.NEUTRAL and confidence >= gate:
                                motion_letter, motion_hold = label, config.MOTION_HOLD_FRAMES
                                yield register(token), confidence
                    moving = onehand_segmenter.is_moving
                else:
                    # ---- motion letters (J / Z) ------------------------
                    gesture = segmenter.update(raw)
                    if gesture is not None:
                        prediction = models.predict_motion(motion_features(gesture))
                        if prediction is not None:
                            letter, confidence = prediction
                            committer.interrupt()   # settling pose must not leak into static
                            token = labels.to_token(letter)
                            if (token != labels.NEUTRAL   # NONE class = transition, ignore
                                    and confidence >= config.MOTION_CONFIDENCE_THRESHOLD):
                                motion_letter, motion_hold = letter, config.MOTION_HOLD_FRAMES
                                yield register(token), confidence
                    moving = segmenter.is_moving

                # ---- static letters and control signs ------------------
                if motion_hold > 0:
                    display = motion_letter
                    motion_hold -= 1
                elif moving:
                    display = "..."             # a gesture is being drawn
                elif raw is not None:
                    label, confidence = models.predict_static(normalize_landmarks(raw))
                    commit = committer.update(label, confidence)
                    if commit is not None:
                        committed_label, mean_conf = commit
                        gate = (config.CONTROL_CONFIDENCE_THRESHOLD
                                if committed_label in labels.CONTROL_SIGN_LABELS
                                else config.STATIC_CONFIDENCE_THRESHOLD)
                        if mean_conf >= gate:
                            yield register(labels.to_token(committed_label)), mean_conf
                    display = committer.current_winner
                else:
                    display = "-"

            # ---- overlay ------------------------------------------------
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 105), (0, 0, 0), -1)
            size = 2.0 if len(display) <= 2 else (1.2 if len(display) <= 6 else 0.9)
            cv2.putText(frame, display or "-", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, size, (0, 255, 255), 3)
            cv2.putText(frame, committed_tail, (200, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, word_hint if word_mode else hint, (200, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
            if use_words:
                cv2.putText(frame, "WORD" if word_mode else "LETTER", (20, 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (0, 165, 255) if word_mode else (160, 160, 160), 2)

            cv2.imshow(window_title, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (_KEY_ESC, ord("q")):
                break
            if not sign_control:            # keyboard fallback only
                if key == _KEY_SPACE:
                    yield register(labels.SPACE), 1.0
                elif key == _KEY_ENTER:
                    yield register(labels.SPEAK), 1.0
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()