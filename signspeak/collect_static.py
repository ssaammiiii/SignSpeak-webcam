"""
Collect STATIC letter samples (single-frame handshapes).

Hold a letter's handshape and press its key to save the current landmarks.
Aim for 30-50+ samples per letter, varying angle/distance/position between
samples so the model generalizes. J and Z are motion letters -- collect
those with collect_motion.py instead.

Also collects the two CONTROL SIGNS — handshapes that trigger pipeline
actions at runtime. Pick shapes distinct from every letter (suggested:
open "5" hand with fingers spread for SPACE, thumbs-up for SPEAK) and
collect them exactly like letters: 30-50 varied samples each.

Controls:
  A-Y (no J)  -> save one sample of that letter's handshape
  SPACE key   -> save one sample of your SPACE control sign
  ENTER key   -> save one sample of your SPEAK control sign
  BACKSPACE   -> undo the last saved sample
  Q / ESC     -> save to CSV and quit
"""

import csv
import os
import time

import cv2

from . import config
from .labels import SPACE_SIGN, SPEAK_SIGN, STATIC_LETTERS
from .tracking import (
    draw_hand,
    bgr_frame_to_mp_image,
    create_landmarker,
    landmarks_to_array,
    normalize_landmarks,
)

OUTPUT_CSV = str(config.STATIC_DATASET_CSV)
FEATURE_COLS = [f"f{i}" for i in range(config.STATIC_FEATURES)]

# Everything collectable here: letters + the two control-sign classes.
STATIC_CLASSES = list(STATIC_LETTERS) + [SPACE_SIGN, SPEAK_SIGN]
_KEY_TO_CONTROL = {32: SPACE_SIGN, 13: SPEAK_SIGN}   # SPACE / ENTER keys


def load_existing_rows():
    if not os.path.exists(OUTPUT_CSV):
        return []
    with open(OUTPUT_CSV, newline="") as f:
        rows = list(csv.reader(f))
    return rows[1:] if rows else []


def save_rows(rows):
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label"] + FEATURE_COLS)
        writer.writerows(rows)


def main():
    landmarker = create_landmarker(num_hands=1)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0). Try a different camera index.")

    rows = load_existing_rows()
    counts = {c: sum(1 for r in rows if r[0] == c) for c in STATIC_CLASSES}
    start = time.time()

    print("Static collection. Press a letter key (A-Y, no J) to save that handshape.")
    print("SPACE key saves a SPACE-sign sample; ENTER saves a SPEAK-sign sample.")
    print("BACKSPACE undoes the last sample. Q/ESC saves and quits.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror -> feels natural facing the camera

            result = landmarker.detect_for_video(
                bgr_frame_to_mp_image(frame), int((time.time() - start) * 1000)
            )

            features = None
            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                features = normalize_landmarks(landmarks_to_array(hand))
                draw_hand(frame, hand)
                cv2.putText(frame, "Hand detected - press a letter key", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No hand detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.putText(frame, " ".join(f"{k}:{v}" for k, v in counts.items()),
                        (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (255, 255, 255), 1)

            cv2.imshow("Static data collection", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 8 and rows:  # backspace
                removed = rows.pop()
                counts[removed[0]] -= 1
                print(f"Undid last sample ('{removed[0]}')")
            elif key in _KEY_TO_CONTROL or (0 <= key < 256 and chr(key).upper() in STATIC_LETTERS):
                label = _KEY_TO_CONTROL.get(key) or chr(key).upper()
                if features is not None:
                    rows.append([label] + features.tolist())
                    counts[label] += 1
                    print(f"Saved '{label}' (total: {counts[label]})")
                else:
                    print("No hand detected - sample not saved.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        save_rows(rows)
        print(f"Saved {len(rows)} static samples to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()