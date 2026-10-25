"""
Collect MOTION letter samples (J and Z) as short drawn gestures — plus the
NONE class: press SPACE to arm a take, then move your hand the way you
naturally do BETWEEN letters (reposition, drift, settle, half-form shapes).
30-40 varied NONE takes teach the motion model to answer "not a gesture"
instead of force-guessing J or Z on every transition; without them, live
recognition sprays spurious J/Z tokens.

Unlike static letters, J and Z are drawn in the air, so each sample is a
short sequence of frames rather than a single pose:

  - J: pinky extended (the "I" handshape), trace a hook (down and curl).
  - Z: index extended, trace a "Z" zigzag in the air.

Press J or Z (or SPACE for a NONE transition take) to start a ~2 second recording, perform the motion within the
window, and it saves automatically. Draw with your whole hand moving (not
just a tiny finger flick) so the trajectory is clear. Aim for 25-40+ samples
each, varying size, speed, and screen position.

Controls:
  J / Z     -> record one gesture sample of that letter (~2s window)
  BACKSPACE -> undo the last saved sample
  Q / ESC   -> save to CSV and quit
"""

import csv
import os
import time

import cv2

from . import config
from .labels import MOTION_LETTERS, NONE_SIGN
from .tracking import (
    draw_hand,
    GestureSegmenter,  # noqa: F401  (imported so collection matches runtime intent)
    bgr_frame_to_mp_image,
    create_landmarker,
    landmarks_to_array,
    motion_features,
)

OUTPUT_CSV = str(config.MOTION_DATASET_CSV)
FEATURE_COLS = [f"m{i}" for i in range(config.MOTION_FEATURES)]

CAPTURE_SECONDS = config.CAPTURE_SECONDS   # length of each recording window
MIN_HAND_FRAMES = config.MIN_HAND_FRAMES   # discard takes with fewer hand frames


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
    counts = {c: sum(1 for r in rows if r[0] == c) for c in list(MOTION_LETTERS) + [NONE_SIGN]}
    start = time.time()

    recording_label = None
    record_until = 0.0
    gesture_frames = []

    print("Motion collection. Press J or Z to record a ~2s gesture.")
    print("BACKSPACE undoes the last sample. Q/ESC saves and quits.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            now = time.time()
            result = landmarker.detect_for_video(
                bgr_frame_to_mp_image(frame), int((now - start) * 1000)
            )

            hand = result.hand_landmarks[0] if result.hand_landmarks else None
            if hand is not None:
                draw_hand(frame, hand)

            if recording_label is not None:
                # Collecting frames for the current take.
                if hand is not None:
                    gesture_frames.append(landmarks_to_array(hand))
                remaining = record_until - now
                cv2.putText(frame, f"RECORDING {recording_label}  {remaining:0.1f}s",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                # Progress bar
                frac = max(0.0, min(1.0, 1.0 - remaining / CAPTURE_SECONDS))
                cv2.rectangle(frame, (10, 45), (10 + int(300 * frac), 60), (0, 165, 255), -1)

                if now >= record_until:
                    if len(gesture_frames) >= MIN_HAND_FRAMES:
                        feats = motion_features(gesture_frames)
                        rows.append([recording_label] + feats.tolist())
                        counts[recording_label] += 1
                        print(f"Saved '{recording_label}' "
                              f"({len(gesture_frames)} frames, total: {counts[recording_label]})")
                    else:
                        print(f"Discarded '{recording_label}' take - "
                              f"only {len(gesture_frames)} frames with a hand detected.")
                    recording_label = None
                    gesture_frames = []
            else:
                cv2.putText(frame, "Press J or Z to record a gesture", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.putText(frame, " ".join(f"{k}:{v}" for k, v in counts.items()),
                        (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)

            cv2.imshow("Motion data collection (J / Z)", frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                break
            elif key == 8 and recording_label is None and rows:  # backspace
                removed = rows.pop()
                counts[removed[0]] -= 1
                print(f"Undid last sample ('{removed[0]}')")
            elif recording_label is None and key == 32:   # SPACE -> NONE take
                recording_label = NONE_SIGN
                gesture_frames = []
                record_until = now + CAPTURE_SECONDS
                print("Recording 'NONE' ... move like a between-letters transition now.")
            elif recording_label is None and 0 <= key < 256 and chr(key).upper() in MOTION_LETTERS:
                recording_label = chr(key).upper()
                gesture_frames = []
                record_until = now + CAPTURE_SECONDS
                print(f"Recording '{recording_label}' ... perform the motion now.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        save_rows(rows)
        print(f"Saved {len(rows)} motion samples to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()