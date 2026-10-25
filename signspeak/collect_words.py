"""
Collect word sign samples, one sign per take.

Two kinds of take, kept in separate datasets and models:

  two-hand words   DIZZY, DOCTOR, GOOD, SICK, WELL      (default)
  one-hand signs   FEVER, FINE, HELP, HOSPITAL, NOT, J, Z  (--one-hand)

Every take is cut out of the camera stream by the same WordSegmenter the
live system uses, so the model is trained on exactly the kind of segment it
will see at runtime. A take is kept only if it ends with your hands coming
to rest; one that runs on until the length cap, or loses the hand, is
discarded with a message, because it would not look like a live sign.

    uv run collect_words.py                    # two-hand words
    uv run collect_words.py --one-hand         # one-hand words, J and Z
    uv run collect_words.py --signer ishaan    # optional: record who signed
    uv run collect_words.py --rebuild          # recompute a CSV from raw takes
    uv run collect_words.py --one-hand --rebuild

How a take works:
  1. press the sign's key (legend on screen)
  2. hold still until the screen says GO (both hands in view for two-hand
     words; only your fingerspelling hand for one-hand signs)
  3. make the sign once, then hold still: it saves itself

Controls:
  1..9      arm a take for the word shown next to that key
  J / Z     (one-hand mode) arm a take for the letter J or Z
  SPACE     arm a NONE take: move the way you do BETWEEN signs. Two-hand
            mode: raise and lower a hand, reposition, start a sign and stop.
            One-hand mode: the transitions you make while fingerspelling.
  BACKSPACE undo the last take saved this session
  Q / ESC   quit (everything is already saved after each take)

Aim for 30-40 takes of every class, varying speed, size and where you sit.
Every take is also kept as a raw landmark array (data/word_raw/ or
data/onehand_raw/, one folder per signer), so if the features ever change,
--rebuild regenerates the CSV without re-recording anything.
"""

import argparse
import csv
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np

from . import config, labels
from .words import (
    HandPairer, WordSegmenter,
    onehand_features_from_array, stack_onehand,
    stack_sequence, word_features_from_array,
)

_KEY_SPACE, _KEY_BACKSPACE, _KEY_ESC = 32, 8, 27
_DIGIT_KEYS = "123456789"


@dataclass(frozen=True)
class Mode:
    name: str
    csv: Path
    raw_dir: Path
    n_features: int
    prefix: str
    classes: Tuple[str, ...]
    keys: Dict[str, str]
    segmenter: dict
    hands: int
    stack: Callable
    features: Callable

    @property
    def header(self):
        return ["label", "signer"] + [f"{self.prefix}{i}" for i in range(self.n_features)]


def _word_keys(words):
    if len(words) > len(_DIGIT_KEYS):
        raise SystemExit(f"Too many words for keys 1-9 ({len(words)}).")
    return dict(zip(_DIGIT_KEYS, words))


TWO_HAND = Mode(
    name="two-hand word",
    csv=Path(config.WORD_DATASET_CSV),
    raw_dir=Path(config.WORD_RAW_DIR),
    n_features=config.WORD_FEATURES,
    prefix="w",
    classes=labels.WORD_CLASSES,
    keys=_word_keys(labels.TWO_HAND_WORDS),
    segmenter=config.WORD_SEGMENTER,
    hands=2,
    stack=stack_sequence,
    features=word_features_from_array,
)

ONE_HAND = Mode(
    name="one-hand",
    csv=Path(config.ONEHAND_DATASET_CSV),
    raw_dir=Path(config.ONEHAND_RAW_DIR),
    n_features=config.ONEHAND_FEATURES,
    prefix="o",
    classes=labels.ONE_HAND_CLASSES,
    keys={**_word_keys(labels.ONE_HAND_WORDS),
          **{m.lower(): m for m in labels.MOTION_LETTERS}},
    segmenter=config.ONEHAND_SEGMENTER,
    hands=1,
    stack=stack_onehand,
    features=onehand_features_from_array,
)


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------

def clean_signer(name: str) -> str:
    """Folder- and CSV-safe signer id: lowercase letters, digits, - and _."""
    name = re.sub(r"[^a-z0-9_-]+", "_", name.strip().lower()).strip("_")
    if not name:
        raise SystemExit("Signer name is empty after cleaning; use letters or digits.")
    return name


def load_rows(mode: Mode):
    if not mode.csv.exists():
        return []
    with open(mode.csv, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return []
    if rows[0] != mode.header:
        flag = " --one-hand" if mode is ONE_HAND else ""
        raise SystemExit(
            f"{mode.csv} has a different column layout from the current {mode.name} "
            f"features ({len(rows[0])} columns, expected {len(mode.header)}). Run\n"
            f"    uv run collect_words.py{flag} --rebuild\n"
            f"to regenerate it from the raw takes, then collect again.")
    return rows[1:]


def save_rows(mode: Mode, rows):
    mode.csv.parent.mkdir(parents=True, exist_ok=True)
    with open(mode.csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(mode.header)
        writer.writerows(rows)


def save_raw(mode: Mode, raw: np.ndarray, label: str, signer: str) -> Path:
    folder = mode.raw_dir / signer
    folder.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
    path = folder / f"{label}_{stamp}.npy"
    np.save(path, raw.astype(np.float32))
    return path


def rebuild(mode: Mode):
    """Regenerate a CSV from every raw take on disk.

    Takes that ran to the segmenter's length cap are skipped: they never
    ended by the hands settling, so they do not look like a live sign.
    """
    if not mode.raw_dir.exists():
        raise SystemExit(f"No raw takes at {mode.raw_dir}; nothing to rebuild.")
    rows, skipped = [], []
    known = set(mode.classes)
    cap = mode.segmenter["max_frames"]
    for path in sorted(mode.raw_dir.glob("*/*.npy")):
        signer = path.parent.name
        label = path.stem.rsplit("_", 1)[0]
        if label not in known:
            skipped.append((path, f"label {label!r} is not a {mode.name} class"))
            continue
        raw = np.load(path)
        if raw.shape[0] >= cap:
            skipped.append((path, f"ran to the {cap}-frame cap without settling"))
            continue
        feats = mode.features(raw)
        if feats is None:
            skipped.append((path, "a hand is missing from too many frames"))
            continue
        rows.append([label, signer] + feats.tolist())
    save_rows(mode, rows)
    print(f"Rebuilt {mode.csv} from {len(rows)} raw takes.")
    for path, why in skipped:
        print(f"  skipped {path.relative_to(mode.raw_dir)}: {why}")
    _print_counts(mode, rows)


def _print_counts(mode: Mode, rows):
    counts = {}
    for r in rows:
        counts.setdefault(r[1], {}).setdefault(r[0], 0)
        counts[r[1]][r[0]] += 1
    for signer, per in sorted(counts.items()):
        print(f"  {signer}: " + "  ".join(f"{c}:{per.get(c, 0)}" for c in mode.classes))


# --------------------------------------------------------------------------
# Live collection
# --------------------------------------------------------------------------

def collect(mode: Mode, signer: str):
    import cv2
    from .tracking import (bgr_frame_to_mp_image, create_landmarker, draw_hand,
                           hands_with_labels)

    rows = load_rows(mode)
    counts = {c: 0 for c in mode.classes}
    for r in rows:
        if r[1] == signer and r[0] in counts:
            counts[r[0]] += 1
    session = []                            # [(row, raw_path)] for undo

    # Two hands are always detected, so the pairer can tell the hands apart
    # in one-hand mode too (the fingerspelling hand is used, as live).
    landmarker = create_landmarker(num_hands=2)
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam (index {config.CAMERA_INDEX}).")

    pairer = HandPairer(config.DOMINANT_HAND)
    segmenter = WordSegmenter(**mode.segmenter)
    dom_tag = config.DOMINANT_HAND[0].upper()
    other_tag = "L" if dom_tag == "R" else "R"

    armed, armed_until = None, 0.0
    peak = 0.0                              # fastest wrist movement this take
    start = time.time()

    print(f"{mode.name.capitalize()} collection, signer '{signer}'. Keys:")
    for k, w in mode.keys.items():
        print(f"  {k.upper()}  {w}")
    print("  SPACE  NONE (between-sign movement)\n  BACKSPACE undo   Q/ESC quit")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            now = time.time()
            result = landmarker.detect_for_video(
                bgr_frame_to_mp_image(frame), int((now - start) * 1000))

            for hand_lms in result.hand_landmarks:
                draw_hand(frame, hand_lms)
            hands = hands_with_labels(result)
            dom, nondom = pairer.pair(hands)
            h, w = frame.shape[:2]
            for arr, tag in ((dom, dom_tag), (nondom, other_tag)):
                if arr is not None:
                    x, y = int(arr[0, 0] * w), int(arr[0, 1] * h)
                    cv2.putText(frame, tag, (x - 10, y + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)

            # ---- recording ----------------------------------------------
            if armed is not None:
                if mode.hands == 2:
                    sign = segmenter.update(dom, nondom)
                else:
                    sign = segmenter.update(dom if dom is not None else nondom, None)
                if segmenter.is_ready or segmenter.is_moving:
                    peak = max(peak, segmenter.last_speed)
                if sign is not None:
                    raw = mode.stack(sign)
                    feats = mode.features(raw)
                    if segmenter.end_reason != "settled":
                        why = ("you kept moving until the length limit"
                               if segmenter.end_reason == "max_frames"
                               else "the hand left the frame")
                        print(f"Discarded '{armed}': {why}. Hold still at the end "
                              f"of the sign so it finishes cleanly.")
                    elif feats is None:
                        print(f"Discarded '{armed}': a hand was out of view for "
                              f"too much of the sign.")
                    else:
                        path = save_raw(mode, raw, armed, signer)
                        row = [armed, signer] + feats.tolist()
                        rows.append(row)
                        session.append((row, path))
                        counts[armed] += 1
                        save_rows(mode, rows)
                        print(f"Saved '{armed}' ({len(sign)} frames, "
                              f"{signer} total {counts[armed]})")
                    armed = None
                elif not segmenter.is_moving and now > armed_until:
                    if segmenter.is_ready:
                        name = ("ONEHAND_SEGMENTER" if mode is ONE_HAND
                                else "WORD_SEGMENTER")
                        print(f"Discarded '{armed}': no movement detected within "
                              f"{config.WORD_ARM_TIMEOUT:.0f}s. Fastest wrist movement "
                              f"seen: {peak:.3f} per frame; a sign starts above "
                              f"move_on = {mode.segmenter['move_on']}. If you WERE "
                              f"signing, set move_on in {name} (config.py) a little "
                              f"below {peak:.3f}.")
                    else:
                        print(f"Discarded '{armed}': hands never settled. Hold still "
                              f"in view until the screen says GO.")
                    armed = None

            # ---- overlay ------------------------------------------------
            cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
            if armed is None:
                status, colour = "Press a key to record a take", (255, 255, 255)
            elif segmenter.is_moving:
                status, colour = f"RECORDING {armed}", (0, 165, 255)
            elif segmenter.is_ready:
                status, colour = f"GO: sign {armed}", (0, 255, 0)
            else:
                status, colour = f"{armed}: hold still...", (0, 255, 255)
            cv2.putText(frame, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2)
            n = len(hands)
            ok_hands = n == mode.hands
            need = "need 2 hands" if mode.hands == 2 else "use ONE hand"
            cv2.putText(frame, f"{mode.name}   signer: {signer}   hands in view: {n}"
                        + ("" if ok_hands else f"   ({need})"),
                        (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (180, 180, 180) if ok_hands else (0, 0, 255), 1)

            legend = [f"{k.upper()} {c} {counts[c]}" for k, c in mode.keys.items()]
            legend.append(f"SPC NONE {counts[labels.NONE_SIGN]}")
            for i, line in enumerate(legend):
                cv2.putText(frame, line, (w - 175, 92 + 18 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            cv2.imshow(f"{mode.name.capitalize()} data collection", frame)
            key = cv2.waitKey(1) & 0xFF

            new_take = None
            if key in (_KEY_ESC, ord("q")):
                break
            elif key == _KEY_BACKSPACE and armed is None and session:
                row, path = session.pop()
                rows.remove(row)
                counts[row[0]] -= 1
                path.unlink(missing_ok=True)
                save_rows(mode, rows)
                print(f"Undid last take ('{row[0]}')")
            elif armed is None and key == _KEY_SPACE:
                new_take = labels.NONE_SIGN
            elif armed is None and 0 <= key < 256 and chr(key).lower() in mode.keys:
                new_take = mode.keys[chr(key).lower()]

            if new_take is not None:
                armed = new_take
                peak = 0.0
                segmenter.reset()
                armed_until = now + config.WORD_ARM_TIMEOUT
                what = ("move like a between-signs transition"
                        if armed == labels.NONE_SIGN else f"sign {armed} once")
                print(f"Armed '{armed}': hold still, then {what}.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        save_rows(mode, rows)
        print(f"{len(rows)} takes in {mode.csv}")
        _print_counts(mode, rows)


def main(argv=None):
    p = argparse.ArgumentParser(description="Collect word sign takes.")
    p.add_argument("--one-hand", action="store_true",
                   help="collect one-hand signs (one-hand words, J, Z) instead of "
                        "two-hand words")
    p.add_argument("--signer", default="unknown",
                   help="who is signing this session (optional, default: unknown)")
    p.add_argument("--rebuild", action="store_true",
                   help="regenerate the CSV from the raw takes and exit")
    args = p.parse_args(argv)
    mode = ONE_HAND if args.one_hand else TWO_HAND
    if args.rebuild:
        rebuild(mode)
        return
    collect(mode, clean_signer(args.signer))


if __name__ == "__main__":
    main(sys.argv[1:])
