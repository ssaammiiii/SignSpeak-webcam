"""
Headless checks for the word-model building blocks (signspeak/words.py) and
the token handling in the pipeline. No camera needed.

    uv run python tests/test_words.py

Prints one line per check and exits non-zero if any fail.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from signspeak import config, labels                      # noqa: E402
from signspeak.words import (                              # noqa: E402
    DOM, NONDOM, HandPairer, ModeArbiter, WordSegmenter,
    fill_gaps, stack_sequence, word_features, word_features_from_array,
    onehand_features, onehand_features_from_array, stack_onehand,
    _normalize_hands,
)

# --------------------------------------------------------------------------
# A crude but anatomically ordered 21-point hand, in image units, wrist at 0.
# Fingers point up the image (negative y). Only used to drive the tests.
# --------------------------------------------------------------------------
_BASE = np.array([
    [0.00, 0.00, 0.00],                                                     # wrist
    [-0.03, -0.02, 0], [-0.05, -0.04, 0], [-0.06, -0.06, 0], [-0.07, -0.08, 0],  # thumb
    [-0.02, -0.08, 0], [-0.02, -0.11, 0], [-0.02, -0.13, 0], [-0.02, -0.15, 0],  # index
    [0.00, -0.08, 0], [0.00, -0.115, 0], [0.00, -0.14, 0], [0.00, -0.16, 0],     # middle
    [0.02, -0.075, 0], [0.02, -0.105, 0], [0.02, -0.125, 0], [0.02, -0.145, 0],  # ring
    [0.035, -0.065, 0], [0.04, -0.085, 0], [0.045, -0.10, 0], [0.05, -0.115, 0],  # pinky
], dtype=np.float32)


def hand_at(x, y, scale=1.0, curl=0.0, rng=None, noise=0.0):
    h = _BASE.copy()
    h[5:, 1] *= (1.0 - curl)            # crude finger curl
    h = h * scale
    h[:, 0] += x
    h[:, 1] += y
    if rng is not None and noise:
        h[:, :2] += rng.normal(0, noise, size=(21, 2)).astype(np.float32)
    return h


RESULTS = []


def check(name, cond):
    RESULTS.append((name, bool(cond)))
    print(f"  {'ok  ' if cond else 'FAIL'}  {name}")


# --------------------------------------------------------------------------
def test_normalize_matches_tracking():
    try:
        from signspeak.tracking import normalize_landmarks
    except Exception as e:                       # mediapipe/cv2 not importable
        print(f"  skip  normalize matches tracking.normalize_landmarks ({e})")
        return
    rng = np.random.default_rng(0)
    arr = np.stack([[hand_at(0.4, 0.6, 1.3, rng=rng, noise=0.004),
                     hand_at(0.6, 0.5, 0.9, rng=rng, noise=0.004)]
                    for _ in range(5)])
    ours = _normalize_hands(arr).reshape(5, 2, 63)
    ref = np.stack([[normalize_landmarks(arr[t, h]) for h in (0, 1)] for t in range(5)])
    check("words._normalize_hands == tracking.normalize_landmarks",
          np.allclose(ours, ref, atol=1e-6))


def test_pairer():
    p = HandPairer("Right")
    r, l = hand_at(0.7, 0.5), hand_at(0.3, 0.5)
    dom, nd = p.pair([(l, "Left"), (r, "Right")])
    check("pairer: right hand (image right, mirrored) is dominant",
          np.allclose(dom, r) and np.allclose(nd, l))

    p.reset()
    dom, nd = p.pair([(l, "Right"), (r, "Left")])           # labels inverted
    check("pairer: inverted MediaPipe labels cannot swap the hands",
          np.allclose(dom, r) and np.allclose(nd, l))

    lefty = HandPairer("Left")
    dom, nd = lefty.pair([(l, "Left"), (r, "Right")])
    check("pairer: DOMINANT_HAND='Left' takes the image-left hand",
          np.allclose(dom, l) and np.allclose(nd, r))

    # Hands cross slowly; detection order and labels both flip mid-way.
    p.reset()
    p.pair([(r, "Right"), (l, "Left")])
    stable = True
    for t in range(1, 21):
        xr = 0.7 - 0.02 * t
        xl = 0.3 + 0.02 * t
        hr, hl = hand_at(xr, 0.5), hand_at(xl, 0.52)
        dom, nd = p.pair([(hl, "Right"), (hr, "Left")])     # order + labels swapped
        stable &= np.allclose(dom, hr) and np.allclose(nd, hl)
    check("pairer: continuity keeps identities through a crossing", stable)

    p.reset()
    p.pair([(r, "Right"), (l, "Left")])
    dom, nd = p.pair([(hand_at(0.31, 0.5), "Right")])       # only the left is seen
    check("pairer: lone hand goes to the nearest slot, not its label",
          dom is None and nd is not None)


def test_arbiter():
    a = ModeArbiter(enter_frames=6, exit_frames=12)
    flips = [a.update(2) for _ in range(5)]
    check("arbiter: 5 two-hand frames is not enough", a.mode == "letter" and not any(flips))
    a.update(2)
    check("arbiter: 6th two-hand frame enters word mode", a.mode == "word")
    for _ in range(11):
        a.update(1)
    a.update(2)                                             # hand comes back
    for _ in range(11):
        a.update(1)
    check("arbiter: brief drop-outs do not leave word mode", a.mode == "word")
    a.update(1)
    check("arbiter: 12 one-hand frames returns to letter mode", a.mode == "letter")


def _still(n, rng):
    return [(hand_at(0.65, 0.5, rng=rng, noise=0.001),
             hand_at(0.35, 0.55, rng=rng, noise=0.001)) for _ in range(n)]


def _tap(n_taps, rng, frames_per_tap=12, amp=0.12):
    """Dominant hand taps down onto the other hand n times."""
    out = []
    for i in range(n_taps * frames_per_tap):
        phase = (i % frames_per_tap) / frames_per_tap
        dy = amp * np.sin(np.pi * phase)
        out.append((hand_at(0.5, 0.42 - dy * 0.5 + 0.06 * phase, rng=rng, noise=0.001),
                    hand_at(0.42, 0.6, rng=rng, noise=0.001)))
    return out


def test_segmenter():
    rng = np.random.default_rng(1)
    seg = WordSegmenter(**config.WORD_SEGMENTER)

    got = [seg.update(*f) for f in _tap(1, rng)[:6]]
    check("segmenter: motion before settling is ignored",
          all(g is None for g in got) and seg.state == "SETTLING")

    seg.reset()
    outs = [seg.update(*f) for f in _still(10, rng)]
    check("segmenter: settles to IDLE after holding still", seg.is_ready)

    stream = _tap(3, rng) + _still(12, rng)
    found = [g for g in (seg.update(*f) for f in stream) if g is not None]
    check("segmenter: three taps come out as ONE sign", len(found) == 1)
    if found:
        g = found[0]
        bound = len(stream) + config.WORD_SEGMENTER["preroll"]
        check(f"segmenter: segment length {len(g)} within [min_frames, {bound}]",
              config.WORD_SEGMENTER["min_frames"] <= len(g) <= bound)
        check("segmenter: ready for the next sign straight away", seg.is_ready)

    jitter = [(hand_at(0.65 + rng.normal(0, 0.003), 0.5), hand_at(0.35, 0.55))
              for _ in range(40)]
    found = [g for g in (seg.update(*f) for f in jitter) if g is not None]
    check("segmenter: tracking jitter never produces a sign", not found)

    seg.reset()
    for f in _still(10, rng):
        seg.update(*f)
    for f in _tap(1, rng)[:8]:
        seg.update(*f)
    g = seg.update(None, None)
    check("segmenter: losing both hands mid-sign ends it",
          g is not None and seg.state == "SETTLING")

    seg.reset()
    for f in _still(10, rng):
        seg.update(*f)
    seg.update(None, hand_at(0.35, 0.55))         # dominant drops out...
    out = seg.update(hand_at(0.20, 0.2), hand_at(0.35, 0.55))  # ...reappears elsewhere
    check("segmenter: a re-appearing hand is not read as a jump",
          out is None and not seg.is_moving)

    # Both hands close together, each travelling less than min_path alone.
    seg.reset()
    for f in _still(10, rng):
        seg.update(*f)
    each = config.WORD_SEGMENTER["min_path"] * 0.8
    n = 10
    prof = 0.5 - 0.5 * np.cos(np.pi * np.linspace(0, 1, n))
    closing = [(hand_at(0.65 - each * p, 0.5), hand_at(0.35 + each * p, 0.55)) for p in prof]
    end = closing[-1]
    found = [g for g in (seg.update(*f) for f in closing + [end] * 12) if g is not None]
    check("segmenter: symmetric two-hand sign counts both hands' travel",
          len(found) == 1)

    # A smooth one-hand stroke whose path only clears min_path if the
    # pre-roll frames (where it was still accelerating) are counted.
    seg1 = WordSegmenter(**config.ONEHAND_SEGMENTER)
    n = 13
    prof = 0.5 - 0.5 * np.cos(np.pi * np.linspace(0, 1, n))
    total = config.ONEHAND_SEGMENTER["min_path"] * 1.3
    stroke = [hand_at(0.5, 0.45 - total * p) for p in prof]
    feed = [stroke[0]] * 12 + stroke + [stroke[-1]] * 12
    found = [g for g in (seg1.update(h, None) for h in feed) if g is not None]
    check("segmenter: a smooth stroke's build-up in the pre-roll counts toward min_path",
          len(found) == 1)

    seg = WordSegmenter(**{**config.WORD_SEGMENTER, "max_frames": 30})
    for f in _still(10, rng):
        seg.update(*f)
    long_motion = _tap(10, rng)
    found = [g for g in (seg.update(*f) for f in long_motion) if g is not None]
    check("segmenter: continuous motion is capped at max_frames",
          found and all(len(g) <= 30 for g in found))


def test_features():
    rng = np.random.default_rng(2)
    seq = _still(4, rng) + _tap(3, rng) + _still(4, rng)
    f = word_features(seq)
    check(f"features: {config.WORD_FEATURES}-d and finite",
          f is not None and f.shape == (config.WORD_FEATURES,) and np.all(np.isfinite(f)))

    raw = stack_sequence(seq)
    shifted = raw.copy()
    shifted[..., 0] += 0.17
    shifted[..., 1] -= 0.09
    check("features: invariant to where you sit in the frame",
          np.allclose(word_features_from_array(shifted), f, atol=1e-5))

    scaled = raw.copy()
    centre = np.array([0.5, 0.5, 0.0], dtype=np.float32)
    scaled = (scaled - centre) * 0.6 + centre
    check("features: invariant to distance from the camera",
          np.allclose(word_features_from_array(scaled), f, atol=1e-4))

    gappy = raw.copy()
    gappy[3:7, NONDOM] = np.nan                   # ~10% of frames
    g = word_features_from_array(gappy)
    check("features: short hand drop-outs are filled", g is not None)
    check("features: filled result stays close to the clean one",
          g is not None and np.abs(g - f).mean() < 0.05)

    holey = raw.copy()
    holey[::2, NONDOM] = np.nan                   # 50% missing
    check("features: a mostly-missing hand is rejected",
          word_features_from_array(holey) is None)

    taps1 = word_features(_still(4, rng) + _tap(1, rng) + _still(4, rng))
    taps3 = word_features(_still(4, rng) + _tap(3, rng) + _still(4, rng))
    rev1, rev3 = taps1[-2], taps3[-2]
    check(f"features: repetition count separates 1 tap ({rev1:.0f}) from 3 ({rev3:.0f})",
          rev3 > rev1)


def test_end_reason():
    rng = np.random.default_rng(4)
    seg = WordSegmenter(**config.WORD_SEGMENTER)
    for f in _still(10, rng):
        seg.update(*f)
    found = [g for g in (seg.update(*f) for f in _tap(2, rng) + _still(12, rng)) if g]
    check("end reason: a sign that comes to rest is 'settled'",
          found and seg.end_reason == "settled")

    seg = WordSegmenter(**{**config.WORD_SEGMENTER, "max_frames": 30})
    for f in _still(10, rng):
        seg.update(*f)
    found = [g for g in (seg.update(*f) for f in _tap(5, rng)) if g]
    check("end reason: never resting until the cap is 'max_frames'",
          found and seg.end_reason == "max_frames")

    seg = WordSegmenter(**config.WORD_SEGMENTER)
    for f in _still(10, rng):
        seg.update(*f)
    for f in _tap(1, rng)[:8]:
        seg.update(*f)
    seg.update(None, None)
    check("end reason: hands leaving mid-sign is 'hands_lost'",
          seg.end_reason == "hands_lost")


def _hook(rng, scale=1.0, n=16, grow=0.0):
    """One hand drawing a J-like hook; `grow` makes it approach the camera."""
    out = []
    for i in range(n):
        p = i / (n - 1)
        k = scale * (1.0 + grow * np.sin(np.pi * p))
        out.append(hand_at(0.6 - 0.08 * max(0.0, p - 0.6) / 0.4,
                           0.40 + 0.2 * min(p / 0.6, 1.0), k, 0.3, rng, 0.0008))
    return out


def test_onehand():
    rng = np.random.default_rng(5)
    seg = WordSegmenter(**config.ONEHAND_SEGMENTER)
    still = [hand_at(0.6, 0.4, rng=rng, noise=0.0008) for _ in range(10)]
    stream = still + _hook(rng) + [hand_at(0.52, 0.6, curl=0.3)] * 10
    found = [g for g in (seg.update(h, None) for h in stream) if g]
    check("one-hand: a hook drawn with one hand is cut as one sign",
          len(found) == 1 and seg.end_reason == "settled")

    f = onehand_features(found[0]) if found else None
    check(f"one-hand: features are {config.ONEHAND_FEATURES}-d and finite",
          f is not None and f.shape == (config.ONEHAND_FEATURES,) and np.all(np.isfinite(f)))

    raw = stack_onehand(found[0])
    moved = raw.copy()
    moved[..., 0] += 0.2
    moved[..., 1] -= 0.1
    check("one-hand: invariant to where you sit in the frame",
          np.allclose(onehand_features_from_array(moved), f, atol=1e-5))
    centre = np.array([0.5, 0.5, 0.0], dtype=np.float32)
    check("one-hand: invariant to distance from the camera",
          np.allclose(onehand_features_from_array((raw - centre) * 0.6 + centre), f, atol=1e-4))

    flat = onehand_features(_hook(rng))
    closer = onehand_features(_hook(rng, grow=0.4))
    sl = slice(2 * config.WORD_TRAJ_LEN, 3 * config.WORD_TRAJ_LEN)
    check(f"one-hand: moving toward the camera shows in the size curve "
          f"(range {np.ptp(closer[sl]):.2f} vs {np.ptp(flat[sl]):.2f} when flat)",
          np.ptp(closer[sl]) > 3 * np.ptp(flat[sl]))

    holey = raw.copy()
    holey[::2] = np.nan
    check("one-hand: a mostly-missing hand is rejected",
          onehand_features_from_array(holey) is None)

    check("labels: one-hand and two-hand word lists do not overlap",
          not set(labels.ONE_HAND_WORDS) & set(labels.TWO_HAND_WORDS))
    check("labels: the one-hand model covers J and Z",
          {"J", "Z"} <= set(labels.ONE_HAND_CLASSES))


def test_rebuild_skips_capped_takes():
    import dataclasses
    import io
    import contextlib
    import tempfile
    from signspeak.collect_words import ONE_HAND, rebuild
    rng = np.random.default_rng(6)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mode = dataclasses.replace(ONE_HAND, csv=tmp / "o.csv", raw_dir=tmp / "raw")
        (tmp / "raw" / "unknown").mkdir(parents=True)
        cap = config.ONEHAND_SEGMENTER["max_frames"]
        np.save(tmp / "raw/unknown/FINE_1.npy", stack_onehand(_hook(rng)))
        long = stack_onehand([hand_at(0.5, 0.5)] * cap)
        np.save(tmp / "raw/unknown/FINE_2.npy", long)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            rebuild(mode)
        lines = (tmp / "o.csv").read_text().strip().splitlines()
        check("rebuild: a take that hit the length cap is skipped, others kept",
              len(lines) == 2 and "cap" in out.getvalue())


def test_labels_and_tokens():
    check("labels: word tokens never collide with control tokens",
          not ({w.lower() for w in labels.WORD_SIGNS} & set(labels.CONTROL_TOKENS)))
    check("labels: 'doctor' is a word token, 'a' and 'space' are not",
          labels.is_word_token("doctor") and not labels.is_word_token("a")
          and not labels.is_word_token("space"))
    check("labels: word NONE maps to the ignored neutral token",
          labels.to_token(labels.NONE_SIGN) == labels.NEUTRAL)


def test_pipeline_rendering():
    from signspeak import sentence
    buf = [("i", .95), (" ", 1.0), ("a", .95), ("m", .70), (" ", 1.0),
           ("sick", .91), (" ", 1.0), ("h", .9), ("e", .9), ("l", .9), ("p", .9)]
    plain, marked = sentence._to_marked(buf)
    check(f"sentence: plain stream keeps words separate -> {plain!r}",
          plain == "i am sick help")
    check(f"sentence: LLM stream brackets words, stars unsure letters -> {marked!r}",
          marked == "i am* [sick] help")
    low = [("fever", 0.5)]
    check("sentence: an unsure word is starred outside its brackets",
          sentence._to_marked(low)[1] == "[fever]*")
    out = sentence.build(buf, use_llm=False)
    check(f"sentence: offline fallback handles mixed stream -> {out!r}",
          out.lower().startswith("i am sick"))


if __name__ == "__main__":
    for t in (test_normalize_matches_tracking, test_pairer, test_arbiter,
              test_segmenter, test_features, test_end_reason, test_onehand,
              test_rebuild_skips_capped_takes, test_labels_and_tokens,
              test_pipeline_rendering):
        print(t.__name__)
        t()
    failed = [n for n, ok in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    sys.exit(1 if failed else 0)
