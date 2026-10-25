# SignSpeak — Unified Fingerspelling Recognition + AI Sentence Translation

One system, end to end:

```
Data Collection ─→ Train Static Model ─→ Train Gesture (motion) Model
      ─→ models exported to models/
            ─→ Recognition engine loads trained models
                  ─→ recognizes signs in real time from the webcam
                        ─→ yields (letter, confidence) tokens
                              ─→ confidence gate + letter buffer
                                    ─→ LLM sentence generation (Gemini,
                                       offline wordninja fallback)
                                          ─→ spoken aloud (pyttsx3)
```

This merges two formerly separate projects:

- **Sign_ASL** — MediaPipe-landmark-based trainer: a static RandomForest for
  the 24 still letters and a motion RandomForest for J/Z, with a
  `GestureSegmenter` deciding at runtime whether the hand is holding a shape
  or drawing one. All of that logic is preserved intact.
- **sign_speak** — the translation pipeline: any *gesture source* yielding
  `(symbol, confidence)` feeds a confidence gate, a letter buffer, LLM
  sentence generation, and text-to-speech. Also preserved intact.

The join is `signspeak/camera_source.py`: the trainer's live recognition
loop, restructured to *be* a gesture source. The pipeline doesn't know the
difference between it and the old `fake_source()`.

## Project layout

```
main.py                     run the full pipeline (webcam → speech)
recognize.py                recognition-only preview (prints committed tokens)
collect_static.py           collect A–Y (no J) handshape samples
collect_motion.py           collect J and Z drawn-gesture samples
collect_words.py            collect word sign takes (--one-hand for one-hand signs)
train.py                    train the models → models/  (--only static|motion|word|onehand)
tests/test_words.py         headless checks for the word path (no camera)
models/                     trained models + MediaPipe bundle (auto-created)
data/                       collected datasets (auto-created)
signspeak/
  config.py                 every tunable + all paths (env-overridable)
  labels.py                 shared label map: letters + control tokens
  tracking.py               MediaPipe setup, normalization, features, GestureSegmenter
  collect_static.py         static collection logic
  collect_motion.py         motion collection logic
  training.py               training logic (writes to models/)
  models.py                 SignModels: configurable loading + predict_proba confidence
  commit.py                 StaticCommitter: debounce layer for held letters
  words.py                  word signs: hand pairing, mode switch, segmenter, features
  collect_words.py          word collection logic (signer column, raw takes, --rebuild)
  camera_source.py          webcam recognition as a (symbol, confidence) source
  sources.py                fake_source() for camera-free pipeline testing
  pipeline.py               gate → buffer → sentence → speech loop
  sentence.py               letters → sentence (Gemini LLM, wordninja fallback)
  tts.py                    offline text-to-speech
```

## Setup (uv)

```bash
uv python install 3.13
uv sync
```

Python 3.13 is the intersection of the two source projects' requirements
(mediapipe ≥0.10.35 runs fine on it; the LLM stack needs ≥3.13).

For LLM sentence generation, copy `.env.example` to `.env` and paste a free
Gemini key. Without it, everything still runs using offline word
segmentation.

## Training workflow (unchanged from the original trainer)

```bash
uv run collect_static.py   # 1a. static letters A–Y (no J) + the two control signs
uv run collect_motion.py   # 1b. J and Z: press J/Z to record a ~2s drawn gesture
uv run train.py            # 2.  trains models/static_model.joblib + motion_model.joblib
```

### Control signs (word break / speak)

Word breaks and "speak the sentence" are **signed, not typed**. They are two
extra static classes collected in `collect_static.py`: hold your chosen
handshape and press the **SPACE key** to save a SPACE-sign sample, or the
**ENTER key** to save a SPEAK-sign sample (30–50 varied samples each, like
letters). Suggested handshapes — anything clearly distinct from all letters
works, since the shapes are learned from your data:

- **SPACE** → open "5" hand, all fingers spread
- **SPEAK** → thumbs-up

Because these trigger actions, they commit only after a *unanimous* vote
window (`CONTROL_MIN_VOTES`), a stricter bar than letters — hold the sign
steadily for a beat. If a model was trained *without* control-sign classes,
the runtime automatically falls back to keyboard control (SPACE / ENTER
keys) and prints a hint, so older models keep working.

Collection tips, sample counts, and threshold tuning guidance from the
original Sign_ASL README all still apply — the features and dataset formats
are identical.

### Model selection

`train.py` doesn't assume RandomForest. It benchmarks seven classifier
families (rf, extratrees, hgb, knn, svm, mlp, logreg) with stratified
k-fold cross-validation, additionally evaluates a soft-voting **ensemble**
of the top three, prints the leaderboard and a cross-validated per-letter
report, then fits the winner on all data and saves it. Whatever wins —
single model or ensemble — exposes `predict_proba`, so recognition and the
confidence gate work identically.

```bash
uv run train.py               # compare everything, keep the best
uv run train.py --model rf    # skip comparison, force one candidate
uv run train.py --list        # show candidate names
```

With very small datasets (fewer than 4 samples in any class) comparison is
skipped and RandomForest is used until you collect more. Comparison takes
longer than single-model training (SVM/MLP dominate the time) — with a full
alphabet at ~40 samples/letter expect a couple of minutes, not seconds.

**Retraining:** just re-run collection and `train.py`. Recognition loads
models from `models/` at startup, so restart `main.py`/`recognize.py` and it
uses the new models automatically. To point at a different set of models or
datasets without touching code:

```bash
SIGNSPEAK_MODEL_DIR=/path/to/models SIGNSPEAK_DATA_DIR=/path/to/data uv run main.py
```

### Migrating existing data/models

Your previously collected datasets and trained models are fully compatible —
same features, same formats. Copy them in:

```
Sign_ASL/static_dataset.csv   → data/static_dataset.csv
Sign_ASL/motion_dataset.csv   → data/motion_dataset.csv
Sign_ASL/static_model.joblib  → models/static_model.joblib   (optional; or retrain)
Sign_ASL/motion_model.joblib  → models/motion_model.joblib   (optional; or retrain)
```

## Running inference

```bash
uv run recognize.py   # preview: prints committed (symbol, confidence) tokens
uv run main.py        # full pipeline: recognize → sentence → speech
uv run main.py --fake      # exercise the pipeline with no camera/models
uv run main.py --no-llm    # skip the LLM, use offline segmentation
```

In the camera window: fingerspell letters (they commit once each when the
prediction stabilizes — the letter tail shows at the top), hold your
**SPACE sign** for a word break, hold your **SPEAK sign** to turn the buffer
into a sentence and speak it, **Q/ESC** to quit. (Keyboard SPACE/ENTER only
apply as a fallback when the model has no trained control signs.)

**Double letters** (the "hello" problem): a committed letter is suppressed
while you keep holding it. To sign the same letter twice, briefly relax or
move your hand between the two — the committer re-arms when the hand drops
out or the winning letter changes.

## Tuning

Everything lives in `signspeak/config.py`:

- `CONFIDENCE_THRESHOLD` — pipeline gate; now meaningful because the camera
  source yields real `predict_proba` confidences. If valid letters get
  dropped, lower it; if garbage letters slip through, raise it.
- `STATIC_SMOOTH_WINDOW` / `STATIC_MIN_VOTES` — commit responsiveness vs.
  stability.
- `SEGMENTER` dict — J/Z motion detection thresholds (same meanings and
  tuning table as the original Sign_ASL README).

## Word signs (one-hand and two-hand)

The ten words from the glove system's vocabulary are split by how they are
signed, set in `signspeak/labels.py`:

| | words | recognised in | model |
|---|---|---|---|
| **two-hand** | DIZZY, DOCTOR, GOOD, SICK, WELL | word mode (both hands in view) | `word_model.joblib` |
| **one-hand** | FEVER, FINE, HELP, HOSPITAL, NOT | letter mode, like J and Z | `onehand_model.joblib` |

**Two-hand words.** Bring your second hand into view and the system
switches to **word mode** (top-left shows WORD). Sign, and the word goes
straight into the sentence. Drop the second hand to go back to letters.
This is the webcam version of the glove's "raise the left hand" switch.

**One-hand words.** Sign them with one hand while fingerspelling, the same
way you draw J or Z. The one-hand model classifies every one-hand movement
as a one-hand word, J, Z or NONE (an ordinary transition between letters).
It replaces the old J/Z model, so J and Z are recorded again alongside the
one-hand words, and both are cut by the same segmenter as live.

Each model is optional. Until `word_model.joblib` exists the camera uses
one-hand detection and there is no word mode. Until `onehand_model.joblib`
exists, J and Z use the original `motion_model.joblib` exactly as before.

### Workflow

```bash
# two-hand words (keys 1-5, SPACE = NONE)
uv run collect_words.py
uv run train.py --only word

# one-hand words, J and Z (keys 1-5, J, Z, SPACE = NONE)
uv run collect_words.py --one-hand
uv run train.py --only onehand

uv run analyze.py                       # balance and outliers for every dataset
uv run figures.py --only word           # or --only onehand
uv run main.py
```

`--signer NAME` is optional on `collect_words.py`. Without it takes are
filed under `unknown`. With two or more named signers, `train.py` also
reports accuracy on a signer the model has never seen (below).

### Recording a take

1. Press the sign's key (legend on screen). **SPACE** records a NONE take.
2. Hold still until the screen says **GO**: both hands in view for
   two-hand words, only your fingerspelling hand for one-hand signs.
3. Make the sign once, then **hold still**. The take saves itself.

A take is kept only if it ends with your hands coming to rest. If you keep
moving until the length limit, or the hand leaves the frame, the take is
discarded with a message saying so. Takes that run on do not look like a
live sign, and mixed into training they become outliers.

**BACKSPACE** undoes the last take of the session, **Q/ESC** quits.
Everything is written to disk after every take.

Aim for 30 to 40 takes of every class, varying speed, size and where you
sit. **NONE takes are essential for both models**: without them, every
movement is forced into one of the signs.

- Two-hand NONE: raise a hand into view, lower it, reposition, start a sign
  and abandon it. Raising the second hand is what switches modes, so it is
  the most important one to include.
- One-hand NONE: the transitions you make while fingerspelling, moving from
  one letter to the next.

### Raw takes and `--rebuild`

Every take is also saved as a raw landmark array in
`data/word_raw/<signer>/` or `data/onehand_raw/<signer>/`. If the features
are ever changed, regenerate a CSV without re-recording anything:

```bash
uv run collect_words.py --rebuild
uv run collect_words.py --one-hand --rebuild
```

`--rebuild` skips any raw take that ran to the segmenter's length limit and
lists it, so takes recorded before that rule existed are cleaned up too.

### Accuracy on an unseen signer

With two or more named signers in the data, `train.py` prints two numbers
that answer different questions:

- **Stratified k-fold** (the leaderboard and per-class report) mixes every
  signer's takes into both training and testing. It measures how
  consistently the model recognises hands it was trained on.
- **Leave-one-signer-out** trains on everyone else and tests on one signer
  it has never seen, once per signer. This is the cross-user figure.

With unnamed (`unknown`) data only the first is available, so a live test
on new people, with each attempt tallied, is the cross-user evidence.

### How it works

- `HandPairer` decides which hand is dominant (`DOMINANT_HAND` in
  `config.py`, default Right). It goes by position in the mirrored image,
  then tracks each hand frame to frame so the assignment holds even when
  hands cross. MediaPipe's own Left/Right label is not trusted for this.
- `ModeArbiter` enters word mode after two hands are seen for
  `WORD_MODE_ENTER_FRAMES` frames, and leaves after fewer than two for
  `WORD_MODE_EXIT_FRAMES`. Leaving is slower so a hand blurring out of
  detection mid-sign does not drop you back to letters.
- `WordSegmenter` cuts every word sign, two-hand and one-hand. It waits for
  the hands to settle, then cuts one sign from motion onset (plus a short
  pre-roll, whose movement also counts toward the minimum) until the hands
  are still again. The settle-first rule stops the act of raising your hand
  from being read as a sign. The collector uses the same segmenter, so
  training and live signs are cut identically.
- `word_features` (two-hand, 256 numbers): each wrist's path, the distance
  between the palms over time, each hand's average shape and finger
  movement, speed and path length, closest approach between the hands, and
  direction reversals (taps).
- `onehand_features` (one-hand, 262 numbers): the wrist's path, the hand's
  apparent size over time (the only sign of movement toward the camera,
  since MediaPipe gives the wrist no real depth), the handshape at the
  start, middle and end, finger movement, speed, path and reversals.
- Both are measured relative to the hands, in hand lengths, so they do not
  depend on where you sit or how far you are from the camera.
- Training compares the same candidates as for letters, now including
  **LDA with Ledoit-Wolf shrinkage** (the glove system's production model).

**Retraining letters now includes LDA too.** On the current letter data it
ties the ensemble at the top (about 0.998 in 5-fold CV) and is selected,
giving a 62 KB model instead of the 22 MB ensemble. Its confidence values
run differently, so if you retrain letters, re-check
`STATIC_CONFIDENCE_THRESHOLD` with `recognize.py`. To leave the letter
models exactly as they are, train with `--only word` / `--only onehand`.

### Tuning (all in `signspeak/config.py`)

- `move_on` in `WORD_SEGMENTER` / `ONEHAND_SEGMENTER`: when a take times
  out with "no movement detected", the collector prints the fastest wrist
  movement it saw; set `move_on` a little below that number. If "..."
  appears while your hands are resting, raise it.
- `still_frames_to_end`: raise it if a sign with a pause in it (a repeated
  tap, a two-stroke sign such as HOSPITAL's cross) is being cut into two.
- `WORD_CONFIDENCE_THRESHOLD` / `ONEHAND_WORD_CONFIDENCE_THRESHOLD`: gates
  for word tokens. The camera loop applies them before `recognize.py` sees
  anything, so to calibrate: set one to 0, sign each word a few times while
  `recognize.py` runs, note the confidences of right and wrong words, then
  set it between them. J and Z keep `MOTION_CONFIDENCE_THRESHOLD`.
- `WORD_MODE_ENTER_FRAMES` / `WORD_MODE_EXIT_FRAMES`: mode-switch speed.

Run the headless checks after any change to the word code:

```bash
uv run python tests/test_words.py
```

## Notes / limitations

- **Movement toward the camera is hard to see.** The segmenter starts on
  wrist movement across the image. A sign whose motion is mostly toward or
  away from the camera (a tap on the chest, a forward flick) moves the
  wrist very little on screen and may not trigger. The collector's timeout
  message gives the number to set `move_on` to. If that is not enough,
  sitting slightly side-on to the camera turns the movement into one the
  camera can see.
- **Holding a starting handshape can commit a letter first.** If you hold
  the opening shape of a one-hand sign (or of J) before moving, the letter
  model may commit that letter. Start the movement promptly.
- **Hands only, no face or body.** Two signs that differ mainly in *where
  on the body* they are made (forehead versus chest, say) can only be told
  apart by hand shape and movement. If such a pair keeps being confused,
  the next step would be MediaPipe's pose landmarker as a body reference.
- Segmentation thresholds are in image units, like the original J/Z
  segmenter, so a signer much further from the camera makes smaller
  movements on screen and may need `move_on` lowered.
- Control gestures (`space`, `speak`) are trained signs; models without
  them fall back to keyboard SPACE/ENTER automatically.
- `pyttsx3` audio requires a working audio backend; the transcript is always
  printed even when audio is unavailable.
