# SignSpeak (webcam)

Webcam-based sign language recognition that turns fingerspelled letters and a
small vocabulary of word signs into spoken sentences. It is the camera
counterpart to the [SignSpeak sensor glove](https://github.com/ssaammiiii/SignSpeak),
built to compare the two approaches on the same vocabulary.

## Features

- The full ASL alphabet: 24 static handshapes plus the drawn letters J and Z
- Two-hand word signs: DIZZY, DOCTOR, GOOD, HELP, SICK, WELL
- One-hand word signs: FEVER, FINE, HOSPITAL, NOT
- Signed controls for a word break (SPACE) and for speaking the sentence (SPEAK)
- Sentence construction with Gemini, an offline fallback, and spoken output
- Runs on an ordinary laptop webcam using MediaPipe hand tracking

## How it works

MediaPipe returns 21 landmarks per hand for every frame. What happens next
depends on the hands:

| Hands in view | Motion | Recognised by |
|---|---|---|
| one | held still | static letter model |
| one | moving | one-hand model (J, Z or a one-hand word) |
| two | any | two-hand word model |

Recognised letters and words collect in a buffer. Signing SPEAK sends the
buffer to Gemini, which splits the letters into words and forms a sentence,
and the sentence is spoken with pyttsx3. Without an API key, offline word
segmentation (wordninja) is used instead.

Each model is selected by 5-fold cross-validation across eight scikit-learn
classifiers and a soft-voting ensemble of the best three.

## Requirements

- Python 3.13 and [uv](https://docs.astral.sh/uv/)
- A webcam
- Optional: a [Gemini API key](https://aistudio.google.com) for sentence construction

## Setup

```bash
uv sync
cp .env.example .env    # then add your GEMINI_API_KEY
```

The MediaPipe hand model is downloaded automatically on first run.

## Usage

Trained models are not tracked in git, so collect data and train first.

**Collect**

```bash
uv run collect_static.py             # static letters and the SPACE/SPEAK signs
uv run collect_words.py              # two-hand words
uv run collect_words.py --one-hand   # one-hand words, J and Z
```

Key bindings are shown on screen. For word takes, hold still until the screen
shows GO, sign once, then hold still again; the take saves itself. Record the
NONE class (SPACE key) in both word collectors so the models can reject
movements that are not signs.

**Train**

```bash
uv run analyze.py                    # check class balance and outliers
uv run train.py                      # train every model that has data
uv run train.py --only word          # or: static, motion, word, onehand
uv run figures.py                    # leaderboard, confusion matrix, per-class F1
```

**Run**

```bash
uv run main.py                       # recognition, sentence and speech
uv run recognize.py                  # recognition only, with confidences
uv run main.py --fake                # test the pipeline without a camera
uv run main.py --no-llm              # offline sentence construction
```

Fingerspell with one hand. Bring your second hand into view for two-hand words
(the overlay switches to WORD) and lower it to return to letters. Sign SPEAK to
hear the sentence.

## Configuration

Settings live in `signspeak/config.py` and the vocabulary in
`signspeak/labels.py`. The settings most often tuned:

| Setting | Effect |
|---|---|
| `*_CONFIDENCE_THRESHOLD` | minimum confidence for a letter or word to be accepted |
| `move_on` in `WORD_SEGMENTER` / `ONEHAND_SEGMENTER` | how fast a movement must be to count as a sign |
| `DOMINANT_HAND` | the hand used for fingerspelling |

`SIGNSPEAK_MODEL_DIR` and `SIGNSPEAK_DATA_DIR` override the model and data
folders.

## Project structure

```
main.py, recognize.py          entry points
collect_*.py                   data collection
train.py, analyze.py, figures.py
signspeak/
  tracking.py                  MediaPipe setup and landmark features
  words.py                     hand pairing, mode switching, segmentation, word features
  camera_source.py             live recognition loop
  training.py, models.py       model selection and loading
  pipeline.py, sentence.py     token buffer and sentence construction
  tts.py                       speech output
  config.py, labels.py         settings and vocabulary
tests/test_words.py            tests for the word pipeline (no camera needed)
```

## Testing

```bash
uv run python tests/test_words.py
```

## Limitations

- Only hand landmarks are used, so signs that differ mainly in where they are
  made on the body are told apart by hand shape and motion alone.
- Movement directly toward the camera is hard to see from a single webcam, and
  such signs may need a lower `move_on`.
- Signs are recognised one at a time; this is not continuous sign language
  translation.
- Accuracy depends on lighting and on how closely a new signer matches the
  people the models were trained on.

## Authors

Samia Soleimani (lead) and Ishaan Bhardwaj, Department of Computer
Engineering, University of Ghana.
