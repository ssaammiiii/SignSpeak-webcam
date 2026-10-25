"""
Shared label configuration — the single source of truth for every stage.

Training labels, recognition output, and the translation pipeline's vocabulary
all reference this module, so the mapping can never drift between projects.

Conventions:
  - Models are trained on UPPERCASE letter labels ("A".."Z"), exactly as the
    original Sign_ASL datasets were collected, so existing CSVs and .joblib
    models remain compatible.
  - The pipeline consumes lowercase tokens ("a".."z") plus control tokens.
    `to_token()` is the one place that conversion happens.
"""

# Letters held as a still handshape (25 letters A-Y, minus J).
STATIC_LETTERS = "ABCDEFGHIKLMNOPQRSTUVWXY"

# Letters drawn as a motion.
MOTION_LETTERS = "JZ"

# The full recognizable alphabet.
ALL_LETTERS = "".join(sorted(STATIC_LETTERS + MOTION_LETTERS))

# Control tokens understood by the pipeline (see signspeak/pipeline.py).
NEUTRAL = "neutral"   # ignored
SPACE = "space"       # appends a word break
SPEAK = "speak"       # flushes the buffer through sentence generation + TTS
CONTROL_TOKENS = (NEUTRAL, SPACE, SPEAK)

# Control SIGNS: trainable static classes that emit control tokens.
# Collected alongside letters (collect_static.py: SPACE key / ENTER key),
# trained as ordinary classes, mapped to tokens here. Suggested handshapes
# (any shape distinct from the letters works — it's learned from your data):
#   SPACE -> open "5" hand, all fingers spread
#   SPEAK -> thumbs-up
SPACE_SIGN = "SPACE"
SPEAK_SIGN = "SPEAK"

# NONE: the motion model's "not a gesture" class. Trained on ordinary hand
# transitions (moving between letters, repositioning), it lets the motion
# classifier answer "neither J nor Z" instead of being forced to pick one.
# Maps to the 'neutral' token, which the pipeline silently ignores.
NONE_SIGN = "NONE"
CONTROL_SIGN_LABELS = {SPACE_SIGN: SPACE, SPEAK_SIGN: SPEAK, NONE_SIGN: NEUTRAL}

# Whole-word signs. Same ten words as the glove system's word model, so the
# two sensing approaches can be compared on equal terms, split by how many
# hands each sign uses. Edit freely: collectors, trainer and runtime all
# read these tuples.
#
# Two-hand words are recognised in word mode (both hands in view) by the
# word model. One-hand words are recognised in letter mode by the one-hand
# model, which also handles J and Z, so they are signed naturally with one
# hand. Both models also get the NONE class (above), trained on movements
# that are not signs, so they can answer "not a sign" instead of guessing.
TWO_HAND_WORDS = ("DIZZY", "DOCTOR", "GOOD", "SICK", "WELL")
ONE_HAND_WORDS = ("FEVER", "FINE", "HELP", "HOSPITAL", "NOT")

WORD_SIGNS = TWO_HAND_WORDS                      # the two-hand word model
WORD_CLASSES = WORD_SIGNS + (NONE_SIGN,)

# The one-hand model: one-hand words plus the drawn letters J and Z.
ONE_HAND_SIGNS = ONE_HAND_WORDS + tuple(MOTION_LETTERS)
ONE_HAND_CLASSES = ONE_HAND_SIGNS + (NONE_SIGN,)

# A word token must never be mistaken for a control token by the pipeline.
_all_words = {w.lower() for w in TWO_HAND_WORDS + ONE_HAND_WORDS}
_collisions = _all_words & set(CONTROL_TOKENS)
assert not _collisions, f"word signs clash with control tokens: {_collisions}"
assert not set(TWO_HAND_WORDS) & set(ONE_HAND_WORDS), "a word is listed as both one- and two-hand"


def to_token(model_label: str) -> str:
    """Classifier label -> pipeline token.

    Letters map to lowercase ("A" -> "a"); control-sign classes map to their
    control tokens ("SPACE" -> "space", "SPEAK" -> "speak").
    """
    return CONTROL_SIGN_LABELS.get(model_label, model_label.lower())


def is_letter_token(token: str) -> bool:
    return len(token) == 1 and token.upper() in ALL_LETTERS


def is_word_token(token: str) -> bool:
    """A whole-word sign ("doctor"), as opposed to a letter or a control."""
    return len(token) > 1 and token not in CONTROL_TOKENS