"""
Turn a fingerspelled letter stream into a clean, punctuated sentence.

    "iamhungrypleasegivemefood"  ->  "I am hungry, please give me food."

`build()` is the only function callers need. It uses the LLM and silently falls
back to offline segmentation when the LLM is unavailable (no key / no network).
"""

import re

from . import config

_SYSTEM_PROMPT = """\
You are the text stage of a sign-language fingerspelling recognizer. You \
receive a lowercase stream of letters that a user spelled out one hand \
shape at a time. The stream may have no spaces, or only some, and it may \
contain a few wrong letters from the recognizer confusing similar hand \
shapes (common mix-ups: a/s/t/m/n/e, u/v/r/k, c/o, g/h, p/q, i/j).
specially m and n.

Letters the recognizer was UNSURE about are marked with a trailing \
asterisk, like h*. Treat marked letters as the most likely to be wrong: \
prefer corrections at marked positions, and change unmarked letters only \
when a word clearly requires it. Keep corrections minimal - do not invent \
words the user did not plausibly spell.

The stream may also contain short bursts of NOISE - stray letters from \
misrecognized hand movement specially random j's that do not belong to any word the user was \
plausibly spelling. Silently drop such letters rather than forcing them \
into words; prefer dropping marked (asterisked) letters, and drop unmarked \
ones only when they clearly fit no surrounding word. Do not drop so much \
that you lose words the user did spell. If the entire stream is \
unintelligible noise with no recoverable words, return an empty response \
with no text at all.

The stream may contain names Non-English descent names such as Middle eastern eg Persian, West African\
such as Ghanaian and Indian. If there are spelling mistakes try to edit them to your best capabilities.

Some tokens are not spelled at all: they are whole words the recognizer \
read directly from a two-handed sign. These appear in square brackets, \
like [doctor] or [sick]. Treat a bracketed word as reliable vocabulary: \
keep it in the sentence as that word (without the brackets), adjusting \
only its form if grammar needs it. A bracketed word followed by an \
asterisk, like [fever]*, was recognized with low confidence; keep it \
unless it makes the sentence nonsensical. Never split a bracketed word \
into letters or merge it with neighbouring letters.

Segment the stream into real English words, repair errors as described, \
add correct capitalization and punctuation, and return one natural \
sentence. Never include asterisks or brackets in your output. Output only \
the sentence - no quotes, no explanation."""

# Letters at or below this confidence get the "unsure" marker in the prompt.
UNSURE_BELOW = 0.75


def _to_marked(letters) -> tuple[str, str]:
    """Normalize input (plain string OR list of (token, conf) pairs) into
    (plain_text, llm_marked_text).

    Tokens are single letters, " " word breaks, or whole words from the word
    model ("doctor"). Whole words always stand apart from neighbouring
    letters; for the LLM they are bracketed ("[doctor]") so it knows they
    were signed as words, and starred after the bracket when unsure.
    """
    if isinstance(letters, str):
        return letters.strip(), letters.strip()

    def render(tok: str, conf: float, mark: bool) -> str:
        if tok == " ":
            return " "
        if len(tok) > 1:                                   # whole-word sign
            if not mark:
                return f" {tok} "
            return f" [{tok}]" + ("*" if conf <= UNSURE_BELOW else "") + " "
        return tok if (not mark or conf > UNSURE_BELOW) else tok + "*"

    plain = re.sub(r"\s+", " ", "".join(render(t, c, False) for t, c in letters)).strip()
    marked = re.sub(r"\s+", " ", "".join(render(t, c, True) for t, c in letters)).strip()
    return plain, marked

# wordninja fuses a handful of common word pairs; undo them in the fallback.
_OFFLINE_FIXUPS = {
    "iam": "i am",
    "togo": "to go",
    "thankyou": "thank you",
    "wantto": "want to",
    "haveto": "have to",
    "goingto": "going to",
}


def _from_llm(letters: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client()  # reads GEMINI_API_KEY from the environment
    reply = client.models.generate_content(
        model=config.LLM_MODEL,
        contents=letters.strip(),
        config=types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            max_output_tokens=config.LLM_MAX_TOKENS,
            thinking_config=types.ThinkingConfig(thinking_budget=0),  # not needed here
        ),
    )
    return reply.text.strip()


def _from_offline(letters: str) -> str:
    import wordninja

    words: list[str] = []
    for word in wordninja.split(letters.lower()):
        words.extend(_OFFLINE_FIXUPS.get(word, word).split())
    if not words:
        return ""

    sentence = " ".join(words)
    sentence = sentence[0].upper() + sentence[1:]
    sentence = re.sub(r"\bi\b", "I", sentence)
    return sentence if sentence.endswith((".", "!", "?")) else sentence + "."


def build(letters, use_llm: bool = True) -> str:
    """Convert a letter stream to a sentence, preferring the LLM.

    `letters` may be a plain string ("hello") or a list of (char, confidence)
    pairs; with pairs, low-confidence letters are flagged to the LLM so its
    corrections target the positions recognition was unsure about. The
    offline fallback always uses the plain letters (wordninja has no use for
    confidence).
    """
    plain, marked = _to_marked(letters)
    if use_llm:
        try:
            return _from_llm(marked)
        except Exception as error:
            print(f"[sentence] LLM unavailable ({error}); using offline fallback.")
    return _from_offline(plain)