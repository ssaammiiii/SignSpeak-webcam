"""
Orchestrate the SignSpeak runtime loop:

    gesture source -> confidence gate -> token buffer -> sentence -> speech

Tokens are single letters (fingerspelling), control tokens ('space',
'speak', 'neutral') and, when a word model is loaded, whole words such as
'doctor'. Whole words get a word break on both sides, so spelled letters
never run into them: "im" followed by the SICK sign stays "im sick", not
"imsick", and the sentence stage sees the word as a unit.
"""

from typing import Iterable

from . import config, labels, sentence, tts
from .sources import Gesture


def run(source: Iterable[Gesture], use_llm: bool = True) -> None:
    """Consume gestures; speak a sentence each time 'speak' is signed."""
    buffer: list[tuple[str, float]] = []   # (letter / word / space, confidence)

    for symbol, confidence in source:
        if confidence < config.CONFIDENCE_THRESHOLD:
            continue

        if symbol == "neutral":
            continue
        if symbol == "space":
            buffer.append((" ", 1.0))
        elif symbol == "speak":
            if any(ch.strip() for ch, _ in buffer):
                text = sentence.build(buffer, use_llm=use_llm)
                if text.strip():
                    tts.speak(text)
                else:
                    print("[pipeline] nothing intelligible to speak; buffer discarded.")
            buffer.clear()
        elif labels.is_word_token(symbol):
            if buffer and buffer[-1][0] != " ":
                buffer.append((" ", 1.0))
            buffer.append((symbol, confidence))
            buffer.append((" ", 1.0))
        else:
            buffer.append((symbol, confidence))