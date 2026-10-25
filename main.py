"""
SignSpeak entry point — the full unified pipeline:

    webcam -> trained static/motion models -> (letter, confidence) stream
           -> confidence gate -> letter buffer -> LLM sentence -> speech

Requires trained models (see README: collect, then train). To exercise the
pipeline without a camera or models, use the scripted fake source:

    python main.py --fake

The Gemini API key is read from a local .env file (GEMINI_API_KEY). If it's
missing, the sentence stage automatically falls back to offline word
segmentation, so this still runs.
"""

import argparse

from dotenv import load_dotenv

from signspeak import pipeline


def parse_args():
    p = argparse.ArgumentParser(description="Run the SignSpeak pipeline.")
    p.add_argument("--fake", action="store_true",
                   help="use the scripted fake gesture source (no camera/models)")
    p.add_argument("--no-llm", action="store_true",
                   help="skip the LLM and use offline word segmentation")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    load_dotenv()  # loads GEMINI_API_KEY from .env into the environment

    if args.fake:
        from signspeak.sources import fake_source
        source = fake_source()
    else:
        from signspeak.camera_source import camera_source
        source = camera_source()

    pipeline.run(source, use_llm=not args.no_llm)
