"""
Recognition-only preview with a confidence gate.

Prints every committed token: ACCEPTED if confidence >= threshold,
rejected (shown dimly with an x) if below — so you can see exactly what
the gate is doing and tune the threshold from real evidence.

    uv run recognize.py                  # uses CONFIDENCE_THRESHOLD from config
    uv run recognize.py --threshold 0.9  # try a different gate without editing config
"""
import argparse

from signspeak import config
from signspeak.camera_source import camera_source

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=config.CONFIDENCE_THRESHOLD,
                   help="minimum confidence to accept a token")
    args = p.parse_args()

    accepted, rejected = [], 0
    print(f"Gate: accepting tokens with confidence >= {args.threshold}")
    for symbol, confidence in camera_source():
        if confidence >= args.threshold:
            accepted.append(symbol)
            print(f"  ACCEPT  {symbol:>6}   conf={confidence:.2f}")
        else:
            rejected += 1
            print(f"  x       {symbol:>6}   conf={confidence:.2f}   (below gate)")

    print(f"\nSession: {len(accepted)} accepted, {rejected} rejected")
    print("Accepted stream:", "".join(s if len(s) == 1 else f"[{s}]" for s in accepted))