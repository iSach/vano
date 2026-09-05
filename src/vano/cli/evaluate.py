"""``vano-evaluate``: report a trained run's paper metrics."""

import argparse
import json
from pathlib import Path

from ..evaluate import evaluate
from ..train import load


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="run directory produced by vano-train")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--json", help="also write the metrics here")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    model, cfg = load(args.run, device=args.device)
    metrics = evaluate(model, cfg, device=args.device)
    text = json.dumps(metrics, indent=2)
    print(text)
    if args.json:
        Path(args.json).write_text(text)


if __name__ == "__main__":
    main()
