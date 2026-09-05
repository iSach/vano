"""``vano-train``: train one model from a named configuration."""

import argparse
import json

from ..configs import CONFIGS, get_config
from ..train import train


def parse_overrides(pairs):
    """``latent_dim=32``/``training.max_steps=1000`` -> a dict of typed values."""
    out = {}
    for pair in pairs or []:
        key, _, raw = pair.partition("=")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        out[key] = value
    return out


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", choices=sorted(CONFIGS))
    parser.add_argument("--out", required=True, help="run directory")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--set", nargs="*", metavar="KEY=VALUE",
                        help="override config fields, e.g. training.max_steps=100")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = get_config(args.config).evolve(**parse_overrides(args.set))
    _, history = train(cfg, device=args.device, out_dir=args.out,
                       progress=not args.quiet)
    print(json.dumps(history[-1], indent=2))


if __name__ == "__main__":
    main()
