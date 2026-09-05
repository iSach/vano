"""Appendix Tables 3-4: trainable parameters and training cost per benchmark.

Reads whatever runs already exist and prints one row each.

    python experiments/cost_table.py
"""

import argparse
import json

from _common import RUNS


def rows():
    for path in sorted(RUNS.glob("*/*/history.json")):
        cfg = json.loads((path.parent / "config.json").read_text())
        meta = json.loads(path.read_text())
        yield {
            "run": path.parent.name,
            "latent_dim": cfg["latent_dim"],
            "parameters": meta["num_parameters"],
            "steps": cfg["training"]["max_steps"],
            "minutes": (meta["wall_time_s"] or 0.0) / 60.0,
            "final_recon": meta["history"][-1]["recon_loss"],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", help="substrings a run must match")
    args = parser.parse_args()

    print(f"| {'run':32s} | params | steps | minutes | final recon |")
    print("| " + " | ".join(["-" * 3] * 5) + " |")
    for row in rows():
        if args.only and not any(k in row["run"] for k in args.only):
            continue
        print(f"| {row['run']:32s} | {row['parameters'] / 1e6:.3f}M "
              f"| {row['steps']} | {row['minutes']:.1f} | {row['final_recon']:.3e} |")


if __name__ == "__main__":
    main()
