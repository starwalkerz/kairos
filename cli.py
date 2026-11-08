"""KAIROS command-line one-shot detection."""

import argparse
import json
import sys

import numpy as np

from core import GmGCycle, from_preset, SYNODIC_PRESETS
from detect import detect


def main() -> int:
    p = argparse.ArgumentParser(description="KAIROS detection")
    p.add_argument(
        "series_file",
        help="CSV file with 't,value' per line, no header",
    )
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--preset", choices=sorted(SYNODIC_PRESETS),
        help="synodic period preset",
    )
    grp.add_argument(
        "--period", type=float,
        help="custom period (same units as t)",
    )
    p.add_argument("--kappa", type=float, default=2.0)
    p.add_argument("--threshold", type=float, default=0.05)
    p.add_argument("--qae", type=int, default=6,
                   help="QAE evaluation qubits")
    p.add_argument("--quantum", action="store_true",
                   help="run the quantum amplitude estimation path")
    p.add_argument("--output", default="-",
                   help="output file for JSON report, or '-' for stdout")
    args = p.parse_args()

    try:
        data = np.loadtxt(args.series_file, delimiter=",")
        if data.ndim != 2 or data.shape[1] != 2:
            raise ValueError("expected two columns: t,value")
        t, x = data[:, 0], data[:, 1]
    except Exception as exc:
        print(f"error reading {args.series_file}: {exc}", file=sys.stderr)
        return 1

    if args.preset:
        cycle = from_preset(args.preset)
    else:
        T = args.period
        if T <= 0:
            print("--period must be positive", file=sys.stderr)
            return 1
        cycle = GmGCycle(
            name="custom",
            period=T,
            phase0=0.0,
            mu_L=-0.5 * T, sigma_L=0.25 * T,
            mu_T=+0.5 * T, sigma_T=0.25 * T,
            amplitude=1.0,
        )

    report = detect(
        x, t, [cycle],
        kappa=args.kappa,
        threshold=args.threshold,
        qae_eval_qubits=args.qae,
        quantum=args.quantum,
    )
    out = json.dumps(report.to_handoff(), indent=2)

    if args.output == "-":
        print(out)
    else:
        with open(args.output, "w") as f:
            f.write(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())