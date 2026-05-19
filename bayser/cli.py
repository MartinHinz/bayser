from __future__ import annotations

import argparse

from bayser.workflow import run_analysis


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Bayesian seriation workflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -------------------------------------------------------------------------
    # Input
    # -------------------------------------------------------------------------

    parser.add_argument("--features", required=True)
    parser.add_argument("--c14", default=None)
    parser.add_argument("--intcal20", default=None)

    # -------------------------------------------------------------------------
    # CSV parsing
    # -------------------------------------------------------------------------

    parser.add_argument("--feature-sep", default=",")
    parser.add_argument("--c14-sep", default=",")
    parser.add_argument("--feature-id-col", default=None)
    parser.add_argument("--c14-id-col", default=None)
    parser.add_argument("--bp-col", default=None)
    parser.add_argument("--error-col", default=None)

    # -------------------------------------------------------------------------
    # Filtering
    # -------------------------------------------------------------------------

    parser.add_argument("--filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-type-count", type=int, default=2)
    parser.add_argument("--min-grave-count", type=int, default=2)
    parser.add_argument("--exclude-col", action="append", default=[])
    parser.add_argument("--exclude-regex", default=None)

    # -------------------------------------------------------------------------
    # Classical comparison
    # -------------------------------------------------------------------------

    parser.add_argument("--ra-method", choices=["ca", "iterative"], default="ca")

    # -------------------------------------------------------------------------
    # Sampling
    # -------------------------------------------------------------------------

    parser.add_argument("--draws", type=int, default=800)
    parser.add_argument("--tune", type=int, default=1200)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.96)
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument("--max-treedepth", type=int, default=12)

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------

    parser.add_argument("--repulsion-strength", type=float, default=0.35)
    parser.add_argument(
        "--include-richness", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--calendar-grid-step", type=int, default=10)
    parser.add_argument("--local-window-padding", type=float, default=500.0)

    # -------------------------------------------------------------------------
    # Outlier handling
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--outlier",
        action="append",
        default=[],
        metavar="GRAVE_ID[:PRIOR]",
        help=(
            "Enable the OxCal-style outlier component for one retained dated assemblage. "
            "Use GRAVE_ID or GRAVE_ID:PRIOR, for example ASO_6 or ASO_6:0.5. "
            "If PRIOR is omitted, 0.5 is used. Can be repeated."
        ),
    )

    parser.add_argument(
        "--outlier-all",
        type=float,
        default=None,
        metavar="PRIOR",
        help=(
            "Convenience option: enable the outlier component for all retained dated assemblages "
            "with the same prior probability, for example --outlier-all 0.05."
        ),
    )

    # -------------------------------------------------------------------------
    # Output verbosity
    # -------------------------------------------------------------------------

    parser.add_argument(
        "--quiet",
        action="store_true",
        help=(
            "Print only a minimal run summary. "
            "Useful for benchmark runs or repeated model comparisons."
        ),
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help=("Print additional input, calibration, posterior, and outlier tables."),
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Print detailed sampler diagnostics, parameter diagnostics, and debug tables. "
            "Implies --verbose."
        ),
    )

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------

    parser.add_argument("--plot-dir", default="plots")
    parser.add_argument("--plot-dpi", type=int, default=200)
    parser.add_argument("--show-plots", action="store_true")

    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------

    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--no-results", action="store_true")

    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    # Debug output includes the verbose layer.
    if args.debug:
        args.verbose = True

    # Quiet should win over verbose/debug for user-facing output.
    # The workflow can still decide to print warnings if something is seriously wrong.
    if args.quiet:
        args.verbose = False
        args.debug = False

    run_analysis(args)


if __name__ == "__main__":
    main()
