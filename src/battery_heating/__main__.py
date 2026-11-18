"""Command-line entry point: ``python -m battery_heating <stage>``."""

from __future__ import annotations

import argparse
import logging
import sys

from battery_heating import pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="battery_heating",
        description="Interpretable ML for cabin-heater activity in a battery electric vehicle.",
    )
    parser.add_argument(
        "stage",
        choices=[*pipeline.STAGES, "all"],
        help="pipeline stage to run (prepare, describe, train, interpret, report) or 'all'",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug-level logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if args.stage == "all":
        pipeline.run_all()
    else:
        pipeline.STAGES[args.stage]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
