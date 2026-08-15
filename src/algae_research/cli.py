"""Command-line entry point for the consolidated research workflows."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from . import __version__
from .experiments.sound import SoundExperimentConfig, run_sound_experiment
from .simulation.run import run_random_policy, write_simulation_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="algae-research",
        description="Run reproducible algae simulations, experiments, and analyses.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    simulate = commands.add_parser("simulate", help="Run the normalized model with a seeded random policy.")
    simulate.add_argument("--steps", type=int, default=50)
    simulate.add_argument("--seed", type=int, default=0)
    simulate.add_argument("--output", type=Path, default=Path("outputs/simulation.csv"))

    sound = commands.add_parser("sound-experiment", help="Run logged tone exposure sessions in Edge.")
    sound.add_argument("--frequency", type=int, action="append", dest="frequencies")
    sound.add_argument("--duration", type=float, default=10)
    sound.add_argument("--break-seconds", type=float, default=10)
    sound.add_argument("--log", type=Path, default=Path("outputs/sound_experiment_log.csv"))
    sound.add_argument("--headless", action="store_true")
    sound.add_argument("--set-windows-volume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "simulate":
            records = run_random_policy(steps=args.steps, seed=args.seed)
            output = write_simulation_csv(records, args.output)
            summary = {
                "output": str(output),
                "steps": len(records),
                "seed": args.seed,
                "final": asdict(records[-1]),
            }
            print(json.dumps(summary, indent=2))
        elif args.command == "sound-experiment":
            config = SoundExperimentConfig(
                frequencies_hz=tuple(args.frequencies or (200, 10_000, 18_000)),
                duration_seconds=args.duration,
                break_seconds=args.break_seconds,
                log_path=args.log,
                set_windows_volume=args.set_windows_volume,
                headless=args.headless,
            )
            entries = run_sound_experiment(config)
            print(json.dumps([asdict(entry) for entry in entries], indent=2))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"algae-research: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
