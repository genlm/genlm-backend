"""Fresh inference and independent saved-result rendering."""

import argparse
import asyncio
from pathlib import Path

from .bundle import render_bundle


def main() -> None:
    """Run the bundled public panel or render a completed result bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fresh = commands.add_parser("run", help="run fresh target-conditioned inference")
    fresh.add_argument("--output", type=Path, required=True)
    fresh.add_argument(
        "--inputs",
        type=Path,
        default=Path(__file__).parent / "data" / "public4.jsonl",
    )
    fresh.add_argument("--runtime", choices=("mlx", "cuda"), default="mlx")
    fresh.add_argument(
        "--methods",
        nargs="+",
        choices=("clue", "beam", "lookahead"),
        default=("clue", "beam", "lookahead"),
    )
    fresh.add_argument("--seed", type=int, default=20261013)
    fresh.add_argument(
        "--bank-cache",
        type=Path,
        default=Path.home() / ".cache" / "genlm-embedding-inversion" / "contriever.pt",
    )
    saved = commands.add_parser("render", help="render saved results without inference")
    saved.add_argument("--bundle", type=Path, required=True)
    saved.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    match args.command:
        case "run":
            from .runtime import run

            asyncio.run(
                run(
                    args.inputs,
                    args.output,
                    args.runtime,
                    tuple(dict.fromkeys(args.methods)),
                    args.seed,
                    args.bank_cache,
                )
            )
        case "render":
            render_bundle(args.bundle, args.output)


if __name__ == "__main__":
    main()
