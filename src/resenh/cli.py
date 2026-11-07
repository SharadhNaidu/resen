"""Command-line interface for resenh."""

import argparse

from resenh import __version__


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="resenh", description=__doc__)
    parser.add_argument("--version", action="version", version=f"resenh {__version__}")
    parser.parse_args(argv)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
