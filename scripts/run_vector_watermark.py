#!/usr/bin/env python3
from faceguard.watermark.vector_system import build_arg_parser, run_experiment


def main() -> None:
    run_experiment(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
