#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def escape(value: object) -> str:
    text = str(value)
    for old, new in (("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        text = text.replace(old, new)
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert FaceGuard result CSV files to a LaTeX table.")
    parser.add_argument("csv")
    parser.add_argument("--output", default="outputs/table.tex")
    parser.add_argument("--caption", default="FaceGuard experimental results")
    parser.add_argument("--label", default="tab:faceguard_results")
    args = parser.parse_args()

    frame = pd.read_csv(args.csv)
    column_spec = "|" + "c|" * len(frame.columns)
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        rf"\caption{{{escape(args.caption)}}}",
        rf"\label{{{escape(args.label)}}}",
        r"\footnotesize",
        r"\begin{adjustbox}{max width=\textwidth}",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\hline",
        " & ".join(rf"\textbf{{{escape(column)}}}" for column in frame.columns) + r" \\",
        r"\hline",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.extend([" & ".join(escape(value) for value in row) + r" \\", r"\hline"])
    lines.extend([r"\end{tabular}", r"\end{adjustbox}", r"\end{table*}"])

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved LaTeX table to {output}")


if __name__ == "__main__":
    main()
