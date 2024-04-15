#!/usr/bin/env python3

import argparse
import sys
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rc("hatch", linewidth=0.3)
hatches = [
    "////",
    "\\\\\\\\\\",
    "xxxxxx",
    "-----",
    "||||||",
    "+++++",
    "//",
    "*",
    "+",
    "O",
    "o",
    "-",
]


def parse_args() -> argparse.Namespace:
    """
    Parse commandline arguments
    :return: arguments
    """
    parser = argparse.ArgumentParser(description="Generate a bar chart")

    parser.add_argument(
        "--csvfile", help="Input CSV data file", required=True, type=str
    )
    parser.add_argument("--xlabel", help="Chart x-axis label", default="DPUs", type=str)
    parser.add_argument(
        "--ylabel", help="Chart y-axis label", default="Time (s)", type=str
    )
    parser.add_argument(
        "--xstepsize", help="Chart x-axis stepsize", default=None, type=float
    )
    parser.add_argument(
        "--xstart", help="x-axis start position", default=None, type=int
    )
    parser.add_argument("--xstop", help="x-axis stop position", default=None, type=int)
    parser.add_argument(
        "--ystepsize", help="Chart y-axis stepsize", default=None, type=float
    )
    parser.add_argument(
        "--title", help="Chart title", default="PIM-HDC Runtimes", type=str
    )
    parser.add_argument(
        "--outputfile", help="Output chart", default="output.pdf", type=str
    )
    parser.add_argument("--headers", nargs="+", help="Headers", required=True)

    args = parser.parse_args()

    if not ((args.xstop is None) and (args.xstart is None)):
        if (args.xstop is None) or (args.xstart is None):
            parser.error("--xstop requires --xstart")

    return args


def calculate_bottom(bottom: List[float], added_bar: List[float]) -> List[float]:
    """
    Calculate the bottom for a bar graph given the existing bottom
    :param bottom:     Existing bottom
    :param added_bar:  Bar to add to bottom
    :return:           New bottom
    """
    return [sum(x) for x in zip(bottom, added_bar)]


def main():
    config = parse_args()

    columns = np.loadtxt(config.csvfile, delimiter=",", unpack=True, skiprows=1)

    row_len = len(columns)
    if row_len != len(config.headers):
        print(
            f"Header length '{len(config.headers)}' must equal data row length '{row_len}'",
            file=sys.stderr,
        )
        exit(-1)

    # 6.8 inch high figure, 2.5 inch across (matches column width of paper)
    fig, ax = plt.subplots(figsize=(6.8, 2.5))

    list_dpus = columns[-1]

    bottom = [0.0] * (len(columns[0]))
    # compute spacing between DPU bars, first figure out the step
    # guess based on the first two, and assert the rest are the same
    # otherwise rendering will be off
    step = list_dpus[1] - list_dpus[0]
    for a, b in zip(list_dpus, list_dpus[1:]):
        check_step = b - a
        assert step == check_step, (f"DPU increment isn't consistent, expected {step} based"
                                    " on first two samples, got: {check_step}")
    # add a small amount of spacing for each to get some white between each bar
    step -= 1

    for i, (column, header) in enumerate(zip(columns[:-1], config.headers)):
        ax.bar(
            list_dpus,
            column,
            bottom=bottom,
            label=header,
            width=step,
            hatch=hatches[i % len(hatches)],
            alpha=0.9,
        )
        bottom = calculate_bottom(bottom, column)

    plt.title(config.title)

    start, end = ax.get_xlim()
    if config.xstart is not None:
        start, end = config.xstart, config.xstop + 1

    if config.xstepsize is not None:
        ax.xaxis.set_ticks(np.arange(start, end, config.xstepsize))
    else:
        ax.xaxis.set_ticks(np.arange(start, end))

    if config.ystepsize is not None:
        start, end = ax.get_ylim()
        ax.yaxis.set_ticks(np.arange(start, end, config.ystepsize))

    # add grid
    ax.set_axisbelow(True)
    ax.yaxis.grid(color="gray", linestyle="dashed")

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis and reverse order
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc='center left', bbox_to_anchor=(1, 0.5))

    plt.xlabel(config.xlabel)
    plt.ylabel(config.ylabel)

    plt.savefig(config.outputfile, bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()
