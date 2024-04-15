#!/usr/bin/env python3
# THIS IS A CUSTOMZIED VERSION FOR THE DPU DATA XFER PLOT
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import argparse

# defaults for nice publication ready rendering
fontsize = 12
legend_fontsize = 10


def read_csv(filename):
    df = pd.read_csv(filename)
    return df


def plot_results(results, filename, yaxis='time', **kwargs):
    # 6.8 inch high figure, 2.5 inch across (matches column width)
    fig, ax = plt.subplots(figsize=(6.8, 2.5))

    # Note: scaling down to byte to use EngFormatter
    ax.axhline(19.456, label="DDR4 Memory Channel x1", linestyle="--", color='red')
    ax.axhline(19.456*4, label="DDR4 Memory Channel x4", linestyle="--", color='orange')

    # for more: https://matplotlib.org/3.2.2/api/markers_api.html
    # markers = ['o', 'v', 's', 'p', '*', 'D', 'x']

    results['rate(MB/s)'] = results['rate(MB/s)'] / 1024
    results["size"] /= 1024
    ax.plot(results["size"], results["rate(MB/s)"], label="DPU Aggregate")

    # set up legend
    ncol = kwargs.get('ncol', 1)
    ax.legend(ncol=ncol, loc='upper left', fontsize=legend_fontsize)

    # configure ticks to be what is in the columns
    ax.xaxis.set_major_locator(plticker.MultipleLocator(4))
    ax.yaxis.set_major_locator(plticker.MultipleLocator(64))

    # special tick for ddr4
    extraticks = [19, 19*4]
    plt.yticks(list(plt.yticks()[0]) + extraticks)

    # set top-axis
    ticks = plticker.MultipleLocator(64).tick_values(0, max(results['rate(MB/s)']))
    ax.set_ylim(0, max(ticks))

    ax.set_ylabel("Throughput (GB/s)", fontsize=fontsize)
    ax.set_xlabel("Memory (GB)", fontsize=fontsize)

    # grids
    ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.xaxis.grid(color='gray', linestyle='dashed')

    print(f"writing file to {filename}")
    plt.savefig(filename, bbox_inches='tight', dpi=300)


parser = argparse.ArgumentParser()
parser.add_argument('--results_csv', '-r', help="results csv file", required=True)
parser.add_argument('--output',  '-o', help="output file name", required=True)
args = parser.parse_args()
print(args)

results = read_csv(args.results_csv)
# a rather circuitious way to avoid arguments in the args from being None
kwargs = dict(filter(lambda x: x[1] is not None, vars(args).items()))
plot_results(results, args.output, **kwargs)
