#!/usr/bin/env python3
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


def plot_results(results, filename, speedup=False, **kwargs):
    # 6.8 inch high figure, 2.5 inch across (matches column width)
    fig, ax = plt.subplots(figsize=(6.8, 2.5))

    # pick out host results, which we assume is always just one number
    host_results = results[results['version'] == 'host']
    if len(host_results) == 0:
        # if no host results present, just assume times are relative to host
        host_results = pd.DataFrame.from_dict({'version': ['host'], 'time': [1]})

    if speedup:
        print(results)
        results['time'] = host_results['time'].values / results['time'] 
        # results['time'] = results['time'] / host_results['time'].values
        host_results = pd.DataFrame.from_dict({'version': ['host'], 'time': [1]})
        print(results)

    # remove host results
    results = results[results['version'] != 'host']

    # print(host_results)
    # print(results)
    ax.axhline(host_results['time'][0], label="Host", linestyle="--")

    # for more: https://matplotlib.org/3.2.2/api/markers_api.html
    markers = ['o', 'v', 's', 'p', '*', 'D', 'x']

    # all versions to plot, in the order they appear in the file?
    versions = results['version'].unique()
    for idx, version in enumerate(versions):
        # draw a line for each version
        subset = results[results['version'] == version]

        # not a great idea to re-use markers, usually better to reduce your lines
        marker = markers[idx % len(markers)]
        ax.plot(subset['parallelism'], subset['time'], label=version, marker=marker)

    # set up legend
    ncol = kwargs.get('ncol', 1)
    ax.legend(bbox_to_anchor=(1, 1.04), ncol=ncol, loc='upper left', fontsize=legend_fontsize)

    # grids
    ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.xaxis.grid(color='gray', linestyle='dashed')

    # configure ticks to be what is in the columns
    xvals = results['parallelism'].unique()
    loc = plticker.FixedLocator(xvals)
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(plticker.AutoLocator())

    if 'xlab' in kwargs:
        ax.set_xlabel(kwargs['xlab'], fontsize=fontsize)
    if 'ylab' in kwargs:
        ax.set_ylabel(kwargs['ylab'], fontsize=fontsize)

    print(f"writing file to {filename}")
    plt.savefig(filename, bbox_inches='tight', dpi=300)


parser = argparse.ArgumentParser()
parser.add_argument('--results_csv', '-r', help="results csv file", required=True)
parser.add_argument('--output',  '-o', help="output file name", required=True)
parser.add_argument('--speedup',  '-s', help="normalize to host time and plot speedup", action="store_true")
# optional args for labels, etc
parser.add_argument('--xlab')
parser.add_argument('--ylab')
# legend stuff
parser.add_argument('--ncol')
args = parser.parse_args()
print(args)

results = read_csv(args.results_csv)
# a rather circuitious way to avoid arguments in the args from being None
kwargs = dict(filter(lambda x: x[1] is not None, vars(args).items()))

plot_results(results, args.output, **kwargs)
