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


def plot_results(results, filename, **kwargs):
    # 6.8 inch high figure, 2.5 inch across (matches column width)
    fig, ax = plt.subplots(figsize=(6.8, 2.5))
    """
    # pick out host results, which we assume is always just one number
    host_results = results[results['version'] == 'host']
    if len(host_results) == 0:
        # if no host results present, just assume times are relative to host
        host_results = pd.DataFrame.from_dict({'version': ['host'], 'time': [1]})

    # remove host results
    results = results[results['version'] != 'host']
    """
    host_results = pd.DataFrame.from_dict({'version': ['host'], 'time': [1]})
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
        ax.plot(subset['parallelism'], subset['time'], label=version, marker=marker, linestyle='--' if version == "host" else None)

    # set up legend
    ncol = kwargs.get('ncol', 1)
    ax.legend(bbox_to_anchor=(1, 1.04), ncol=ncol, loc='upper left', fontsize=legend_fontsize)

    # configure ticks to be what is in the columns
    xvals = results['parallelism'].unique()

    if 'xlab' in kwargs:
        ax.set_xlabel(kwargs['xlab'], fontsize=fontsize)
    if 'ylab' in kwargs:
        ax.set_ylabel(kwargs['ylab'], fontsize=fontsize)

    ax.set_xscale("log", basex=2)
    #ax.set_yscale("log", basey=2)

    loc = plticker.FixedLocator(xvals)
    #x.xaxis.set_major_locator(loc)
    ax.yaxis.grid(color='gray', linestyle='dashed')
    ax.xaxis.grid(color='gray', linestyle='dashed')
    #ax.xaxis.set_major_formatter(plticker.EngFormatter(unit=''))

    def sizeof_fmt(num, idx):
        suffix='B'
        for unit in ['','K','M','G','T','P','E','Z']:
            if abs(num) < 1024.0:
                return "%.0f%s%s" % (num, unit, suffix)
            num /= 1024.0
        return "%.0f%s%s" % (num, 'Yi', suffix)

    ax.xaxis.set_major_formatter(plticker.FuncFormatter(sizeof_fmt))

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
