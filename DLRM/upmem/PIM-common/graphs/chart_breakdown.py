#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import argparse

# defaults for nice publication ready rendering
fontsize = 12
legend_fontsize = 10

# configure the max width of each bar
group_width_total = 0.9
bar_spacing = 0.05  # spacing between bars

# color blind colors - using hatching instead
# plt.style.use('tableau-colorblind10')

matplotlib.rc('hatch', linewidth=0.5)
hatches = ["////", "\\\\\\\\\\", "xxxxxx", "-----", "||||||", "+++++", "//", "*", "+", "O", "o", "-"]

def read_csv(filename):
    df = pd.read_csv(filename)
    return df


def plot_results(results, filename, **kwargs):
    # 6.8 inch high figure, 2.5 inch across (matches column width)
    fig, ax = plt.subplots(figsize=(6.8, 2.5))

    # Pivoting makes it easier to render a grouped and stacked barchart with shared labels
    pivot = pd.pivot_table(data=results, index=['x_group', 'x_subgroup'], columns='y_name', values='y_value', dropna=False, fill_value=0)

    # find all groups and count subgroups
    x_groups = results['x_group'].unique()
    y_names = results['y_name'].unique()
    x_subgroups = pivot.index.get_level_values(1).unique().values

    bar_width = group_width_total / len(x_subgroups)
    
    total_indexes = []
    # computes the centre point for each set of indexes
    subgroup_slots = (bar_width * np.arange(len(x_subgroups)))
    offset = (max(subgroup_slots) - min(subgroup_slots)) / 2
    subgroup_slots = subgroup_slots - offset

    bar_width -= (bar_spacing / 2)  # trim some off bar width to get spacing
    for i in np.arange(len(x_groups)):
        total_indexes.extend(i + subgroup_slots)

    # compute placement of each bar
    np.arange(len(x_groups))

    # we draw all groups at once and stack bars ontop of each other
    prev_bottom = [0] * len(total_indexes)
    for idx, y_name in enumerate(y_names):
        hatch = hatches[idx % len(hatches)]
        # Note to change the order, move first apperance of y_name in the file
        ax.bar(total_indexes, pivot[y_name], width=bar_width, label=y_name, bottom=prev_bottom, hatch=hatch)

        prev_bottom = prev_bottom + pivot[y_name]

    ncol = kwargs.get('ncol', 1)
    # reverse the order of the legend so it lines up with the bars
    handles, labels = ax.get_legend_handles_labels()
    handles = handles[::-1]
    labels = labels[::-1]
    ax.legend(handles, labels, bbox_to_anchor=(1, 1.04), ncol=ncol, loc='upper left', fontsize=legend_fontsize)

    # adjust y-max to ensure everything is spaced nicely,
    # the autolocator appears to give us a nice upper bound
    ax.yaxis.set_major_locator(plticker.AutoLocator())
    ticks = plticker.AutoLocator().tick_values(0, max(prev_bottom))
    ax.set_ylim(min(ticks), max(ticks))

    # label the x-axis with the groups and subgroups
    ax.set_xticks(np.arange(len(x_groups)))
    ax.set_xticklabels(x_groups)
    # set minor ticks, avoid if we only have one subgroup
    if len(x_subgroups) > 1:
        ax.xaxis.remove_overlapping_locs = False
        minor_ticks = x_subgroups.tolist() * len(x_groups)
        ax.set_xticks(total_indexes, minor=True)
        ax.set_xticklabels(minor_ticks, minor=True)
        # adjust position of minor and major
        ax.tick_params(axis='x', which='minor', length=0)
        ax.tick_params(axis='x', which='major', length=10, width=0)

    # add grid 
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray', linestyle='dashed')

    if 'xlab' in kwargs:
        ax.set_xlabel(kwargs['xlab'], fontsize=fontsize)
    if 'ylab' in kwargs:
        ax.set_ylabel(kwargs['ylab'], fontsize=fontsize)

    print(f"writing file to {filename}")
    plt.savefig(filename, bbox_inches='tight', dpi=300)


parser = argparse.ArgumentParser()
parser.add_argument('--results_csv', '-r', help="results csv file", required=True)
parser.add_argument('--output',  '-o', help="output file name", required=True)
# optional args for labels, etc
parser.add_argument('--xlab')
parser.add_argument('--ylab')
# # legend stuff
parser.add_argument('--ncol')
args = parser.parse_args()

results = read_csv(args.results_csv)
# a rather circuitious way to avoid arguments in the args from being None
kwargs = dict(filter(lambda x: x[1] is not None, vars(args).items()))
plot_results(results, args.output, **kwargs)
