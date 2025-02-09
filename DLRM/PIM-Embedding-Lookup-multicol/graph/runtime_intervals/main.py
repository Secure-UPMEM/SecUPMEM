import matplotlib.pyplot as plt
from matplotlib import collections as mc
import pandas as pd

file = "runtimes.csv"
colors = "bgrcmykwbgrcmykw"

def main():
    data = pd.read_csv(file)

    lines = []
    for l in data.iterrows():
        vals = l[1]
        lines.append(
            [(vals[1], vals[0]), (vals[2], vals[0])]
        )

    c = [colors[int(v[0][1])] for v in lines]

    lc = mc.LineCollection(lines, colors=c, linewidths=5)
    fig, ax = plt.subplots()
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)

    # plt.show()

    plt.savefig("out.png")

if __name__ == "__main__":
    main()
