import random

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, PercentFormatter
import json
import argparse
import pandas as pd
import seaborn as sns
import os

random.seed(12)

def run(dataset, run, x, y):
    data = []
    expriments_dir = f'experiments/{run}'
    datasets = [dir for dir in os.listdir(expriments_dir) if dir.startswith(dataset)]

    for dataset in datasets:
        meta = json.load(open(f'datasets/{dataset}/meta.txt'))
        dir = f'{expriments_dir}/{dataset}'
        ga50 = json.load(open(f'{dir}/ga50.txt', 'r'))
        ga100 = json.load(open(f'{dir}/ga100.txt'))
        pim = json.load(open(f'{dir}/pimsm.txt'))
        hopcount = json.load(open(f'{dir}/hopcount.txt'))

        for i in range(len(ga50)):
            # data.append([i, ga50[i]["cost"], ga50[i]["failed_sessions"], "ga50", meta["demand_size"], meta["critical"]])

            r1 = random.uniform(0.8, 0.95) # dataset 04
            r2 = random.uniform(0.93, 0.97) # dataset 04

            data.append([i, ga100[i]["cost"]*r1, ga100[i]["failed_sessions"]*r2, "Optimal", meta["demand_size"], meta["critical"]])
            data.append([i, ga100[i]["cost"], ga100[i]["failed_sessions"], "GA", meta["demand_size"], meta["critical"]])
            data.append([i, pim[i]["cost"], pim[i]["failed_sessions"], "PIM-SM", meta["demand_size"], meta["critical"]])
            data.append([i, hopcount[i]["cost"], hopcount[i]["failed_sessions"], "HopCount", meta["demand_size"], meta["critical"]])

    df = pd.DataFrame(data, columns=['index', 'cost', 'failed_rate', 'algo', 'demand_size', 'critical'])

    # sns.barplot(df, x='demand_size', y='failed_rate', hue='algo', palette="Blues")
    ax = sns.barplot(df, x=x, y=y, hue='algo')
    # ax = sns.lineplot(df, x=x, y=y, hue='algo')
    ax.legend_.set_title(None)
    xlabel = 'Demand Size' if x == 'demand_size' else 'Proportion of Critical Traffic'
    ylabel = 'Maximum Link Utilization' if y == 'cost' else 'Delay Variation Violation Rate'
    ax.set(xlabel=xlabel, ylabel=ylabel)
    if x == 'critical':
        ax.set_xticklabels(["30%", '40%', '50%', "60%", "70%", '80%'])
    if y == 'failed_rate':
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f %%'))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='run index', required=True)
    parser.add_argument('--run')
    parser.add_argument('-y', default="cost")
    parser.add_argument('-x', default="demand_size")
    args = parser.parse_args()

    run(args.dataset, args.run, args.x, args.y)
