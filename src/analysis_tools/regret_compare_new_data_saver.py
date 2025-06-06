import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
from os.path import dirname,join
import csv
import __main__
import json

import pandas as pd
import seaborn as sns

result_to_dump = 'bird/to_compare_new_10'
SLIDES = True
sns.set(style="darkgrid")

def load_data():
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__main__.__file__)))
    results_dir = join(join(dirname(dirname(dirname(__file__))), 'result'), result_to_dump)
    dirs = [dir for dir in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, dir))]
    i = 2
    dfs = []
    for result_dir in sorted(dirs):
        file_dir = join(join(join(root_dir, 'result'), result_to_dump), result_dir)
        dfs.append(pd.read_csv(os.path.join(file_dir, 'data/data.csv')))

    return dfs
def plot_regret():
    for x_axis in ['iteration']:  # , 'interactions', 'dist'
        for y_axis in ['regret']:
            if SLIDES:
                fig, ax = plt.subplots(figsize=(3, 2))  # figsize=(3,2)
            else:
                fig, ax = plt.subplots()
            use_log_scale = True
            dfs = load_data()
            df = pd.concat(dfs, axis=0)
            sns.lineplot(data=df, x=x_axis, y=y_axis, hue='alg', err_kws={'alpha': 0.1})

            if SLIDES:
                handles, labels = ax.get_legend_handles_labels()
                ax.get_legend().remove()

            plt.ylabel(y_axis + ' regret')
            # plt.legend(legends)
            plt.grid(b=True, which='major', color='grey', linestyle='-', alpha=0.6)
            plt.grid(b=True, which='minor', color='grey', linestyle='--', alpha=0.3)
            plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            plt.tight_layout()
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__main__.__file__)))
            objective = result_to_dump
            y_axis = 'tight' + y_axis if SLIDES else y_axis
            if use_log_scale:
                plt.yscale('log')
                plt.savefig(root_dir + '/result/' + objective + '/' + y_axis + '_regret_log_' + x_axis + '.pdf',
                            bbox_inches='tight')
                plt.savefig(root_dir + '/result/' + objective + '/' + y_axis + '_regret_log_' + x_axis + '.png',
                            bbox_inches='tight')
            else:
                plt.savefig(root_dir + '/result/' + objective + '/' + y_axis + '_regret_' + x_axis + '.pdf',
                            bbox_inches='tight')
                plt.savefig(root_dir + '/result/' + objective + '/' + y_axis + '_regret_' + x_axis + '.png',
                            bbox_inches='tight')

            if SLIDES:
                legfig, legax = plt.subplots(figsize=(7.5, 0.75))
                legax.set_facecolor('white')
                leg = legax.legend(handles, labels, loc='center', ncol=len(labels) / 2, handlelength=1.5,
                                   mode="expand", borderaxespad=0., prop={'size': 13})
                legax.xaxis.set_visible(False)
                legax.yaxis.set_visible(False)
                for line in leg.get_lines():
                    line.set_linewidth(4.0)
                plt.tight_layout(pad=0.5)
                plt.savefig(root_dir + '/result/' + objective + '/' + 'legend.pdf',
                            bbox_inches='tight')

# def plot_heatmap():


if __name__ == '__main__':
    plot_regret()