import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json


def plot_features_len_histograms(data_path, feature_name, ax, label, num_bins=30, hard_limit=None, bw_method=None, ylabel=False):

    data = pd.read_csv(
        data_path, delimiter='\1', on_bad_lines='skip', header=0, names=['word', 'context', 'description']
    )
    data[f'{feature_name}_len'] = data[feature_name].str.split().str.len()

    ax.set_title(label)
    ax.set_xlim(0, hard_limit)
    ax.set_xlabel(f'{feature_name} length')
    sns.histplot(
        data=data[f'{feature_name}_len'], bins=num_bins,
        kde=True, kde_kws={'bw_method': bw_method},
        color='mediumseagreen', ax=ax
    )
    ax.lines[0].set_color("darkgreen")
    ax.lines[0].set_linewidth(2)
    ax.axvline(x=data[f'{feature_name}_len'].mean(), color='orangered', ls='--', lw=2, label='Mean')
    ax.axvline(x=data[f'{feature_name}_len'].median(), color='mediumblue', ls='--', lw=2, label='Median')
    plt.gcf().subplots_adjust(left=0.15)
    if ylabel:
        ax.set_ylabel('Num of Samples')
    else:
        ax.set_ylabel('')
    ax.legend()


def plot_histograms_in_subplots():

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plot_features_len_histograms(
        'data/AllNoConcat/train.csv', 'context', axes[0], 'Histogram of length of contexts',
        100, hard_limit=400, ylabel=True
    )
    plot_features_len_histograms(
        'data/AllNoConcat/train.csv', 'description', axes[1], 'Histogram of length of definitions',
        num_bins=40, hard_limit=40, bw_method=0.2
    )
    plt.show()


def plot_categories_barplots(data_path, ax, label, start=0, end=10, ylabel=False):

    categories = defaultdict(int)
    with open(data_path, 'r') as f:
        for line in f:
            data = list(json.loads(line).values())[0]
            for k, v in data['types'].items():
                categories[v['label']] += 1

    categories = {k: v for k, v in sorted(categories.items(), key=lambda item: item[1], reverse=True)[start:end]}
    sns.barplot(ax=ax, x=list(categories.keys()), y=list(categories.values()), color='mediumseagreen')
    plt.gcf().subplots_adjust(bottom=0.5, left=0.15)
    ax.set_title(label, fontsize=8)
    ax.tick_params('y', labelsize=8)
    if ylabel:
        ax.set_ylabel('Num of Entities')
    ax.set_xticklabels(list(categories.keys()), rotation=90, fontsize=8)


def plot_categories_barplot_in_subplots():

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plot_categories_barplots(
        'data/Jsons/train.json', axes[0], '(a) 10 first categories in whole data', ylabel=True
    )
    plot_categories_barplots(
        'data/Jsons/train.json', axes[1], '(b) 10 first categories in non-humans part of data', start=1, end=11
    )
    plot_categories_barplots(
        'data/Jsons/train_human.json', axes[2], '(c) 10 first categories in humans part of data', start=1, end=11
    )
    plt.show()
