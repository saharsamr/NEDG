import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import json


def plot_features_len_bar_plots(data_path, feature_name, num_bins=30, hard_limit=None, bw_method=None):

    data = pd.read_csv(
        data_path, delimiter='\1', on_bad_lines='skip', header=0, names=['word', 'context', 'description']
    )
    data[f'{feature_name}_len'] = data[feature_name].str.split().str.len()

    plt.xlim(0, hard_limit)
    plt.xlabel(f'{feature_name} length')
    plt.ylabel('Num of Samples')
    ax = sns.histplot(
        data=data[f'{feature_name}_len'], bins=num_bins,
        kde=True, kde_kws={'bw_method': bw_method},
        color='mediumseagreen'
    )
    ax.lines[0].set_color("darkgreen")
    ax.lines[0].set_linewidth(2)
    plt.axvline(x=data[f'{feature_name}_len'].mean(), color='orangered', ls='--', lw=2, label='Mean')
    plt.axvline(x=data[f'{feature_name}_len'].median(), color='mediumblue', ls='--', lw=2, label='Median')
    plt.gcf().subplots_adjust(left=0.15)
    plt.legend()
    plt.show()


def plot_categories_bar_plots(data_path):

    categories = defaultdict(int)
    with open(data_path, 'r') as f:
        for line in f:
            data = list(json.loads(line).values())[0]
            for k, v in data['types'].items():
                categories[v['label']] += 1

    plt.figure(figsize=(8, 6))
    categories = {k: v for k, v in sorted(categories.items(), key=lambda item: item[1], reverse=True)[:20]}
    sns.barplot(x=list(categories.keys()), y=list(categories.values()), color='mediumseagreen')
    plt.xticks(rotation=90, fontsize=5)
    plt.yticks(fontsize=5)
    plt.xlabel('Category')
    plt.ylabel('Num of Entities')
    plt.gcf().subplots_adjust(bottom=0.3)
    plt.show()
