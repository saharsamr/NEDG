import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
    plt.legend()
    plt.show()


