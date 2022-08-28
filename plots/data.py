import pandas as pd
import matplotlib.pyplot as plt


def plot_features_len_bar_plots(data_path, feature_name, hard_limit=None):

    data = pd.read_csv(
        data_path, delimiter='\1', on_bad_lines='skip', header=0, names=['word', 'context', 'description']
    )
    data[f'{feature_name}_len'] = data[feature_name].str.split().str.len()

    if hard_limit:
        plt.hist([f for f in data[f'{feature_name}_len'] if f < hard_limit], bins=10)
    else:
        plt.hist(data[f'{feature_name}_len'], bins=10)
    plt.show()


