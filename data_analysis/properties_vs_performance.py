import pickle
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from data_analysis.config import *
from data_analysis.data_plots import plot_properties_in_CPE_CME, plot_metric_kde, \
    plot_metric_differences, metric_difference_box_plot, models_box_plot
from data_analysis.utils import compute_correlation


def add_description_length(df):
    df['description_length'] = df['label'].apply(lambda x: len(x.split()))
    return df


def add_description_context_overlap_ratio(df):

    df['description_context_overlap_ratio'] = df.apply(
        lambda x:
        len(set(x['label'].split()) & set(x['CPE-context'].split())) / len(set(x['label'].split())) if len(set(x['label'].split())) != 0 else 0
        , axis=1)
    return df


def add_popularity(df):

    with open(ENTITY_POPULARITY_PATH, 'rb') as f:
        popularity = pickle.load(f)

    df['popularity'] = df['title'].apply(lambda x: popularity[x])
    return df


def add_popularity_log(df):
    df['popularity_log'] = df['popularity'].apply(lambda x: np.log(x))
    return df


data = pd.read_csv(TEST_ANALYSIS_FILE, delimiter='\1')
print(len(data))
# data = data.replace({'<pad>': ''}, regex=True)
# data = add_description_length(data)
# data = add_description_context_overlap_ratio(data)
# data = add_popularity(data)
# data = add_popularity_log(data)
print(len(data))
print('------------------------------------')
# data.to_csv(TEST_ANALYSIS_FILE, sep='\1', index=False)
for metric in ['bert', 'bleu', 'rouge']:
    # plot_properties_in_CPE_CME(data, 'description_length', metric, title='short-context')
    # plot_properties_in_CPE_CME(data, 'description_context_overlap_ratio', metric, title='short-context')
    # plot_properties_in_CPE_CME(data, 'popularity_log', metric, title='short-context')
    # plot_metric_kde(data, metric)
    # print(f'{metric}: CPE vs Desc-length: {compute_correlation(data[f"CPE-{metric}"], data["description_length"])}')
    # print(f'{metric}:CPE vs overlap: {compute_correlation(data[f"CPE-{metric}"], data["description_context_overlap_ratio"])}')
    # print(f'{metric}:CPE vs popularity-log: {compute_correlation(data[f"CPE-{metric}"], data["popularity_log"])}')
    # print(f'{metric}:CPE vs popularity: {compute_correlation(data[f"CPE-{metric}"], data["popularity"])}')
    # print('****')
    # print(f'{metric}:CME vs Desc-length: {compute_correlation(data[f"CME-{metric}"], data["description_length"])}')
    # print(f'{metric}:CME vs overlap: {compute_correlation(data[f"CME-{metric}"], data["description_context_overlap_ratio"])}')
    # print(f'{metric}:CME vs popularity-log: {compute_correlation(data[f"CME-{metric}"], data["popularity_log"])}')
    # print(f'{metric}:CME vs popularity: {compute_correlation(data[f"CME-{metric}"], data["popularity"])}')
    # print('****')
    # print(f'{metric}:CSME vs Desc-length: {compute_correlation(data[f"CSME-{metric}"], data["description_length"])}')
    # print(f'{metric}:CSME vs overlap: {compute_correlation(data[f"CSME-{metric}"], data["description_context_overlap_ratio"])}')
    # print(f'{metric}:CSME vs popularity-log: {compute_correlation(data[f"CSME-{metric}"], data["popularity_log"])}')
    # print(f'{metric}:CSME vs popularity: {compute_correlation(data[f"CSME-{metric}"], data["popularity"])}')
    # print(f'CPE-{metric}:', data[f'CPE-{metric}'].mean(), data[f'CPE-{metric}'].std())
    # print(f'CME-{metric}:', data[f'CME-{metric}'].mean(), data[f'CME-{metric}'].std())
    # stat, p_value = ttest_ind(data[f'CPE-{metric}'], data[f'CME-{metric}'])
    # print(f"t-test: statistic={stat:.4f}, p-value={p_value}")
    print('---------------------------------')


# mean_popularity = data['popularity'].mean()
# popular = data[data['popularity'] > mean_popularity]
# unpopular = data[data['popularity'] <= mean_popularity]
# for metric in ['bert', 'bleu', 'rouge']:
#     plot_metric_differences(
#         popular[f'CPE-{metric}'] - popular[f'CME-{metric}'],
#         unpopular[f'CPE-{metric}'] - unpopular[f'CME-{metric}'], metric)

decile_length = len(data) // 10
data.sort_values(by='popularity', inplace=True, ascending=False)
popular = data[:decile_length]
unpopular = data[-decile_length:]
# for metric in ['bert', 'bleu', 'rouge']:
#     plot_metric_differences(
#         popular[f'CPE-{metric}'] - popular[f'CME-{metric}'],
#         unpopular[f'CPE-{metric}'] - unpopular[f'CME-{metric}'], metric, title='decile')
# metric_names, difference, value, popularity, model_names = [], [], [], [], []
# for metric in ['bert', 'bleu', 'rouge']:
#     for model_name in ['CPE', 'CME', 'CSME']:
#         for popularity_status, df in zip(['Popular', 'Unpopular'], [popular, unpopular]):
#             metric_value = df[f'{model_name}-{metric}']
#             value.extend(metric_value.values)
#             metric_names.extend([metric for _ in metric_value])
#             popularity.extend([popularity_status for _ in metric_value])
#             model_names.extend([model_name for _ in metric_value])
#
# df = pd.DataFrame({'metric': metric_names, 'value': value, 'popularity': popularity, 'model': model_names})
# models_box_plot(df[df['popularity'] == 'popular'], 'Popular')
# models_box_plot(df[df['popularity'] == 'unpopular'], 'Unpopular')

metric_name, difference, popularity = [], [], []
for metric in ['bert', 'bleu', 'rouge']:
    popular_diff = popular[f'CPE-{metric}'] - popular[f'CME-{metric}']
    unpopular_diff = unpopular[f'CPE-{metric}'] - unpopular[f'CME-{metric}']
    print('popular: ', popular_diff.mean(), popular_diff.std())
    print('unpopular: ', unpopular_diff.mean(), unpopular_diff.std())
    stat, p_value = ttest_ind(popular[f'CPE-{metric}'] - popular[f'CME-{metric}'], unpopular[f'CPE-{metric}'] - unpopular[f'CME-{metric}'])
    print(f"{metric} t-test: statistic={stat:.4f}, p-value={p_value}")
    metric_name.extend([metric for _ in popular_diff])
    metric_name.extend([metric for _ in unpopular_diff])
    difference.extend(popular_diff.values)
    popularity.extend(['Popular' for _ in popular_diff])
    difference.extend(unpopular_diff.values)
    popularity.extend(['Unpopular' for _ in unpopular_diff])

df = pd.DataFrame({
    'metric': metric_name,
    'difference': difference,
    'popularity': popularity
})
metric_difference_box_plot(df)

# for metric in ['bert', 'bleu', 'rouge']:
#     # plot_properties_in_CPE_CME(popular, 'popularity_log', metric, title='popular-short-context')
#     # plot_properties_in_CPE_CME(unpopular, 'popularity_log', metric, title='unpopular-short-context')
#     plot_metric_kde(popular, metric, title='popular')
#     print(f'popular-CPE-{metric}:', popular[f'CPE-{metric}'].mean(), popular[f'CPE-{metric}'].std())
#     print(f'popular-CME-{metric}:', popular[f'CME-{metric}'].mean(), popular[f'CME-{metric}'].std())
#     stat, p_value = ttest_ind(popular[f'CPE-{metric}'], popular[f'CME-{metric}'])
#     print(f"popular t-test: statistic={stat:.4f}, p-value={p_value}")
#     print('---------------------------------')
#     plot_metric_kde(unpopular, metric, title='unpopular')
#     print(f'unpopular-CPE-{metric}:', unpopular[f'CPE-{metric}'].mean(), unpopular[f'CPE-{metric}'].std())
#     print(f'unpopular-CME-{metric}:', unpopular[f'CME-{metric}'].mean(), unpopular[f'CME-{metric}'].std())
#     stat, p_value = ttest_ind(unpopular[f'CPE-{metric}'], unpopular[f'CME-{metric}'])
#     print(f"unpopular t-test: statistic={stat:.4f}, p-value={p_value}")
#     print('---------------------------------')
