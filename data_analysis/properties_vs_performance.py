import pickle
import numpy as np
import pandas as pd

from data_analysis.config import *
from data_analysis.data_plots import plot_properties_in_CPE_CME


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
data = data.replace({'<pad>': ''}, regex=True)
data = add_description_length(data)
data = add_description_context_overlap_ratio(data)
data = add_popularity(data)
data = add_popularity_log(data)
data.to_csv(TEST_ANALYSIS_FILE, sep='\1', index=False)
for metric in ['bert', 'bleu', 'rouge']:
    plot_properties_in_CPE_CME(data, 'description_length', metric, xlim=(0, 20))
    plot_properties_in_CPE_CME(data, 'description_context_overlap_ratio', metric, xlim=(0, 1))
    plot_properties_in_CPE_CME(data, 'popularity_log', metric, xlim=(0, 6.5))

mean_popularity = data['popularity'].mean()
popular = data[data['popularity'] > mean_popularity]
unpopular = data[data['popularity'] <= mean_popularity]
for metric in ['bert', 'bleu', 'rouge']:
    plot_properties_in_CPE_CME(popular, 'popularity_log', metric, xlim=(0, 6.5), title='popular')
    plot_properties_in_CPE_CME(unpopular, 'popularity_log', metric, xlim=(0, 6.5), title='unpopular')
