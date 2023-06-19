import pickle
import numpy as np
import pandas as pd

from data_analysis.config import *


def add_description_length(df):
    df['description_length'] = df['label'].apply(lambda x: len(x.split()))
    return df


def add_description_context_overlap_ratio(df):

    df['description_context_overlap_ratio'] = df.apply(
        lambda x:
        len(set(x['label'].split()) & set(x['CPE-context'].split())) / len(set(x['label'].split()))
        , axis=1)
    return df


def add_popularity(df):

    with open(ENTITY_POPULARITY_PATH, 'r') as f:
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
