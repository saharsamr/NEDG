import json
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
from statsmodels.stats.weightstats import ztest
from tqdm import tqdm

from data_analysis.config import *
from GNED.config import INPUT_GENERATION_MAX_LENGTH, ADDITIONAL_SPECIAL_TOKENS
from data_analysis.utils import compute_correlation
from GNED.data_handler.wiki_dataset import WikiDataset
from transformers import BartTokenizerFast, T5TokenizerFast


def find_entity_popularity():

    title_to_popularity = defaultdict(int)

    with open(MAIN_JSONL_PATH, 'r') as f:
        for line in tqdm(f.readlines()):
            data = json.loads(line)
            title_to_popularity[data['wikipedia_title']] = len(data['contexts'])

    with open(ENTITY_POPULARITY_PATH, 'wb') as file:
        pickle.dump(title_to_popularity, file)


def add_description_length(df):
    df['description_length'] = df['label'].apply(lambda x: len(x.split()))
    return df


def add_description_context_overlap_ratio(df):
    df['description_context_overlap_ratio'] = df.apply(
        lambda x:
        len(set(x['label'].split()) & set(x['context'].split())) / len(set(x['label'].split())) if len(
            set(x['label'].split())) != 0 else 0
        , axis=1)
    return df


def add_popularity(df):
    with open(ENTITY_POPULARITY_PATH, 'rb') as f:
        popularity = pickle.load(f)

    df['popularity'] = df['entity_name'].apply(lambda x: popularity[x])
    return df


def add_popularity_log(df):
    df['popularity_log'] = df['popularity'].apply(lambda x: np.log(x))
    return df


def add_entity_token_count(df, tokenizer):
    def count_entity_tokens(context_tok_ids, tokenizer):

        entity_start_token_id = tokenizer.convert_tokens_to_ids('<NE>')
        entity_end_token_id = tokenizer.convert_tokens_to_ids('</NE>')

        entity_start_token_indices = [i for i, tok_id in enumerate(context_tok_ids) if tok_id == entity_start_token_id]
        entity_end_token_indices = [i for i, tok_id in enumerate(context_tok_ids) if tok_id == entity_end_token_id]

        if len(entity_start_token_indices) == 0 or len(entity_end_token_indices) == 0:
            print('Entity token not found in context.')
            return -1

        if len(entity_start_token_indices) > 1 or len(entity_end_token_indices) > 1:
            return np.mean([e - s - 1 for e, s in zip(entity_end_token_indices, entity_start_token_indices)])

        return entity_end_token_indices[0] - entity_start_token_indices[0] - 1

    dataset = WikiDataset(tokenizer, list(df['context']), list(df['label']), mask_entity=False)
    entity_token_counts = [
        count_entity_tokens(sample['input_ids'], tokenizer) for sample in tqdm(dataset)]

    df['entity_token_count'] = entity_token_counts
    return df


def add_properties(df):

    df = df.replace({'<pad>': ''}, regex=True)
    df = add_description_length(df)
    df = add_description_context_overlap_ratio(df)
    df = add_popularity(df)
    df = add_popularity_log(df)

    if 'bart' in MODEL_GENERATION_NAME:
        tokenizer = BartTokenizerFast.from_pretrained(
            MODEL_GENERATION_NAME, model_max_length=INPUT_GENERATION_MAX_LENGTH, padding=True, truncation=True,
        )
    elif 't5' in MODEL_GENERATION_NAME:
        tokenizer = T5TokenizerFast.from_pretrained(
            MODEL_GENERATION_NAME, model_max_length=INPUT_GENERATION_MAX_LENGTH, padding=True, truncation=True,
        )
    else:
        raise ValueError('Model name is not valid.')

    tokenizer.add_special_tokens({'additional_special_tokens': ADDITIONAL_SPECIAL_TOKENS})
    df = add_entity_token_count(df, tokenizer)

    return df


def properties_correlation(df, model_name, metric='rouge'):

    for property in [
        'description_length', 'description_context_overlap_ratio', 'entity_token_count',
        'popularity', 'popularity_log'
    ]:
        print(
            f'{metric}: {model_name} vs {property}: '
            f''f'{compute_correlation(df[metric], df[property])}'
        )


def metrics_mean_std(df, model_name):

    for metric in ['bleu', 'rouge', 'bert-score']:
        print(f'{model_name}-{metric}:', df[metric].mean(), df[metric].std())


def popularity_mean_std(popular_df, unpopular_df, model_name):

    for metric in ['bert-score', 'bleu', 'rouge']:
        print(
            f'popular-{model_name}-{metric}:',
            popular_df[metric].mean(), popular_df[metric].std()
        )
        print(
            f'unpopular-{model_name}-{metric}:',
            unpopular_df[metric].mean(), unpopular_df[metric].std()
        )


def popularity_metrics_ztest(popular_df, unpopular_df, model_name):

    for metric in ['bert-score', 'bleu', 'rouge']:

        stat, p_value = ztest(popular_df[metric], unpopular_df[metric])
        print(f"popular vs. unpopular in {model_name} and {metric} z-test: statistic={stat:.4f}, p-value={p_value}")


def popularity_analysis(df):

    decile_length = len(df) // 10
    df.sort_values(by='popularity', inplace=True, ascending=False)
    popular = df[:decile_length]
    unpopular = df[-decile_length:]

    popularity_mean_std(popular, unpopular, MODEL_NAME)
    popularity_metrics_ztest(popular, unpopular, MODEL_NAME)



if __name__ == '__main__':

    find_entity_popularity()

    data = pd.read_csv(TEST_ANALYSIS_FILE, delimiter='\1')
    data['label'].fillna('', inplace=True)
    data['entity_name'].fillna('', inplace=True)
    print(len(data))

    data = add_properties(data)
    data = data[data['entity_token_count'] != -1]
    print(len(data))

    properties_correlation(data, MODEL_NAME)
    metrics_mean_std(data, MODEL_NAME)
    popularity_analysis(data)

    data.to_csv(TEST_ANALYSIS_FILE, sep='\1', index=False)
