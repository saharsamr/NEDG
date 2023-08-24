import pickle
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from tqdm import tqdm

from data_analysis.config import *
from GNED.config import *
from data_analysis.data_plots import plot_properties_in_CPE_CME_CSME, plot_metric_kde, \
    plot_metric_differences_kde, metric_difference_box_plot, models_box_plot
from data_analysis.utils import compute_correlation
from GNED.data_handler.wiki_dataset import WikiDataset
from transformers import BartTokenizerFast


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

    df['popularity'] = df['title'].apply(lambda x: popularity[x])
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

        if len(entity_start_token_indices) != 1 or len(entity_end_token_indices) != 1:
            return -1
        else:
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
    # df = add_popularity(df)
    # df = add_popularity_log(df)

    tokenizer = BartTokenizerFast.from_pretrained(
        MODEL_GENERATION_NAME, model_max_length=INPUT_GENERATION_MAX_LENGTH, padding=True, truncation=True,
    )
    tokenizer.add_special_tokens({'additional_special_tokens': ADDITIONAL_SPECIAL_TOKENS})
    df = add_entity_token_count(df, tokenizer)

    return df


def plot_properties_cpe_vs_cme_vs_csme(df):

    for metric in ['bleu', 'rouge']:
        plot_properties_in_CPE_CME_CSME(df, 'description_length', metric, title='short-context')
        plot_properties_in_CPE_CME_CSME(df, 'description_context_overlap_ratio', metric, title='short-context')
        plot_properties_in_CPE_CME_CSME(df, 'popularity_log', metric, title='short-context')
        plot_properties_in_CPE_CME_CSME(df, 'entity_token_count', metric, title='entity token count')


def properties_correlation_cpe_vs_cme_vs_csme(df):

    for metric in ['bleu', 'rouge']:
        for model in ['CSME']:
            for property in [
                'description_length', 'description_context_overlap_ratio', 'entity_token_count'
            ]:
                print(
                    f'{metric}: {model} vs {property}: '
                    f''f'{compute_correlation(df[f"{model}-{metric}"], df[f"{property}"])}'
                )


def metrics_mean_std(df):

    for metric in ['bleu', 'rouge']:
        for model in ['CSME']:
            print(f'{model}-{metric}:', df[f'{model}-{metric}'].mean(), df[f'{model}-{metric}'].std())


def metrics_ttest(df):

    for metric in ['bert', 'bleu', 'rouge']:
        for (model1, model2) in [['CPE', 'CME'], ['CPE', 'CSME'], ['CME', 'CSME']]:
            stat, p_value = ttest_ind(df[f'{model1}-{metric}'], df[f'{model2}-{metric}'])
            print(f"t-test, {model1}-{metric} vs. {model2}-{metric}: statistic={stat:.4f}, p-value={p_value}")


def plot_kde(df):

    for metric in ['bert', 'bleu', 'rouge']:
        plot_metric_kde(df, metric)


def plot_popularity_metric_differences(popular_df, unpopular_df):

    for metric in ['bert', 'bleu', 'rouge']:
        for (model1, model2) in [['CPE', 'CME'], ['CPE', 'CSME'], ['CME', 'CSME']]:
            plot_metric_differences_kde(
                popular_df[f'{model1}-{metric}'] - popular_df[f'{model2}-{metric}'],
                unpopular_df[f'{model1}-{metric}'] - unpopular_df[f'{model2}-{metric}'], metric, model1, model2)


def popularity_models_boxplots(popular_df, unpopular_df):

    metric_names, difference, value, popularity, model_names = [], [], [], [], []
    for metric in ['bert', 'bleu', 'rouge']:
        for model in ['CPE', 'CME', 'CSME']:
            for popularity_status, df in zip(['Popular', 'Unpopular'], [popular_df, unpopular_df]):
                metric_value = df[f'{model}-{metric}']
                value.extend(metric_value.values)
                metric_names.extend([metric for _ in metric_value])
                popularity.extend([popularity_status for _ in metric_value])
                model_names.extend([model for _ in metric_value])

    df = pd.DataFrame({'metric': metric_names, 'value': value, 'popularity': popularity, 'Model': model_names})
    models_box_plot(df[df['popularity'] == 'popular'], 'Popular')
    models_box_plot(df[df['popularity'] == 'unpopular'], 'Unpopular')


def popularity_metric_differences_boxplot(popular_df, unpopular_df):

    for (model1, model2) in [['CPE', 'CME'], ['CPE', 'CSME'], ['CME', 'CSME']]:
        metric_name, difference, popularity = [], [], []
        for metric in ['bert', 'bleu', 'rouge']:
            popular_diff = popular_df[f'{model1}-{metric}'] - popular_df[f'{model2}-{metric}']
            unpopular_diff = unpopular_df[f'{model1}-{metric}'] - unpopular_df[f'{model2}-{metric}']
            print('popular: ', popular_diff.mean(), popular_diff.std())
            print('unpopular: ', unpopular_diff.mean(), unpopular_diff.std())
            stat, p_value = ttest_ind(popular_df[f'{model1}-{metric}'] - popular_df[f'{model2}-{metric}'],
                                      unpopular_df[f'{model1}-{metric}'] - unpopular_df[f'{model2}-{metric}'])
            print(f"{model1}-{model2}-{metric} t-test: statistic={stat:.4f}, p-value={p_value}")
            metric_name.extend([metric for _ in popular_diff])
            metric_name.extend([metric for _ in unpopular_diff])
            difference.extend(popular_diff.values)
            popularity.extend(['Popular' for _ in popular_diff])
            difference.extend(unpopular_diff.values)
            popularity.extend(['Unpopular' for _ in unpopular_diff])

        df = pd.DataFrame({
            'metric': metric_name,
            'difference': difference,
            'Popularity': popularity
        })
        metric_difference_box_plot(df, model1,  model2)


def popularity_plot_models_properties(popular_df, unpopular_df):

    for metric in ['bert', 'bleu', 'rouge']:
        plot_properties_in_CPE_CME_CSME(popular_df, 'popularity_log', metric, title='popular-short-context')
        plot_properties_in_CPE_CME_CSME(unpopular_df, 'popularity_log', metric, title='unpopular-short-context')
        plot_metric_kde(popular_df, metric, title='popular')
        plot_metric_kde(unpopular_df, metric, title='popular')


def popularity_mean_std(popular_df, unpopular_df):

    for metric in ['bert', 'bleu', 'rouge']:
        for model in ['CPE', 'CME', 'CSME']:
            print(
                f'popular-{model}-{metric}:',
                popular_df[f'{model}-{metric}'].mean(), popular_df[f'{model}-{metric}'].std()
            )
            print(
                f'unpopular-{model}-{metric}:',
                unpopular_df[f'{model}-{metric}'].mean(), unpopular_df[f'{model}-{metric}'].std()
            )


def popularity_models_metrics_ttest(popular_df, unpopular_df):

    for metric in ['bert', 'bleu', 'rouge']:
        for (model1, model2) in [['CPE', 'CME'], ['CPE', 'CSME'], ['CME', 'CSME']]:

            stat, p_value = ttest_ind(popular_df[f'{model1}-{metric}'], popular_df[f'{model2}-{metric}'])
            print(f"popular-{model1}-{model2}-{metric} t-test: statistic={stat:.4f}, p-value={p_value}")
            stat, p_value = ttest_ind(unpopular_df[f'{model1}-{metric}'], unpopular_df[f'{model2}-{metric}'])
            print(f"unpopular-{model1}-{model2}-{metric} t-test: statistic={stat:.4f}, p-value={p_value}")


def popularity_analysis(df):

    decile_length = len(df) // 10
    df.sort_values(by='popularity', inplace=True, ascending=False)
    popular = df[:decile_length]
    unpopular = df[-decile_length:]

    plot_popularity_metric_differences(popular, unpopular)
    popularity_models_boxplots(popular, unpopular)
    popularity_metric_differences_boxplot(popular, unpopular)
    popularity_plot_models_properties(popular, unpopular)
    popularity_mean_std(popular, unpopular)
    popularity_models_metrics_ttest(popular, unpopular)


data = pd.read_csv(TEST_ANALYSIS_FILE, delimiter='\1')
print(len(data))
print(data)
data = add_properties(data)
print(len(data))
data = data[data['entity_token_count'] != -1]
print(len(data))
# plot_properties_cpe_vs_cme_vs_csme(data)
properties_correlation_cpe_vs_cme_vs_csme(data)
metrics_mean_std(data)
# metrics_ttest(data)
# plot_kde(data)
# popularity_analysis(data)
# data.to_csv(TEST_ANALYSIS_FILE, sep='\1', index=False)

