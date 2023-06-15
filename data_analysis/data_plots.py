import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np


def count_words(sentence):
    words = sentence.split()
    return len(words)


def distribution_of_entity_context_number(path):
    with open(path, 'r') as f:
        number_of_contexts = []
        for line in f.readlines():
            data = json.loads(line)
            number_of_contexts.append(len(data['contexts']))
        plt.hist(number_of_contexts, bins=20)
        plt.title('Distribution of Number of Contexts per Entity')
        plt.xlabel('Number of Contexts')
        plt.ylabel('Frequency')
        plt.show()


def distribution_of_contexts_length_in_json(path):
    with open(path, 'r') as f:
        context_lengths = []
        for line in f.readlines():
            data = json.loads(line)
            for context in data['contexts']:
                context_lengths.append(count_words(context))
        plt.hist(context_lengths, bins=20)
        plt.title('Distribution of Context Lengths in JSONL File')
        plt.xlabel('Context Length')
        plt.ylabel('Frequency')
        plt.show()


def files_distribution_of_descriptions_length(path):
    with open(path, 'r') as f:
        wikipedia_description_lengths = []
        wikidata_description_lengths = []
        for line in f.readlines():
            data = json.loads(line)
            wikipedia_description_lengths.append(count_words(data['wikipedia_description']))
            wikidata_description_lengths.append(count_words(data['wikidata_description']))
        plt.hist(wikipedia_description_lengths, bins=20, alpha=0.5, label='Wikipedia')
        plt.hist(wikidata_description_lengths, bins=20, alpha=0.5, label='Wikidata')
        plt.title('Distribution of Description Lengths by Source')
        plt.xlabel('Description Length')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()


def plot_metrics(
  cpe_bertscores, cme_bertscores, cpe_bleus, cme_bleus, cpe_rouges, cme_rouges, x_label, decile, plot_name):

    plt.figure(figsize=(17, 4.5))

    plt.subplot(1, 3, 1)
    plt.plot(np.arange(10, 110, 10), cme_bertscores, 'o-', label='CME')
    plt.plot(np.arange(10, 110, 10), cpe_bertscores, 'o-', label='CPE')
    plt.xticks(np.arange(10, 110, 10),
               ['1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th'] if decile else None)
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel('Bertscore')

    plt.subplot(1, 3, 2)
    plt.plot(np.arange(10, 110, 10), cme_bleus, 'o-', label='CME')
    plt.plot(np.arange(10, 110, 10), cpe_bleus, 'o-', label='CPE')
    plt.xticks(np.arange(10, 110, 10),
               ['1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th'] if decile else None)
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel('BLEU')

    plt.subplot(1, 3, 3)
    plt.plot(np.arange(10, 110, 10), cme_rouges, 'o-', label='CME')
    plt.plot(np.arange(10, 110, 10), cpe_rouges, 'o-', label='CPE')
    plt.xticks(np.arange(10, 110, 10),
               ['1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th'] if decile else None)
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel('ROUGE')

    plt.savefig(f'{plot_name}{"-decile" if decile else ""}.svg')


def plot_correlation(first, second, x_label, y_label):
    plt.scatter(first, second)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(y_label)


def violin_plot_for_popularity(most_popular_df_path, second_most_popular_df_path, delimiter='\1'):

    def add_dataframes_vertically(df, popularity_level, plot_df):

        popularity = [popularity_level] * len(most_popular_df)

        bert_diff = df['CPE_bert'] - df['CME_bert']
        plot_df = pd.concat([plot_df, pd.DataFrame(
            {'Popularity': popularity, 'Metric Difference': bert_diff,
             'Metric': ['BertScore'] * len(most_popular_df)})])

        bleu_diff = most_popular_df['CPE_bleu'] - most_popular_df['CME_bleu']
        plot_df = pd.concat([plot_df, pd.DataFrame(
            {'Popularity': popularity, 'Metric Difference': bleu_diff,
             'Metric': ['BLEU'] * len(most_popular_df)})])

        rouge_diff = most_popular_df['CPE_rouge'] - most_popular_df['CME_rouge']
        plot_df = pd.concat([plot_df, pd.DataFrame(
            {'Popularity': popularity, 'Metric Difference': rouge_diff,
             'Metric': ['ROUGE'] * len(most_popular_df)})])

        return plot_df

    most_popular_df = pd.read_csv(most_popular_df_path, delimiter=delimiter)
    second_most_popular_df = pd.read_csv(second_most_popular_df_path, delimiter=delimiter)

    plot_df = pd.DataFrame(columns=['popularity', 'score-difference', 'score-type'])

    plot_df = add_dataframes_vertically(most_popular_df, 'Most Popular Entity', plot_df)
    plot_df = add_dataframes_vertically(second_most_popular_df, 'Second Most Popular Entity', plot_df)

    sns.violinplot(data=plot_df, x='Metric', y='Metric Difference', hue='Popularity', split=True)
    plt.savefig('violin_plot_for_popularity.svg')


