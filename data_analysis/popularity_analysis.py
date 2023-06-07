import json
from collections import defaultdict
from tqdm import tqdm
import pickle

import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

from data_analysis.config import ENTITY_POPULARITY_PATH, CLASSIFICATION_RESULT_PATH, JSONL_PATH
from data_analysis.utils import compute_metrics, compute_correlation


def find_entity_popularity():

    title_to_popularity = defaultdict(int)

    with open(JSONL_PATH, 'r') as f:
        for line in tqdm(f.readlines()):
            data = json.loads(line)
            title_to_popularity[data['wikipedia_title']] = len(data['contexts'])

    # save the dictionary in the path
    print("Saving dictionary to file...")
    with open(ENTITY_POPULARITY_PATH, 'wb') as file:
        pickle.dump(title_to_popularity, file)

    print("Dictionary saved successfully.")


def map_metric_to_popularity(title_to_popularity, classification_result):

    for i, row in classification_result.iterrows():
        title = row['title']
        popularity = title_to_popularity[title]
        classification_result.at[i, 'popularity'] = popularity

    return classification_result


def plot_correlation(first, second, x_label, y_label):

      plt.scatter(first, second)
      plt.xlabel(x_label)
      plt.ylabel(y_label)
      plt.savefig(y_label)


def popularity_and_performance_correlation():

    with open(ENTITY_POPULARITY_PATH, 'rb') as file:
        title_to_popularity = pickle.load(file)
    with open(CLASSIFICATION_RESULT_PATH, 'rb') as file:
        classification_result = pickle.load(file)

    classification_result = map_metric_to_popularity(title_to_popularity, classification_result)
    classification_result = classification_result[classification_result['label'] != '']
    metrics = compute_metrics(classification_result)

    print(compute_correlation(metrics['CPE']['bleu'], metrics['CPE']['popularity']))
    # print(compute_correlation(metrics['CME']['bleu'], metrics['CME']['popularity']))
    # print(compute_correlation(metrics['Hybrid']['bleu'], metrics['Hybrid']['popularity']))

    plot_correlation(metrics['CPE']['popularity'], metrics['CPE']['bleu'], 'popularity', 'CPE-bleu')
    # plot_correlation(metrics['CME']['popularity'], metrics['CME']['bleu'], 'popularity', 'CME-bertscore')
    # plot_correlation(metrics['Hybrid']['popularity'], metrics['Hybrid']['bleu'], 'popularity', 'Hybrid-bertscore')


def compare_highest_popularity():

    with open(ENTITY_POPULARITY_PATH, 'rb') as file:
        title_to_popularity = pickle.load(file)
    with open(CLASSIFICATION_RESULT_PATH, 'rb') as file:
        classification_result = pickle.load(file)

    classification_result = map_metric_to_popularity(title_to_popularity, classification_result)

    cpe_bertscores, cme_bertscores = [], []
    for i in np.arange(0.1, 1.1, 0.1):

        classification_result.sort_values(by='CPE-bert', inplace=True, ascending=False)
        classification_result = classification_result[classification_result['CPE-bert'] != 0]
        classification_result = classification_result[classification_result['CME-bert'] != 0]
        classification_result_to_analyze = classification_result[int((i-0.1)*len(classification_result)):int(i*len(classification_result))]

        cpe_bert = np.mean(classification_result_to_analyze["CPE-bert"].values)
        cme_bert = np.mean(classification_result_to_analyze["CME-bert"].values)

        print(
            f'For the {int(i * 10)}th 10 percent of the most popular entities, we have:\n'
            f'mean of bertscores in CPE: {cpe_bert}\n'
            f'and in CME:{cme_bert}\n'
            f'and the t-test results:\n'
            f'{stats.ttest_ind(classification_result_to_analyze["CPE-bert"].values, classification_result_to_analyze["CME-bert"].values)}'
        )
        print('-------------------------')

        cpe_bertscores.append(cpe_bert)
        cme_bertscores.append(cme_bert)

    plt.plot(np.arange(10, 110, 10), cme_bertscores, 'o-', label='CME')
    plt.plot(np.arange(10, 110, 10), cpe_bertscores, 'o-', label='CPE')
    plt.xticks(np.arange(10, 110, 10), ['1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th'])
    plt.legend()
    plt.xlabel('Decile of data')
    plt.ylabel('Bertscore')
    plt.savefig('decile-popularity.svg')


# find_entity_popularity()
# popularity_and_performance_correlation()
compare_highest_popularity()
