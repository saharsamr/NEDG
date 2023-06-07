import json
from collections import defaultdict
import pickle

import matplotlib.pyplot as plt

from data_analysis.config import ENTITY_POPULARITY_PATH, CLASSIFICATION_RESULT_PATH, JSONL_PATH
from data_analysis.utils import compute_metrics, compute_correlation


def find_entity_popularity():

    title_to_popularity = defaultdict(int)

    with open(JSONL_PATH, 'r') as f:
        for line in f.readlines():
            data = json.loads(line)
            title_to_popularity[data['title']] = len(data['contexts'])

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

    plot_correlation(metrics['CPE']['popularity'], metrics['CPE']['bleu'], 'popularity', 'CPE-bertscore')
    # plot_correlation(metrics['CME']['popularity'], metrics['CME']['bleu'], 'popularity', 'CME-bertscore')
    # plot_correlation(metrics['Hybrid']['popularity'], metrics['Hybrid']['bleu'], 'popularity', 'Hybrid-bertscore')


# find_entity_popularity()
popularity_and_performance_correlation()
