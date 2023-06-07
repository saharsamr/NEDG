import pickle
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from data_analysis.config import CLASSIFICATION_RESULT_PATH


def compare_lowest_bertscores():

    with open(CLASSIFICATION_RESULT_PATH, 'rb') as file:
        classification_result = pickle.load(file)

    cpe_bertscores, cme_bertscores = [], []
    for i in np.arange(0.1, 1.1, 0.1):

        classification_result.sort_values(by='CPE-bert', inplace=True, ascending=True)
        classification_result = classification_result[classification_result['CPE-bert'] != 0]
        classification_result = classification_result[classification_result['CME-bert'] != 0]
        classification_result_to_analyze = classification_result[:int(i*len(classification_result))]

        cpe_bert = np.mean(classification_result_to_analyze["CPE-bert"].values)
        cme_bert = np.mean(classification_result_to_analyze["CME-bert"].values)

        print(
            f'For the {i * 100} percent of the lowest CPE bertscores, we have:\n'
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
    plt.xticks(np.arange(10, 110, 10))
    plt.legend()
    plt.xlabel('Percentage of whole data selected based on lowest Bertscore on CPE')
    plt.ylabel('Bertscore')
    plt.savefig('bertscores.svg')


compare_lowest_bertscores()