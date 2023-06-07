import pickle
import numpy as np
from scipy import stats
from matplotlib.pyplot as plt

from data_analysis.config import CLASSIFICATION_RESULT_PATH


def compare_lowest_bertscores():

    with open(CLASSIFICATION_RESULT_PATH, 'rb') as file:
        classification_result = pickle.load(file)

    cpe_bertscores, cme_bertscores = [], []
    for i in range(1, 11):
        classification_result.sort_values(by='CPE-bert', inplace=True, ascending=True)
        classification_result_to_analyze = classification_result[:len(classification_result) // i]

        cpe_bert = np.mean(classification_result_to_analyze["CPE-bert"].values)
        cme_bert = np.mean(classification_result_to_analyze["CME-bert"].values)

        print(
            f'For the {(1/i) * 100} percent of the lowest CPE bertscores, we have:\n'
            f'mean of bertscores in CPE: {cpe_bert}\n'
            f'and in CME:{cme_bert}\n'
            f'and the t-test results:\n'
            f'{stats.ttest_ind(classification_result_to_analyze["CPE-bert"].values, classification_result_to_analyze["CME-bert"].values)}'
        )

        cpe_bertscores.append(cpe_bert)
        cme_bertscores.append(cme_bert)

    plt.plot(cpe_bertscores, label='CPE')
    plt.plot(cme_bertscores, label='CME')
    plt.legend()
    plt.xlabel('Percentage of lowest CPE bertscores')
    plt.ylabel('Mean of bertscores')
    plt.savefig('bertscores.png')


compare_lowest_bertscores()