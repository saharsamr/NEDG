import pickle
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from data_analysis.config import CLASSIFICATION_RESULT_PATH


def compare_lowest_bertscores(decile=False):

    with open(CLASSIFICATION_RESULT_PATH, 'rb') as file:
        classification_result = pickle.load(file)

    cpe_bertscores, cme_bertscores = [], []
    if decile:
        description = f'{i * 10}th 10'
        x_axis_label = 'decile of Data'
    else:
        description = f'{i * 100}'
        x_axis_label = 'Percentage of Data'
    
    for i in np.arange(0.1, 1.1, 0.1):

        classification_result.sort_values(by='CPE-bert', inplace=True, ascending=True)
        classification_result = classification_result[classification_result['CPE-bert'] != 0]
        classification_result = classification_result[classification_result['CME-bert'] != 0]
        if decile:
            classification_result_to_analyze = classification_result[int((i-1)*len(classification_result)):int(i*len(classification_result))]
        else:
            classification_result_to_analyze = classification_result[:int(i * len(classification_result))]

        cpe_bert = np.mean(classification_result_to_analyze["CPE-bert"].values)
        cme_bert = np.mean(classification_result_to_analyze["CME-bert"].values)

        print(
            f'For the {description} percent of the lowest CPE bertscores, we have:\n'
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
    plt.xticks(np.arange(10, 110, 10), ['1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th'] if decile else None)
    plt.legend()
    plt.xlabel(x_axis_label)
    plt.ylabel('Bertscore')
    plt.savefig('bertscores.svg')


compare_lowest_bertscores()