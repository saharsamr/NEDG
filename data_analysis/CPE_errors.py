import pickle
import numpy as np
from scipy import stats

from data_analysis.config import CLASSIFICATION_RESULT_PATH


def compare_lowest_bertscores():

    with open(CLASSIFICATION_RESULT_PATH, 'rb') as file:
        classification_result = pickle.load(file)

    classification_result.sort_values(by='CPE-bert', inplace=True, ascending=True)
    classification_result_to_analyze = classification_result[:len(classification_result) // 5]

    print(
        f'mean of bertscores in CPE: {np.mean(classification_result_to_analyze["CPE-bert"].values)}'
        f'and in CME:{np.mean(classification_result_to_analyze["CME-bert"].values)}\n'
        f'and the t-test results:\n'
        f'{stats.ttest_ind(classification_result_to_analyze["CPE-bert"].values, classification_result_to_analyze["CME-bert"].values)}'
    )


compare_lowest_bertscores()