import pickle
import numpy as np
from scipy import stats

from data_analysis.config import CLASSIFICATION_RESULT_PATH
from data_analysis.utils import add_bleu_rouge_to_df, remove_empty_preds, compute_metrics_for_every_fraction
from data_analysis.data_plots import plot_metrics


def compare_lowest_bertscores(decile=False):

    with open(CLASSIFICATION_RESULT_PATH, 'rb') as file:
        classification_result = pickle.load(file)

    classification_result = remove_empty_preds(classification_result)
    classification_result = add_bleu_rouge_to_df(classification_result)

    cpe_bertscores, cme_bertscores = [], []
    cpe_bleus, cme_bleus = [], []
    cpe_rouges, cme_rouges = [], []
    if decile:
        x_axis_label = 'decile of Data'
    else:
        x_axis_label = 'Percentage of Data'

    for i in np.arange(0.1, 1.1, 0.1):

        classification_result.sort_values(by='CPE-bert', inplace=True, ascending=True)
        if decile:
            classification_result_to_analyze = classification_result[int((i-0.1)*len(classification_result)):int(i*len(classification_result))]
            description = f'{i * 10}th 10'
        else:
            classification_result_to_analyze = classification_result[:int(i * len(classification_result))]
            description = f'{i * 100}'
        classification_result_to_analyze = classification_result_to_analyze

        cpe_bert, cme_bert, cpe_bleu, cme_bleu, cpe_rouge, cme_rouge = \
            compute_metrics_for_every_fraction(classification_result_to_analyze)

        print(
            f'For the {description} percent of the lowest CPE bertscores, we have:\n'
            f'mean of bertscores in CPE: {cpe_bert}\n'
            f'and in CME:{cme_bert}\n'
            f'and the t-test results:\n'
            f'{stats.ttest_ind(classification_result_to_analyze["CPE-bert"].values, classification_result_to_analyze["CME-bert"].values)}'
            f'mean of bleu in CPE: {cpe_bleu}\n'
            f'and in CME:{cme_bleu}\n'
            f'and the t-test results:\n'
            f'{stats.ttest_ind(cpe_bleu, cme_bleu)}'
            f'mean of rouge in CPE: {cpe_rouge}\n'
            f'and in CME:{cme_rouge}\n'
            f'and the t-test results:\n'
            f'{stats.ttest_ind(cpe_rouge, cme_rouge)}'
        )
        print('-------------------------')

        cpe_bertscores.append(cpe_bert)
        cme_bertscores.append(cme_bert)
        cpe_bleus.append(cpe_bleu)
        cme_bleus.append(cme_bleu)
        cpe_rouges.append(cpe_rouge)
        cme_rouges.append(cme_rouge)

    plot_metrics(
        cpe_bertscores, cme_bertscores, cpe_bleus, cme_bleus, cpe_rouges, cme_rouges, x_axis_label, decile, 'bertscore')


compare_lowest_bertscores()
compare_lowest_bertscores(decile=True)
