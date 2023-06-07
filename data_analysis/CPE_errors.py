import pickle
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from tqdm import tqdm

from data_analysis.config import CLASSIFICATION_RESULT_PATH
from data_analysis.utils import compute_bleu, compute_rouge


def compare_lowest_bertscores(decile=False):

    with open(CLASSIFICATION_RESULT_PATH, 'rb') as file:
        classification_result = pickle.load(file)

    classification_result = classification_result[classification_result['CPE-bert'] != 0]
    classification_result = classification_result[classification_result['CME-bert'] != 0]

    cpe_preds = [word_tokenize(pred) for pred in classification_result['CPE-pred']]
    cme_preds = [word_tokenize(pred) for pred in classification_result['CME-pred']]
    labels = [[word_tokenize(label)] for label in classification_result['label']]

    classification_result['CPE-bleu'] = [
            compute_bleu([cpe_pred], [cpe_label], 1) for
            cpe_pred, cpe_label in tqdm(zip(cpe_preds, labels), total=len(cpe_preds))]
    classification_result['CME-bleu'] = [
            compute_bleu([cme_pred], [cme_label], 1) for
            cme_pred, cme_label in tqdm(zip(cme_preds, labels), total=len(cme_preds))]

    classification_result['CPE-rouge'] = [
            compute_rouge([cpe_pred], [cpe_label]) for
            cpe_pred, cpe_label in tqdm(zip(cpe_preds, labels), total=len(cpe_preds))]
    classification_result['CME-rouge'] = [
        compute_rouge([cme_pred], [cme_label]) for
        cme_pred, cme_label in tqdm(zip(cme_preds, labels), total=len(cme_preds))]

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
            classification_result_to_analyze = classification_result[int((i-1)*len(classification_result)):int(i*len(classification_result))]
            description = f'{i * 10}th 10'
        else:
            classification_result_to_analyze = classification_result[:int(i * len(classification_result))]
            description = f'{i * 100}'
        classification_result_to_analyze = classification_result_to_analyze

        cpe_bert = np.mean(classification_result_to_analyze["CPE-bert"].values)
        cme_bert = np.mean(classification_result_to_analyze["CME-bert"].values)

        cpe_bleu = np.mean(classification_result_to_analyze["CPE-bleu"].values)
        cme_bleu = np.mean(classification_result_to_analyze["CME-bleu"].values)

        cpe_rouge = np.mean(classification_result_to_analyze["CPE-rouge"].values)
        cme_rouge = np.mean(classification_result_to_analyze["CME-rouge"].values)

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

    plt.figure(figsize=(17, 4.5))

    plt.subplot(1, 3, 1)
    plt.plot(np.arange(10, 110, 10), cme_bertscores, 'o-', label='CME')
    plt.plot(np.arange(10, 110, 10), cpe_bertscores, 'o-', label='CPE')
    plt.xticks(np.arange(10, 110, 10), ['1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th'] if decile else None)
    plt.legend()
    plt.xlabel(x_axis_label)
    plt.ylabel('Bertscore')

    plt.subplot(1, 3, 2)
    plt.plot(np.arange(10, 110, 10), cme_bleus, 'o-', label='CME')
    plt.plot(np.arange(10, 110, 10), cpe_bleus, 'o-', label='CPE')
    plt.xticks(np.arange(10, 110, 10),
               ['1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th'] if decile else None)
    plt.legend()
    plt.xlabel(x_axis_label)
    plt.ylabel('BLEU')

    plt.subplot(1, 3, 3)
    plt.plot(np.arange(10, 110, 10), cme_rouges, 'o-', label='CME')
    plt.plot(np.arange(10, 110, 10), cpe_rouges, 'o-', label='CPE')
    plt.xticks(np.arange(10, 110, 10),
               ['1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th'] if decile else None)
    plt.legend()
    plt.xlabel(x_axis_label)
    plt.ylabel('ROUGE')

    # plt.tight_layout()

    plt.savefig(f'metrics{"-decile" if decile else ""}.svg')


compare_lowest_bertscores()
compare_lowest_bertscores(decile=True)