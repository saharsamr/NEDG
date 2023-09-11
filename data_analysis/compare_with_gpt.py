import pandas as pd
from nltk.tokenize import word_tokenize
from tqdm import tqdm

from data_analysis.utils import compute_bleu, compute_rouge, compute_bertscore


if __name__ == "__main__":

    cpe = pd.read_csv('data/CPE.csv', delimiter='\1', names=['context', 'label', 'pred'], header=None)
    cme = pd.read_csv('data/CME.csv', delimiter='\1', names=['context', 'label', 'pred'], header=None)
    csme = pd.read_csv('data/CSME.csv', delimiter='\1', names=['context', 'label', 'pred'], header=None)
    gpt = pd.read_csv('data/gpt-response.csv', delimiter='\1')
    gpt.drop([0, 1], inplace=True)

    csme['cpe-pred'] = cpe['pred']
    csme['cme-pred'] = cme['pred']
    csme['csme-pred'] = csme['pred']
    csme = csme[csme['context'].isin(gpt['context'])]

    final = csme
    final.reset_index(inplace=True)
    gpt.reset_index(inplace=True)
    final['gpt'] = gpt['response']
    print(len(final))

    final.to_csv('data/final-gpt.csv', sep='\1', index=False)

    descriptions = final['label']
    CSME_preds = final['csme-pred']
    GPT_preds = final['gpt']

    CSME_preds = [word_tokenize(pred) for pred in CSME_preds.values]
    GPT_preds = [word_tokenize(pred) for pred in GPT_preds.values]
    descriptions = [[word_tokenize(description)] for description in descriptions.values]

    CSME_bleu = [
        compute_bleu([csme_pred], [label], 1) for
        csme_pred, label in tqdm(zip(CSME_preds, descriptions), total=len(CSME_preds))]
    GPT_bleu = [
        compute_bleu([csme_pred], [label], 1) for
        csme_pred, label in tqdm(zip(GPT_preds, descriptions), total=len(GPT_preds))]

    CSME_rouge = [
        compute_rouge([csme_pred], [label]) for
        csme_pred, label in tqdm(zip(CSME_preds, descriptions), total=len(CSME_preds))]
    GPT_rouge = [
        compute_rouge([csme_pred], [label]) for
        csme_pred, label in tqdm(zip(GPT_preds, descriptions), total=len(GPT_preds))]

    CSME_bert = [
        compute_bertscore([csme_pred], [label])[0] for
        csme_pred, label in tqdm(zip(CSME_preds, descriptions), total=len(CSME_preds))]
    GPT_bert = [
        compute_bertscore([csme_pred], [label])[0] for
        csme_pred, label in tqdm(zip(GPT_preds, descriptions), total=len(GPT_preds))]

    final['CSME-bleu'] = CSME_bleu
    final['GPT-bleu'] = GPT_bleu
    final['CSME-rouge'] = CSME_rouge
    final['GPT-rouge'] = GPT_rouge
    final['CSME-bert'] = CSME_bert
    final['GPT-bert'] = GPT_bert

    print(f'Bleu -> CSME: {final["CSME-bleu"].mean()}, ChatGPT: {final["GPT-bleu"].mean()}')
    print(f'Rouge -> CSME: {final["CSME-rouge"].mean()}, ChatGPT: {final["GPT-rouge"].mean()}')
    print(f'BertScore -> CSME: {final["CSME-bert"].mean()}, ChatGPT: {final["GPT-bert"].mean()}')




