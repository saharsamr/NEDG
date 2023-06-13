import pandas as pd
from transformers import TrainingArguments
from datasets import load_metric
import csv

from GNED.models.BART import BART
from make_datasets.config import *

import os
dirname = os.path.dirname(__file__)


def make_classification_dataset(CPE_model_name, CME_model_name, input_file, output_file, split, delimiter='\1'):

    input_data = pd.read_csv(input_file, delimiter=delimiter).sample(frac=0.2, random_state=42).dropna()
    input_x, input_y = list(input_data['contexts']), list(input_data['entity_description'])

    training_args = TrainingArguments(
        output_dir=f'{dirname}/../results',
        logging_dir=LOGGING_DIR,
        logging_strategy='steps',
        logging_steps=100,
    )
    CPE_model = BART(
        training_args,
        input_x, input_y, input_x, input_y, input_x, input_y,
        load=True, model_load_path=CPE_model_name,
        model_name=CPE_model_name, mask_entity=False
    )
    CME_model = BART(
        training_args,
        input_x, input_y, input_x, input_y, input_x, input_y,
        load=True, model_load_path=CME_model_name,
        model_name=CME_model_name, mask_entity=True
    )

    CPE_preds, CPE_inputs, CPE_labels = CPE_model.pred()
    CME_preds, CME_inputs, CME_labels = CME_model.pred()

    bertscore = load_metric('bertscore')
    CPE_bert = bertscore.compute(
        predictions=CPE_preds, references=CPE_labels, lang='en', model_type='bert-large-uncased')['f1']
    CME_bert = bertscore.compute(
        predictions=CME_preds, references=CME_labels, lang='en', model_type='bert-large-uncased')['f1']
    classification_labels = [1 if cpe >= cme else 0 for cpe, cme in zip(CPE_bert, CME_bert)]

    with open(output_file, 'w') as f, \
      open(f'{DATA_PATH}/{split}_classification_t1.csv', 'w') as f1, \
      open(f'{DATA_PATH}/{split}_classification_t2.csv', 'w') as f2, \
      open(f'{DATA_PATH}/{split}_classification_t3.csv', 'w') as f3, \
      open(f'{DATA_PATH}/{split}_classification_t4.csv', 'w') as f4, \
      open(f'{DATA_PATH}/{split}_classification_t5.csv', 'w') as f5:

        columns = ['label', 'title', 'CPE-context', 'CPE-pred', 'CPE-bert',
                   'CME-context', 'CME-pred', 'CME-bert', 'class-label']

        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow(columns)

        writer1 = csv.writer(f1, delimiter=delimiter)
        writer1.writerow(columns)
        writer2 = csv.writer(f2, delimiter=delimiter)
        writer2.writerow(columns)
        writer3 = csv.writer(f3, delimiter=delimiter)
        writer3.writerow(columns)
        writer4 = csv.writer(f4, delimiter=delimiter)
        writer4.writerow(columns)
        writer5 = csv.writer(f5, delimiter=delimiter)
        writer5.writerow(columns)

        for label, title, cpe_context, cpe_pred, cpe_bert, cme_context, cme_pred, cme_bert, class_label in \
            zip(input_y, list(input_data['entity_name']), CPE_inputs, CPE_preds,
                CPE_bert, CME_inputs, CME_preds, CME_bert, classification_labels):

            row = [label.replace(delimiter, ''), title.replace(delimiter, ''), cpe_context.replace(delimiter, ''),
                   cpe_pred.replace(delimiter, ''), cpe_bert, cme_context.replace(delimiter, ''),
                   cme_pred.replace(delimiter, ''), cme_bert, class_label]
            writer.writerow(row)

            if abs(cpe_bert - cme_bert) >= 0.1:
                writer1.writerow(row)
            if abs(cpe_bert - cme_bert) >= 0.2:
                writer2.writerow(row)
            if abs(cpe_bert - cme_bert) >= 0.3:
                writer3.writerow(row)
            if abs(cpe_bert - cme_bert) >= 0.4:
                writer4.writerow(row)
            if abs(cpe_bert - cme_bert) >= 0.5:
                writer5.writerow(row)


make_classification_dataset(CPE_MODEL_NAME, CME_MODEL_NAME, TRAIN_CSV_PATH, TRAIN_CLASSIFICATION_PATH, 'train')
make_classification_dataset(CPE_MODEL_NAME, CME_MODEL_NAME, TEST_CSV_PATH, TEST_CLASSIFICATION_PATH, 'test')
make_classification_dataset(CPE_MODEL_NAME, CME_MODEL_NAME, VAL_CSV_PATH, VAL_CLASSIFICATION_PATH, 'val')
