import pandas as pd
from transformers import TrainingArguments
from datasets import load_metric
import csv

from GNED.models.BART import BART
from make_datasets.config import *

import os
dirname = os.path.dirname(__file__)


def make_CSME_dataset(CSME_model_name, input_file, output_file, delimiter='\1'):

    input_data = pd.read_csv(input_file, delimiter=delimiter).sample(frac=0.05, random_state=42)
    print(f'input-data length before dropping non values: {len(input_data)}')
    input_data = input_data.dropna()
    print(f'input-data length after dropping non values: {len(input_data)}')
    input_x, input_y = list(input_data['contexts']), list(input_data['entity_description'])

    training_args = TrainingArguments(
        output_dir=f'{dirname}/../results',
        logging_dir=LOGGING_DIR,
        logging_strategy='steps',
        logging_steps=100,
    )
    CSME_model = BART(
        training_args,
        input_x, input_y, input_x, input_y, input_x, input_y,
        load=True, model_load_path=CSME_model_name,
        model_name=CSME_model_name, mask_entity=False
    )

    CSME_preds, CSME_inputs, CSME_labels = CSME_model.pred()

    bertscore = load_metric('bertscore')
    CSME_bert = bertscore.compute(
        predictions=CSME_preds, references=CSME_labels, lang='en', model_type='bert-large-uncased')['f1']

    with open(output_file, 'w') as f:

        columns = ['label', 'title', 'CSME-context', 'CSME-pred', 'CSME-bert']

        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow(columns)

        for label, title, csme_context, csme_pred, csme_bert in \
            zip(input_y, list(input_data['entity_name']), CSME_inputs, CSME_preds,
                CSME_bert):

            row = [label.replace(delimiter, ''), title.replace(delimiter, ''), csme_context.replace(delimiter, ''),
                   csme_pred.replace(delimiter, ''), csme_bert]
            writer.writerow(row)


make_CSME_dataset(CSME_MODEL_NAME, TEST_CSV_PATH, CSME_TEST_PATH)
