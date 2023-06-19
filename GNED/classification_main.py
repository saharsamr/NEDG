from transformers import TrainingArguments
import pandas as pd
import numpy as np
import pickle

from GNED.utils.save_data import save_classification_prediction
from GNED.models.BERTBinaryClassifier import BERTBinaryClassification
from GNED.utils.metrics import evaluate_classification
from GNED.config import \
    TRAIN_CLASSIFICATION_FILE, TEST_CLASSIFICATION_FILE, VALID_CLASSIFICATION_FILE,\
    EPOCHS, TRAIN_CLASSIFICATION_BATCH_SIZE, EVAL_CLASSIFICATION_BATCH_SIZE, \
    WARMUP_STEPS, WEIGHT_DECAY, LOGGING_DIR, \
    OUTPUT_DIR, LOAD_CLASSIFICATION_MODEL, EVALUATE_CLASSIFICATION


def classification_main():

    # col_names = ['label', 'title', 'CPE-context', 'CPE-pred', 'CPE-bert',
    #              'CME-context', 'CME-pred', 'CME-bert', 'class-label']

    train = pd.read_csv(TRAIN_CLASSIFICATION_FILE, delimiter='\1')
    test = pd.read_csv(TEST_CLASSIFICATION_FILE, delimiter='\1')
    valid = pd.read_csv(VALID_CLASSIFICATION_FILE, delimiter='\1')

    train = train.replace({'<pad>': ''}, regex=True)
    test = test.replace({'<pad>': ''}, regex=True)
    valid = valid.replace({'<pad>': ''}, regex=True)

    train['CME-pred'] = train['CME-pred'].fillna('')
    test['CME-pred'] = test['CME-pred'].fillna('')
    valid['CME-pred'] = valid['CME-pred'].fillna('')

    train['CPE-pred'] = train['CPE-pred'].fillna('')
    test['CPE-pred'] = test['CPE-pred'].fillna('')
    valid['CPE-pred'] = valid['CPE-pred'].fillna('')

    train['text'] = train['CME-context'] + "[SEC]" + train['CME-pred'] + "[SEC]" + train['CPE-pred']
    test['text'] = test['CME-context'] + "[SEC]" + test['CME-pred'] + "[SEC]" + test['CPE-pred']
    valid['text'] = valid['CME-context'] + "[SEC]" + valid['CME-pred'] + "[SEC]" + valid['CPE-pred']

    train_x, train_y = list(train['text']), list(train['class-label'])
    test_x, test_y = list(test['text']), list(test['class-label'])
    valid_x, valid_y = list(valid['text']), list(valid['class-label'])

    training_args = TrainingArguments(
        num_train_epochs=EPOCHS,
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=TRAIN_CLASSIFICATION_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_CLASSIFICATION_BATCH_SIZE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        logging_dir=LOGGING_DIR,
        logging_strategy='steps',
        logging_steps=100,
        evaluation_strategy='steps',
        do_eval=True,
        eval_steps=1000,
        load_best_model_at_end=True,
        save_strategy='steps',
        save_steps=1000,
        save_total_limit=5
    )

    print('Initialing the model...')
    model = BERTBinaryClassification(
        training_args,
        train_x, train_y,
        test_x, test_y,
        valid_x, valid_y,
        load=LOAD_CLASSIFICATION_MODEL
    )

    model.set_learnable_params(freeze_encoder=True)
    print('Start training...')
    model.train()

    print('Start prediction...')
    preds, inputs, labels = model.pred()
    save_classification_prediction(inputs, labels, preds)
    preds = np.array(preds).reshape([-1])
    test['class-pred'] = preds

    with open('test_result_df.pkl', 'wb') as f:
        pickle.dump(test, f)

    if EVALUATE_CLASSIFICATION:
        evaluate_classification(test)
