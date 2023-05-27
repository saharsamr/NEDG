from transformers import TrainingArguments
from utils.save_data import save_classification_prediction
from models.BERTBinaryClassifier import BERTBinaryClassification
from utils.metrics import evaluate_classification
import pandas as pd
import numpy as np
from config import \
    TRAIN_CLASSIFICATION_FILE, TEST_CLASSIFICATION_FILE, VALID_CLASSIFICATION_FILE,\
    EPOCHS, TRAIN_CLASSIFICATION_BATCH_SIZE, EVAL_CLASSIFICATION_BATCH_SIZE, \
    WARMUP_STEPS, WEIGHT_DECAY, LOGGING_DIR, \
    OUTPUT_DIR, LOAD_CLASSIFICATION_MODEL, EVALUATE_CLASSIFICATION
import pickle


def classification_main():

    # col_names = ['label', 'title', 'CPE-context', 'CPE-pred', 'CPE-bert',
    #              'CME-context', 'CME-pred', 'CME-bert', 'class-label']

    train = pd.read_csv(TRAIN_CLASSIFICATION_FILE, delimiter='\1').dropna()
    test = pd.read_csv(TEST_CLASSIFICATION_FILE, delimiter='\1').dropna()
    valid = pd.read_csv(VALID_CLASSIFICATION_FILE, delimiter='\1').dropna()

    train['text'] = train['CME-context'] + "[SEP]" + train['CME-pred'] + "[SEP]" + train['CPE-pred']
    test['text'] = test['CME-context'] + "[SEP]" + test['CME-pred'] + "[SEP]" + test['CPE-pred']
    valid['text'] = valid['CME-context'] + "[SEP]" + valid['CME-pred'] + "[SEP]" + valid['CPE-pred']

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
        eval_steps=500,
        load_best_model_at_end=True,
        save_strategy='steps',
        save_total_limit=3
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

    if EVALUATE_CLASSIFICATION:
        evaluate_classification(test)

    with open('test_result_df.pkl', 'wb') as f:
        pickle.dump(test, f)
