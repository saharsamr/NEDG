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


def classification_main():

    col_names = [
        'context_we', 'label_we', 'pred_we', 'bert_we', 'context_woe',
        'label_woe', 'pred_woe', 'bert_woe', 'classification_label',
        'masked_context', 'entity_name'
    ]

    train = pd.read_csv(TRAIN_CLASSIFICATION_FILE, delimiter='~', header=None, names=col_names)
    test = pd.read_csv(TEST_CLASSIFICATION_FILE, delimiter='~', header=None, names=col_names)
    valid = pd.read_csv(VALID_CLASSIFICATION_FILE, delimiter='~', header=None, names=col_names)

    train['text'] = train['entity_name'] + "[SEP]" + train['pred_we'] + "[SEP]" + train['pred_woe'] + "[SEP]" + train['masked_context']
    test['text'] = test['entity_name'] + "[SEP]" + test['pred_we'] + "[SEP]" + test['pred_woe'] + "[SEP]" + test['masked_context']
    valid['text'] = valid['entity_name'] + "[SEP]" + valid['pred_we'] + "[SEP]" + valid['pred_woe'] + "[SEP]" + valid['masked_context']

    train_x, train_y = list(train['text']), list(train['classification_label'])
    test_x, test_y = list(test['text']), list(test['classification_label'])
    valid_x, valid_y = list(valid['text']), list(valid['classification_label'])

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
        eval_steps=100,
        disable_tqdm=True,
        load_best_model_at_end=True
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
    test['classification_prediction'] = preds
    test['selected_prediction'] = np.where(test['classification_prediction'] == 1, test['pred_we'], test['pred_woe'])

    if EVALUATE_CLASSIFICATION:
        evaluate_classification(test)
