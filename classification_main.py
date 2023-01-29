from transformers import TrainingArguments
from utils.save_data import save_classification_prediction
from models.BERTBinaryClassifier import BERTBinaryClassification
from utils.metrics import evaluate_classification
import pandas as pd
from config import \
    TRAIN_CLASSIFICATION_FILE, TEST_CLASSIFICATION_FILE, VALID_CLASSIFICATION_FILE,\
    EPOCHS, TRAIN_CLASSIFICATION_BATCH_SIZE, EVAL_CLASSIFICATION_BATCH_SIZE, \
    WARMUP_STEPS, WEIGHT_DECAY, LOGGING_DIR, \
    OUTPUT_DIR, LOAD_CLASSIFICATION_MODEL, EVALUATE_CLASSIFICATION


def classification_main():

    train = pd.read_csv(
        TRAIN_CLASSIFICATION_FILE, delimiter='\1', header=None, names=['title', 'context', 'label', 'entity_name']
    ).dropna()
    test = pd.read_csv(
        TEST_CLASSIFICATION_FILE, delimiter='\1', header=None, names=['title', 'context', 'label', 'entity_name']
    ).dropna()
    valid = pd.read_csv(
        VALID_CLASSIFICATION_FILE, delimiter='\1', header=None, names=['title', 'context', 'label', 'entity_name']
    )

    train['text'] = train['entity_name'].astype(str) + train['context']
    test['text'] = test['entity_name'].astype(str) + test['context']
    valid['text'] = valid['entity_name'].astype(str) + valid['context']

    train_x, train_y = list(train['text']), list(train['label'])
    test_x, test_y = list(test['text']), list(test['label'])
    valid_x, valid_y = list(valid['text']), list(valid['label'])

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
        eval_steps=100
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

    if EVALUATE_CLASSIFICATION:
        evaluate_classification('classification_preds.csv')
