from transformers import TrainingArguments
from utils.save_data import save_classification_prediction
from models.BERTBinaryClassifier import BERTBinaryClassification
import pandas as pd
from config import \
    TRAIN_FILE, TEST_FILE, \
    EPOCHS, TRAIN_BATCH_SIZE, \
    WARMUP_STEPS, WEIGHT_DECAY, LOGGING_DIR, \
    OUTPUT_DIR, LOAD_MODEL


def classification_main():

    train = pd.read_csv(TRAIN_FILE, delimiter='\1', header=0, names=['title', 'context', 'label']).dropna()
    test = pd.read_csv(TEST_FILE, delimiter='\1', header=0, names=['title', 'context', 'label']).dropna()

    train_x, train_y = list(train['context']), list(train['label'])
    test_x, test_y = list(test['context']), list(test['label'])

    training_args = TrainingArguments(
        num_train_epochs=EPOCHS,
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        logging_dir=LOGGING_DIR,
        logging_strategy='steps',
        logging_steps=100
    )

    print('Initialing the model...')
    model = BERTBinaryClassification(
        training_args,
        train_x, train_y,
        test_x, test_y,
        load=LOAD_MODEL
    )

    model.set_learnable_params(freeze_encoder=False)
    print('Start training...')
    model.train()

    print('Start prediction...')
    preds, inputs, labels = model.pred()
    save_classification_prediction(inputs, labels, preds)
