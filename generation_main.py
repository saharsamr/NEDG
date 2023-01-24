from transformers import TrainingArguments
from utils.save_data import save_generation_predictions
from utils.metrics import evaluate
from models.BART import BART
import pandas as pd
from config import \
    TRAIN_FILE, TEST_FILE, VALID_FILE, \
    EPOCHS, TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE, \
    WARMUP_STEPS, WEIGHT_DECAY, LOGGING_DIR, \
    OUTPUT_DIR, LOAD_MODEL, PRED_FILE_PATH, EVALUATE


def generation_main():

    train = pd.read_csv(TRAIN_FILE, delimiter='\1', header=0, names=['title', 'context', 'description'])
    test = pd.read_csv(TEST_FILE, delimiter='\1', header=0, names=['title', 'context', 'description'])
    valid = pd.read_csv(VALID_FILE, delimiter='\1', header=0, names=['title', 'context', 'description'])

    train_x, train_y = list(train['context']), list(train['description'])
    test_x, test_y = list(test['context']), list(test['description'])
    valid_x, valid_y = list(valid['context']), list(valid['description'])

    training_args = TrainingArguments(
        num_train_epochs=EPOCHS,
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        logging_dir=LOGGING_DIR,
        logging_strategy='steps',
        logging_steps=100,
        load_best_model_at_end=True,
        evaluation_strategy='steps',
        do_eval=True,
        eval_steps=500,
        save_strategy='steps',
        save_total_limit=3
    )

    print('Initialing the model...')
    model = BART(
        training_args,
        train_x, train_y,
        test_x, test_y,
        valid_x, valid_y,
        load=LOAD_MODEL
    )

    model.set_learnable_params(freeze_decoder=False)
    print('Start training...')
    model.train()

    print('Start prediction...')
    preds, inputs, labels = model.pred()
    save_generation_predictions(inputs, labels, preds)

    if EVALUATE:
        evaluate(PRED_FILE_PATH)