from transformers import TrainingArguments
import pandas as pd

from GNED.utils.save_data import save_generation_predictions
from GNED.utils.metrics import evaluate_generation
from GNED.models.BART import BART
from GNED.config import \
    TRAIN_GENERATION_FILE, TEST_GENERATION_FILE, VALID_GENERATION_FILE, \
    EPOCHS, TRAIN_GENERATION_BATCH_SIZE, EVAL_GENERATION_BATCH_SIZE, \
    WARMUP_STEPS, WEIGHT_DECAY, LOGGING_DIR, \
    OUTPUT_DIR, LOAD_GENERATION_MODEL, PRED_GENERATION_FILE_PATH, EVALUATE_GENERATION, MASK_ENTITY


def generation_main():

    train = pd.read_csv(TRAIN_GENERATION_FILE, delimiter='\1').sample(frac=0.2, random_state=42)
    print('train size before dropping NaNs: ', len(train))
    test = pd.read_csv(TEST_GENERATION_FILE, delimiter='\1').sample(frac=0.2, random_state=42)
    print('test size before dropping NaNs: ', len(test))
    valid = pd.read_csv(VALID_GENERATION_FILE, delimiter='\1').sample(frac=0.2, random_state=42)
    print('valid size before dropping NaNs: ', len(valid))

    train = train.dropna()
    print('train size after dropping NaNs: ', len(train))
    test = test.dropna()
    print('test size after dropping NaNs: ', len(test))
    valid = valid.dropna()
    print('valid size after dropping NaNs: ', len(valid))

    train_x, train_y = list(train['contexts']), list(train['entity_description'])
    test_x, test_y = list(test['contexts']), list(test['entity_description'])
    valid_x, valid_y = list(valid['contexts']), list(valid['entity_description'])

    training_args = TrainingArguments(
        num_train_epochs=EPOCHS,
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=TRAIN_GENERATION_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_GENERATION_BATCH_SIZE,
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
        save_total_limit=5
    )

    print('Initialing the model...')
    model = BART(
        training_args,
        train_x, train_y,
        test_x, test_y,
        valid_x, valid_y,
        load=LOAD_GENERATION_MODEL,
        mask_entity=MASK_ENTITY
    )

    model.set_learnable_params(freeze_decoder=False)
    print('Start training...')
    model.train()

    print('Start prediction...')
    preds, inputs, labels = model.pred()
    save_generation_predictions(inputs, labels, preds)

    if EVALUATE_GENERATION:
        evaluate_generation(PRED_GENERATION_FILE_PATH)
