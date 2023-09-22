from transformers import TrainingArguments
import pandas as pd

from GNED.utils.save_data import save_generation_predictions
from GNED.utils.metrics import evaluate_generation
from GNED.models.BART import BART
from GNED.models.T5 import T5
from GNED.models.GPT2 import GPT2
from GNED.config import *


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

    train_entity_names = list(train['entity_name'])
    test_entity_names = list(test['entity_name'])
    valid_entity_names = list(valid['entity_name'])

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
        load_best_model_at_end=False,
        evaluation_strategy='steps',
        do_eval=True,
        eval_steps=1000,
        save_strategy='steps',
        save_steps=1000,
        save_total_limit=2
    )

    print('Initialing the model...')
    if 'bart' in MODEL_GENERATION_NAME:
        model = BART(
            training_args,
            train_x, train_y,
            test_x, test_y,
            valid_x, valid_y,
            train_entity_names, test_entity_names, valid_entity_names,
            load=LOAD_GENERATION_MODEL
        )
    elif 't5' in MODEL_GENERATION_NAME:
        model = T5(
            training_args,
            train_x, train_y,
            test_x, test_y,
            valid_x, valid_y,
            train_entity_names, test_entity_names, valid_entity_names,
            load=LOAD_GENERATION_MODEL
        )
    elif 'gpt2' in MODEL_GENERATION_NAME:
        model = GPT2(
            training_args,
            train_x, train_y,
            test_x, test_y,
            valid_x, valid_y,
            train_entity_names, test_entity_names, valid_entity_names,
            load=LOAD_GENERATION_MODEL
        )
    else:
        raise Exception('Model name not recognized')

    model.set_learnable_params(freeze_decoder=False)
    print('Start training...')
    model.train()

    print('Start prediction...')
    preds, inputs, labels, entity_names = model.pred()
    save_generation_predictions(inputs, labels, preds, entity_names)

    if EVALUATE_GENERATION:
        evaluate_generation(PRED_GENERATION_FILE_PATH)
