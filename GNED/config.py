from make_datasets.config import MAX_CONTEXT_NUMBER
import os
dirname = os.path.dirname(__file__)

TASK = 'GENERATION'

WARMUP_STEPS = 200
WEIGHT_DECAY = 0.1
LEARNING_RATE = 5e-5

DEFINITION_SOURCE = 'wikidata'

# ========================== GENERATION CONFIGS ==========================
MASK_PROB = 0.0
MASKING_STRATEGY = 'Complete'  # or Complete
MODEL_GENERATION_NAME = 'facebook/bart-large'  # facebook/bart-large-cnn or t5-base
MODEL_NAME = f'MASK_{MASK_PROB}_{DEFINITION_SOURCE}'
OUTPUT_DIR = f'{dirname}/../results/{MODEL_GENERATION_NAME.split("/")[-1]}/{MODEL_NAME}'
LOGGING_DIR = f'{dirname}/../logs/{MODEL_GENERATION_NAME.split("/")[-1]}/{MODEL_NAME}'

TRAIN_GENERATION_BATCH_SIZE = 16
EVAL_GENERATION_BATCH_SIZE = 16
TEST_GENERATION_BATCH_SIZE = 16

INPUT_GENERATION_MAX_LENGTH = 600
OUTPUT_GENERATION_MAX_LENGTH = 25
OUTPUT_GENERATION_MIN_LENGTH = 5

DATA_GENERATION_FOLDER = f'{dirname}/../data/wikipedia/'
TRAIN_GENERATION_FILE = f'{DATA_GENERATION_FOLDER}{MAX_CONTEXT_NUMBER}_contexts_{DEFINITION_SOURCE}_train.csv'
TEST_GENERATION_FILE = f'{DATA_GENERATION_FOLDER}{MAX_CONTEXT_NUMBER}_contexts_{DEFINITION_SOURCE}_test.csv'
VALID_GENERATION_FILE = f'{DATA_GENERATION_FOLDER}{MAX_CONTEXT_NUMBER}_contexts_{DEFINITION_SOURCE}_val.csv'

if MAX_CONTEXT_NUMBER == 1:
    if 'gpt' in MODEL_GENERATION_NAME:
        ADDITIONAL_SPECIAL_TOKENS = ['<NE>', '</NE>', '<entity_description>', '<entity_context>']
    else:
        ADDITIONAL_SPECIAL_TOKENS = ['<NE>', '</NE>']
else:
    if 'gpt' in MODEL_GENERATION_NAME:
        ADDITIONAL_SPECIAL_TOKENS = ['<NE>', '</NE>', '<entity_description>', '<entity_context>', '<CNTXT>', '</CNTXT>']
    else:
        ADDITIONAL_SPECIAL_TOKENS = ['<NE>', '</NE>', '<CNTXT>', '</CNTXT>']

LOAD_GENERATION_MODEL = False
MODEL_GENERATION_PATH = f'{OUTPUT_DIR}/final_model/'

EVALUATE_GENERATION = True
PRED_GENERATION_FILE_PATH = f'{OUTPUT_DIR}/{MAX_CONTEXT_NUMBER}_contexts_{DEFINITION_SOURCE}_{MODEL_NAME}_preds.csv'

EPOCHS = 2
# ========================== GENERATION CONFIGS ==========================

# ======================== CLASSIFICATION CONFIGS ========================
TRAIN_CLASSIFICATION_BATCH_SIZE = 16
EVAL_CLASSIFICATION_BATCH_SIZE = 16
TEST_CLASSIFICATION_BATCH_SIZE = 16

INPUT_CLASSIFICATION_MAX_LENGTH = 510

MODEL_CLASSIFICATION_NAME = 'bert-large-uncased'

CLASSIFICATION_SPECIAL_TOKENS = ['<NE>', '</NE>', '[SEC]']

DATA_CLASSIFICATION_FOLDER = f'{dirname}/../data/wikipedia/'
TRAIN_CLASSIFICATION_FILE = f'{DATA_CLASSIFICATION_FOLDER}{MAX_CONTEXT_NUMBER}_contexts_{DEFINITION_SOURCE}_classification_train.csv'
TEST_CLASSIFICATION_FILE = f'{DATA_CLASSIFICATION_FOLDER}{MAX_CONTEXT_NUMBER}_contexts_{DEFINITION_SOURCE}_classification_test.csv'
VALID_CLASSIFICATION_FILE = f'{DATA_CLASSIFICATION_FOLDER}{MAX_CONTEXT_NUMBER}_contexts_{DEFINITION_SOURCE}_classification_val.csv'

LOAD_CLASSIFICATION_MODEL = False
MODEL_CLASSIFICATION_PATH = f'{OUTPUT_DIR}/{MAX_CONTEXT_NUMBER}_contexts_{DEFINITION_SOURCE}_classifier/'

EVALUATE_CLASSIFICATION = True
PRED_CLASSIFICATION_FILE_PATH = f'{OUTPUT_DIR}/{MAX_CONTEXT_NUMBER}_contexts_{DEFINITION_SOURCE}_classifier_preds.csv'
# ======================== CLASSIFICATION CONFIGS ========================
