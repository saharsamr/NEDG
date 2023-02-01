TASK = 'CLASSIFICATION'

TRAIN_GENERATION_BATCH_SIZE = 16
EVAL_GENERATION_BATCH_SIZE = 16
TEST_GENERATION_BATCH_SIZE = 16

TRAIN_CLASSIFICATION_BATCH_SIZE = 32
EVAL_CLASSIFICATION_BATCH_SIZE = 32
TEST_CLASSIFICATION_BATCH_SIZE = 32

EPOCHS = 4

INPUT_GENERATION_MAX_LENGTH = 600
OUTPUT_GENERATION_MAX_LENGTH = 15
OUTPUT_GENERATION_MIN_LENGTH = 3

INPUT_CLASSIFICATION_MAX_LENGTH = 512

MODEL_GENERATION_NAME = 'facebook/bart-large-cnn'

LOGGING_DIR = 'logs'
OUTPUT_DIR = 'results'

DATA_GENERATION_FOLDER = 'data/HumanConcatenated'
DATA_GENERATION_CATEGORY = 'human'
DATA_GENERATION_STYLE = 'ne_with_context'

TRAIN_GENERATION_FILE = f'{DATA_GENERATION_FOLDER}/train_{DATA_GENERATION_CATEGORY}_{DATA_GENERATION_STYLE}.csv'
TEST_GENERATION_FILE = f'{DATA_GENERATION_FOLDER}/test_{DATA_GENERATION_CATEGORY}_{DATA_GENERATION_STYLE}.csv'
VALID_GENERATION_FILE = f'{DATA_GENERATION_FOLDER}/valid_{DATA_GENERATION_CATEGORY}_{DATA_GENERATION_STYLE}.csv'

DATA_CLASSIFICATION_FOLDER = 'data/Classification'
DATA_CLASSIFICATION_CATEGORY = 'human'
TRAIN_CLASSIFICATION_FILE = f'{DATA_CLASSIFICATION_FOLDER}/train_{DATA_CLASSIFICATION_CATEGORY}_classification.csv'
TEST_CLASSIFICATION_FILE = f'{DATA_CLASSIFICATION_FOLDER}/test_{DATA_CLASSIFICATION_CATEGORY}_classification.csv'
VALID_CLASSIFICATION_FILE = f'{DATA_CLASSIFICATION_FOLDER}/valid_{DATA_CLASSIFICATION_CATEGORY}_classification.csv'

WARMUP_STEPS = 200
WEIGHT_DECAY = 0.1
LEARNING_RATE = 5e-5

ADDITIONAL_SPECIAL_TOKENS = ['<NE>', '</NE>', '<CNTXT>', '</CNTXT>']

MIN_CONTEXT_LEN = 30  # TODO: I removed this parameter in this version of data, if that matters at all.
MAX_CONTEXT_NUM = 5

LOAD_GENERATION_MODEL = False
MODEL_GENERATION_PATH = 'results/...'

LOAD_CLASSIFICATION_MODEL = False
MODEL_CLASSIFICATION_PATH = 'results/checkpoint-7000'

EVALUATE_GENERATION = True
PRED_GENERATION_FILE_PATH = 'preds.csv'

EVALUATE_CLASSIFICATION = True
PRED_CLASSIFICATION_FILE_PATH = 'classification_preds.csv'

MASKING_PROBABILITY = 1.0
