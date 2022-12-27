BATCH_SIZE = 32
EPOCHS = 4

INPUT_MAX_LENGTH = 600
OUTPUT_MAX_LENGTH = 10

MODEL_NAME = 'facebook/bart-large-cnn'

LOGGING_DIR = 'logs'
OUTPUT_DIR = 'results'

TRAIN_FILE = 'data/train_human.csv'
TEST_FILE = 'data/test_human.csv'
VALID_FILE = 'data/valid_human.csv'

WARMUP_STEPS = 200
WEIGHT_DECAY = 0
LEARNING_RATE = 5e-5

ADDITIONAL_SPECIAL_TOKENS = ['<NE>', '</NE>', '<CNTXT>', '</CNTXT>']

MIN_CONTEXT_LEN = 30
MAX_CONTEXT_NUM = 5

LOAD_MODEL = False
MODEL_PATH = 'results/...'



