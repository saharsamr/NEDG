import os
dirname = os.path.dirname(__file__)
# ========================== GENERATION DATASETS ==========================
MONGODB_LINK = 'mongodb://localhost'
MONGODB_PORT = 27017
MONGODB_DATABASE = 'wikipedia'
MONGODB_COLLECTION = 'dump'
MONGODB_READ_BATCH_SIZE = 4000
MONGODB_WRITE_BATCH_SIZE = 2000
MONGODB_USERNAME = 'user'
MONGODB_PASSWORD = 'pass'

WIKI_JSONS_PATH = f'{dirname}/../../wikipedia/text'

MAX_ENTITY_NAME_LENGTH = 5
MIN_CONTEXT_LENGTH = 20
WIKI_DUMP_JSONL_PATH = f'{dirname}/../data/wikipedia/wiki_dump.jsonl'
FINAL_MIN_CONTEXT_LEN = 15

TRAIN_SHARE = 0.8
TEST_SHARE = 0.1
VAL_SHARE = 0.1

SOURCE_DEFINITION = 'wikidata'
TRAIN_JSONL_PATH = f'{dirname}/../data/wikipedia/train_{SOURCE_DEFINITION}.jsonl'
TEST_JSONL_PATH = f'{dirname}/../data/wikipedia/test_{SOURCE_DEFINITION}.jsonl'
VAL_JSONL_PATH = f'{dirname}/../data/wikipedia/val_{SOURCE_DEFINITION}.jsonl'

CSVS_PATH = f'{dirname}/../data/wikipedia/'
MAX_CONTEXT_NUMBER = 1
# ========================== GENERATION DATASETS ==========================

# ========================== CLASSIFICATION DATASETS ==========================
LOGGING_DIR = f'{dirname}/../logs'
CPE_MODEL_NAME = f'{dirname}/../results/{MAX_CONTEXT_NUMBER}-context-wikidata-CPE'
CME_MODEL_NAME = f'{dirname}/../results/{MAX_CONTEXT_NUMBER}-context-wikidata-CME'

TRAIN_CSV_PATH = f'{dirname}/../data/wikipedia/{MAX_CONTEXT_NUMBER}_contexts_{SOURCE_DEFINITION}_train.csv'
TEST_CSV_PATH = f'{dirname}/../data/wikipedia/{MAX_CONTEXT_NUMBER}_contexts_{SOURCE_DEFINITION}_test.csv'
VAL_CSV_PATH = f'{dirname}/../data/wikipedia/{MAX_CONTEXT_NUMBER}_contexts_{SOURCE_DEFINITION}_val.csv'

TRAIN_CLASSIFICATION_PATH = f'{dirname}/../data/wikipedia/{MAX_CONTEXT_NUMBER}_contexts_{SOURCE_DEFINITION}_classification_train.csv'
TEST_CLASSIFICATION_PATH = f'{dirname}/../data/wikipedia/{MAX_CONTEXT_NUMBER}_contexts_{SOURCE_DEFINITION}_classification_test.csv'
VAL_CLASSIFICATION_PATH = f'{dirname}/../data/wikipedia/{MAX_CONTEXT_NUMBER}_contexts_{SOURCE_DEFINITION}_classification_val.csv'
# ========================== CLASSIFICATION DATASETS ==========================
