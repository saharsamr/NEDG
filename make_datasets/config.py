MONGODB_LINK = 'mongodb://localhost'
MONGODB_PORT = 27017
MONGODB_DATABASE = 'wikipedia'
MONGODB_COLLECTION = 'dump'
MONGODB_READ_BATCH_SIZE = 4000
MONGODB_WRITE_BATCH_SIZE = 2000
MONGODB_USERNAME = 'user'
MONGODB_PASSWORD = 'pass'

WIKI_JSONS_PATH = '../../wikipedia/text'

MAX_ENTITY_NAME_LENGTH = 5
MIN_CONTEXT_LENGTH = 20
WIKI_DUMP_JSONL_PATH = '../data/wikipedia/wiki_dump.jsonl'
FINAL_MIN_CONTEXT_LEN = 15

TRAIN_SHARE = 0.8
TEST_SHARE = 0.1
VAL_SHARE = 0.1

SOURCE_DEFINITION = 'wikipedia'
TRAIN_JSONL_PATH = f'../data/wikipedia/train_{SOURCE_DEFINITION}.jsonl'
TEST_JSONL_PATH = f'../data/wikipedia/test_{SOURCE_DEFINITION}.jsonl'
VAL_JSONL_PATH = f'../data/wikipedia/val_{SOURCE_DEFINITION}.jsonl'

CSVS_PATH = '../data/wikipedia/'
MAX_CONTEXT_NUMBER = 1
