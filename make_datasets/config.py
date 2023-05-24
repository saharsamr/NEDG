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

TRAIN_JSONL_PATH = '../data/wikipedia/train.jsonl'
TEST_JSONL_PATH = '../data/wikipedia/test.jsonl'
VAL_JSONL_PATH = '../data/wikipedia/val.jsonl'

TRAIN_SHARE = 0.8
TEST_SHARE = 0.1
VAL_SHARE = 0.1

SOURCE_DEFINITION = 'wikipedia'