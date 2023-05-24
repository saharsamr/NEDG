import random
import json
from config import WIKI_DUMP_JSONL_PATH, TRAIN_JSONL_PATH, TEST_JSONL_PATH, \
    VAL_JSONL_PATH, TRAIN_SHARE, TEST_SHARE, SOURCE_DEFINITION


with open(WIKI_DUMP_JSONL_PATH, 'r') as data_f:
    with open(TRAIN_JSONL_PATH, 'w+') as train_f, (TEST_JSONL_PATH, 'w+') as test_f, (VAL_JSONL_PATH, 'w+') as val_f:

        for entity in data_f.readlines():
            rand = random.random()
            json_obj = json.loads(entity)
            if SOURCE_DEFINITION == 'wikipedia':
                if json_obj['wikipedia_description'] is None or json_obj['wikipedia_description'] == '':
                    continue
            elif SOURCE_DEFINITION == 'wikidata':
                if json_obj['wikidata_description'] is None or json_obj['wikidata_description'] == '':
                    continue
            if rand < TRAIN_SHARE:
                train_f.write(entity)
            elif TRAIN_SHARE < rand < TRAIN_SHARE + TEST_SHARE:
                test_f.write(entity)
            else:
                val_f.write(entity)
