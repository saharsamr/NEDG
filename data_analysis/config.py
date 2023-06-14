import os
dirname = os.path.dirname(__file__)

LOGGING_DIR = f'{dirname}/../logs'

JSONL_PATH = f'{dirname}/../data/wikipedia/wiki_dump.jsonl'
ENTITY_POPULARITY_PATH = f'{dirname}/../data/entity_popularity.pkl'
ENTITY_NAME_CARDINALITY_PATH = f'{dirname}/../data/cardinality.json'
ENTITY_PAGE_VIEW_PATH = f'{dirname}/../data/title_to_view.json'
ENTITY_WIKIDATA_ID_PATH = f'{dirname}/../data/title_to_wikidata_id.json'

TRAIN_JSONL_PATH = f'{dirname}/../data/wikipedia/train_wikidata.jsonl'
TEST_JSONL_PATH = f'{dirname}/../data/wikipedia/test_wikidata.jsonl'
VAL_JSONL_PATH = f'{dirname}/../data/wikipedia/val_wikidata.jsonl'

ENTITY_ALIASES_DICT_PATH = f'{dirname}/../data/entity_aliases.pkl'
ENTITY_CARDINALITY_PATH = f'{dirname}/../data/cardinality.pkl'
CLASSIFICATION_RESULT_PATH = f'{dirname}/../results/1-context-4epoch-wikidata-con+pred1+pred2-classification-9500/test_result_df.pkl'
