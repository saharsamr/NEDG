import os
dirname = os.path.dirname(__file__)

LOGGING_DIR = f'{dirname}/../logs'

JSONL_PATH = f'{dirname}/../data/wikipedia/wiki_dump.jsonl'
ENTITY_POPULARITY_PATH = f'{dirname}/../data/entity_popularity.pkl'
ENTITY_NAME_CARDINALITY_PATH = f'{dirname}/../data/cardinality.json'
ENTITY_PAGE_VIEW_PATH = f'{dirname}/../data/title_to_view.json'
ENTITY_WIKIDATA_ID_PATH = f'{dirname}/../data/title_to_wikidata_id.json'
CARDINALITY_DATA_JSON_PATH = f'{dirname}/../data/cardinality_analysis_data.json'

TRAIN_JSONL_PATH = f'{dirname}/../data/wikipedia/train_wikidata.jsonl'
TEST_JSONL_PATH = f'{dirname}/../data/wikipedia/test_wikidata.jsonl'
VAL_JSONL_PATH = f'{dirname}/../data/wikipedia/val_wikidata.jsonl'

ENTITY_ALIASES_DICT_PATH = f'{dirname}/../data/entity_aliases.pkl'
ENTITY_CARDINALITY_PATH = f'{dirname}/../data/cardinality.pkl'
CLASSIFICATION_RESULT_PATH = \
    f'{dirname}/../results/1-context-4epoch-wikidata-con+pred1+pred2-classification-9500/test_result_df.pkl'

MASK_PROB = 0.0
MASKING_STRATEGY = 'Complete'  # or Complete
MODEL_GENERATION_NAME = 'facebook/bart-large-cnn'  # facebook/bart-large-cnn or t5-base
DEFINITION_SOURCE = 'wikidata'  # wikidata or wikipedia
MODEL_NAME = f'MASK_{MASK_PROB}_{DEFINITION_SOURCE}'
OUTPUT_DIR = f'{dirname}/../results/{MODEL_GENERATION_NAME.split("/")[-1]}/{MODEL_NAME}'
MAX_CONTEXT_NUMBER = 1

TEST_RESULTS = f'{OUTPUT_DIR}/{MAX_CONTEXT_NUMBER}_contexts_{DEFINITION_SOURCE}_{MODEL_NAME}_preds.csv'
TEST_ANALYSIS_FILE = f'{OUTPUT_DIR}/{MAX_CONTEXT_NUMBER}_contexts_{DEFINITION_SOURCE}_{MODEL_NAME}_analysis.csv'

PLOT_SAVING_PATH = f'{dirname}/../plots/'

CSME_TEST_FILE = 'data/wikipedia/1_contexts_wikidata_CSME_test.csv'
