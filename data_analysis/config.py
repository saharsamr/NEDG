import os
dirname = os.path.dirname(__file__)

JSONL_PATH = f'{dirname}/../data/wikipedia/wiki_dump.jsonl'
ENTITY_POPULARITY_PATH = f'{dirname}/../data/entity_popularity.pkl'
CLASSIFICATION_RESULT_PATH = f'{dirname}/../results/1-context-4epoch-wikidata-con+pred1+pred2-classification-9500/test_result_df.pkl'
