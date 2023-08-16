import json
from collections import defaultdict
from tqdm import tqdm
import pickle
import pymongo

from scipy import stats
import numpy as np

from data_analysis.config import CLASSIFICATION_RESULT_PATH, JSONL_PATH, \
    ENTITY_ALIASES_DICT_PATH, ENTITY_CARDINALITY_PATH
from data_analysis.utils import compute_metrics, compute_correlation, \
    add_bleu_rouge_to_df, remove_empty_preds, compute_metrics_for_every_fraction
from data_analysis.data_plots import plot_metrics, plot_correlation
from make_datasets.config import MONGODB_LINK, MONGODB_PORT, MONGODB_USERNAME, MONGODB_PASSWORD, MONGODB_DATABASE, \
    MONGODB_COLLECTION, MONGODB_READ_BATCH_SIZE


def make_cardinality_dictionary():
    with open(ENTITY_ALIASES_DICT_PATH, 'rb') as file:
        entity_aliases = pickle.load(file)
    cardinality = defaultdict(int)
    for entity in entity_aliases:
        cardinality[entity.key()] = cardinality[entity.key()] + 1
        for alias in cardinality.values():
            cardinality[alias['value']] = cardinality[alias['value']] + 1

    print("Saving dictionary cardinality to file...")
    with open(ENTITY_CARDINALITY_PATH, 'wb') as file:
        pickle.dump(cardinality, file)


def make_title_alias_dictionary():
    client = pymongo.MongoClient(f'{MONGODB_LINK}:{MONGODB_PORT}/', username=MONGODB_USERNAME,
                                 password=MONGODB_PASSWORD)
    db = client[MONGODB_DATABASE]
    collection = db[MONGODB_COLLECTION]

    total_documents = collection.count_documents({})
    documents_cursor = collection.find(batch_size=MONGODB_READ_BATCH_SIZE)

    dictionary = defaultdict(list)
    for document in tqdm(documents_cursor, total=total_documents):
        title = document['title']
        try:
            aliases = document["wikidata_info"]["aliases"]["en"]
        except:
            aliases = []
        dictionary[title] = aliases

    print("Saving dictionary aliases to file...")
    with open(ENTITY_ALIASES_DICT_PATH, 'wb') as file:
        pickle.dump(dictionary, file)


make_title_alias_dictionary()
make_cardinality_dictionary()
