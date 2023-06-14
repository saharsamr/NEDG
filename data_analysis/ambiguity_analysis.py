import json
from collections import defaultdict
import re
from tqdm import tqdm
from pymongo import MongoClient
import requests
import pickle
import pandas as pd
import csv

from transformers import TrainingArguments
from datasets import load_metric
from data_analysis.config import *
from GNED.models.BART import BART

from make_datasets.config import MONGODB_LINK, MONGODB_PORT, MONGODB_DATABASE, \
    MONGODB_COLLECTION, MONGODB_READ_BATCH_SIZE, MONGODB_PASSWORD, \
    MONGODB_USERNAME
from data_analysis.config import ENTITY_NAME_CARDINALITY_PATH, ENTITY_POPULARITY_PATH, \
    ENTITY_PAGE_VIEW_PATH, ENTITY_WIKIDATA_ID_PATH, TRAIN_JSONL_PATH, TEST_JSONL_PATH, VAL_JSONL_PATH


def extract_file_mentions(file, cardinality):
    for entity in tqdm(file.readlines()):
        entity = json.loads(entity)
        for context in entity['contexts']:
            mentions = re.findall(r'<NE>(.*?)</NE>', context)
            for mention in mentions:
                cardinality[mention].add(entity['wikipedia_title'])

    return cardinality


def extract_wikidata_ids(cardinality):
    title_set = []
    for k, v in tqdm(cardinality.items()):
        title_set.extend(v)
    title_set = set(title_set)

    client = MongoClient(f'{MONGODB_LINK}:{MONGODB_PORT}/', username=MONGODB_USERNAME, password=MONGODB_PASSWORD)
    db = client[MONGODB_DATABASE]
    collection = db[MONGODB_COLLECTION]
    documents_cursor = collection.find(
        {'title': {'$in': title_set}}, {'wikidata_info.id': 1, 'id': 1},
        batch_size=MONGODB_READ_BATCH_SIZE)

    title_to_wikidata_id = {}
    for doc in documents_cursor:
        title_to_wikidata_id[doc['title']] = doc['wikidata_info']['id']

    with open(ENTITY_WIKIDATA_ID_PATH, 'w') as f:
        json.dump(title_to_wikidata_id, f)

    return title_to_wikidata_id


def extract_page_view(title_to_wikidata_id):
    title_to_view = {}
    for title, id_ in title_to_wikidata_id.items():

        get_req = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/" \
                  f"en.wikipedia.org/all-access/user/{id_}/monthly/2023010100/2023020100"

        headers = {"Accept": "application/json"}

        resp = requests.get(get_req, headers=headers)
        print(resp.content)
        details = resp.json()

        item_view = 0
        for item in details['items']:
            item_view += item['view']

        title_to_view[title] = item_view

    with open(ENTITY_PAGE_VIEW_PATH, 'w') as f:
        json.dump(title_to_view, f)

    return title_to_view


def extract_cardinality_and_popularity(train_path, test_path, valid_path):
    cardinality = defaultdict(set)
    with open(train_path, 'r') as file:
        cardinality = extract_file_mentions(file, cardinality)

    with open(test_path, 'r') as file:
        cardinality = extract_file_mentions(file, cardinality)

    with open(valid_path, 'r') as file:
        cardinality = extract_file_mentions(file, cardinality)

    cardinality = {k: list(v) for k, v in tqdm(cardinality.items()) if len(v) > 1}
    # title_to_wikidata_id = extract_wikidata_ids(cardinality)

    with open(ENTITY_POPULARITY_PATH, 'rb') as file:
        entity_popularity = pickle.load(file)

    cardinality = {k: {t: entity_popularity[t] for t in v} for k, v in tqdm(cardinality.items())}

    with open(ENTITY_NAME_CARDINALITY_PATH, 'w') as file:
        json.dump(cardinality, file)


extract_cardinality_and_popularity(
    TRAIN_JSONL_PATH, TEST_JSONL_PATH, VAL_JSONL_PATH
)


def make_cpe_cme_dataset_for_cardinality_analysis(CPE_model_name, CME_model_name, input_file, output_file, delimiter='\1'):
    # Read the file cardinality data json path
    with open(ENTITY_NAME_CARDINALITY_PATH, 'r') as file:
        cardinality = json.load(file)

    # Create contexts and descriptions
    # TODO: i should replace the correct reading data. this is shit
    input_data = pd.read_csv(input_file, delimiter=delimiter)
    input_x, input_y = list(input_data['contexts']), list(input_data['entity_description'])

    # Train CPE and CME models
    training_args = TrainingArguments(
        output_dir=f'{dirname}/../results',
        logging_dir=LOGGING_DIR,
        logging_strategy='steps',
        logging_steps=100,
    )
    CPE_model = BART(
        training_args,
        input_x, input_y, input_x, input_y, input_x, input_y,
        load=True, model_load_path=CPE_model_name,
        model_name=CPE_model_name, mask_entity=False
    )
    CME_model = BART(
        training_args,
        input_x, input_y, input_x, input_y, input_x, input_y,
        load=True, model_load_path=CME_model_name,
        model_name=CME_model_name, mask_entity=True
    )

    # Create a mapping for the context and entity name in a dataframe
    df = pd.DataFrame(columns=['context', 'entity_name', 'entity_id'])

    for entity_name, entity_list in tqdm(cardinality.items()):
        # Get the top two most popular entities for the current entity name
        # TODO: entity_popularity from where? I should make it or read it
        entity_list = sorted(entity_list, key=lambda entity: entity_popularity[entity], reverse=True)[:2]

        for context in input_x:
            # Replace the entity name in the context with the current entity from the list
            for entity_id in entity_list:

                # Use CPE and CME to generate the masked and unmasked descriptions
                masked_desc = CME_model.predict(context)
                unmasked_desc = CPE_model.predict(context)

                # Add the context, entity name, and entity ID to the dataframe
                df = df.append({'context': context, 'entity_name': entity_name, 'entity_id': entity_id,
                                'masked_description': masked_desc, 'unmasked_description': unmasked_desc}, ignore_index=True)

    # Save the dataframe to a CSV file
    df.to_csv(output_file, index=False)