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
    ENTITY_PAGE_VIEW_PATH, ENTITY_WIKIDATA_ID_PATH, TRAIN_JSONL_PATH, TEST_JSONL_PATH, \
    VAL_JSONL_PATH, CARDINALITY_DATA_JSON_PATH


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


def extract_entity_specific_context(file_path, json_dump_path):

    with open(file_path, 'r') as file:
        cardinality = json.load(file)

    cardinality = {k: v for k, v in cardinality.items() if len(v) < 6}

    title_to_entity_info = {}
    with open(json_dump_path, 'r') as file:
        for line in file.readlines():
            data = json.loads(line)
            title_to_entity_info[data['wikipedia_title']] = data

    cardinality_analysis_data = defaultdict(dict)
    for entity_name, entity_list in cardinality.items():

        for entity, popularity in entity_list.items():

            entity_contexts = title_to_entity_info[entity]['contexts']
            entity_contexts = [context for context in entity_contexts if f'<NE>{entity_name}</NE>' in context]
            entity_description = title_to_entity_info[entity]['wikidata_description']

            cardinality_analysis_data[entity_name][entity] = {
                'popularity': popularity,
                'contexts': entity_contexts,
                'description': entity_description
            }

    with open(CARDINALITY_DATA_JSON_PATH, 'w') as file:
        json.dump(cardinality_analysis_data, file)


extract_cardinality_and_popularity(
    TRAIN_JSONL_PATH, TEST_JSONL_PATH, VAL_JSONL_PATH
)


def make_cpe_cme_dataset_for_cardinality_analysis(CPE_model_name, CME_model_name, input_file, output_file,
                                                  delimiter='\1'):
    # Read the file cardinality data json path
    with open(ENTITY_NAME_CARDINALITY_PATH, 'r') as file:
        cardinality = json.load(file)

    # Create input_x and input_y from the contexts and description fields in the cardinality dictionary
    input_x = []
    input_y = []
    for entity_name, entity_dict in cardinality.items():
        for entity_title, entity_info in entity_dict.items():
            if 'description' in entity_info:
                for context in entity_info['contexts']:
                    input_x.append(context)
                    input_y.append(entity_info['description'])

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

    # Create dataframes for the most popular and second most popular entities for each entity name
    df_most_popular = pd.DataFrame(
        columns=['context', 'entity_name', 'entity_id', 'masked_description', 'unmasked_description'])
    df_second_most_popular = pd.DataFrame(
        columns=['context', 'entity_name', 'entity_id', 'masked_description', 'unmasked_description'])
    for entity_name, entity_dict in tqdm(cardinality.items()):
        # Sort the entities by popularity and get the top two
        entities_sorted = sorted(entity_dict.items(), key=lambda x: x[1]['popularity'], reverse=True)[:2]
        entity_most_popular, entity_second_most_popular = entities_sorted

        # Create a dataframe with the context, entity name, entity ID, masked description, and unmasked description
        # for the most popular entity
        df_most_popular_entity = pd.DataFrame(
            columns=['context', 'entity_name', 'entity_id', 'masked_description', 'unmasked_description'])
        for context in input_x:
            # Replace the entity name in the context with the most popular entity title
            context_with_entity = context.replace(entity_name, entity_most_popular[0])

            # Use CPE and CME to generate the masked and unmasked descriptions
            masked_desc = CME_model.predict(context_with_entity)
            unmasked_desc = CPE_model.predict(context_with_entity)

            # Add the context, entity name, entity ID, masked description, and unmasked description to the dataframe
            df_most_popular_entity = df_most_popular_entity.append({'context': context, 'entity_name': entity_name,
                                                                    'entity_id': entity_most_popular[0],
                                                                    'masked_description': masked_desc,
                                                                    'unmasked_description': unmasked_desc},
                                                                   ignore_index=True)

        # Create a dataframe with the context, entity name, entity ID, masked description, and unmasked description
        # for the second most popular entity
        df_second_most_popular_entity = pd.DataFrame(
            columns=['context', 'entity_name', 'entity_id', 'masked_description', 'unmasked_description'])
        for context in input_x:
            # Replace the entity name in the context with the second most popular entity title
            context_with_entity = context.replace(entity_name, entity_second_most_popular[0])

            # Use CPE and CME to generate the masked and unmasked descriptions
            masked_desc = CME_model.predict(context_with_entity)
            unmasked_desc = CPE_model.predict(context_with_entity)

            # Add the context, entity name, entity ID, masked description, and unmasked description to the dataframe
            df_second_most_popular_entity = df_second_most_popular_entity.append(
                {'context': context, 'entity_name': entity_name,
                 'entity_id': entity_second_most_popular[0],
                 'masked_description': masked_desc,
                 'unmasked_description': unmasked_desc}, ignore_index=True)

        # Add the dataframes to the final dataframes
        df_most_popular = pd.concat([df_most_popular, df_most_popular_entity], ignore_index=True)
        df_second_most_popular = pd.concat([df_second_most_popular, df_second_most_popular_entity], ignore_index=True)

    # Write the dataframes to file
    df_most_popular.to_csv(output_file + '_most_popular.csv', sep=delimiter, index=False)
    df_second_most_popular.to_csv(output_file + '_second_most_popular.csv', sep=delimiter, index=False)
