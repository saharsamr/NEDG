import pickle

from pymongo import MongoClient
import datasets
from tqdm import tqdm
from collections import defaultdict
import logging

import json
import csv

from utils.data_structures import is_array_of_empty_strings


def import_to_mongo():

    logging.info("Connecting to MongoDB")
    client = MongoClient('localhost', 27017)
    db = client.kilt_wikipedia
    collection = db.articles

    logging.info("Importing data to MongoDB")
    data = datasets.load_dataset("kilt_wikipedia", streaming=True, split='full')
    categories = defaultdict(int)
    counter = 0
    for d in tqdm(data):

        if is_array_of_empty_strings(d['anchors']['wikipedia_id']):
            continue

        new_anchor = {'paragraph_id': d['anchors']['paragraph_id'], 'start': d['anchors']['start'],
                      'end': d['anchors']['end'], 'text': d['anchors']['text'],
                      'wikipedia_title': d['anchors']['wikipedia_title'], 'wikipedia_id': d['anchors']['wikipedia_id']}

        new_record = {
            'wikipedia_id': d['wikipedia_id'], 'wikipedia_title': d['wikipedia_title'], 'text': d['text'],
            'categories': d['categories'], 'wikidata_info': d['wikidata_info'], 'anchors': new_anchor
        }

        cats = d['categories'].split(',')
        for cat in cats:
            categories[cat] += 1

        counter += 1
        if counter % 1000:
            with open('categories.pkl', 'wb') as f:
                pickle.dump(categories, f)

        collection.insert_one(new_record)

    logging.info("Creating indices")
    collection.create_index([('wikipedia_id', 1)])

    with open('total_categories.pkl', 'wb') as f:
        pickle.dump(categories, f)


def create_dataset():

    client = MongoClient('localhost', 27017)
    db = client.kilt_wikipedia
    collection = db.articles

    with open('data.csv', 'w+') as f:

        logging.info("Creating dataset from knowledgebase")
        failed_counts = 0
        for doc in tqdm(collection.find()):

            paragraphs = doc['text']['paragraph']
            anchors = doc['anchors']

            if not is_array_of_empty_strings(anchors['wikipedia_id']):
                for start, end, p_id, text, wiki_id in zip(
                        anchors['start'], anchors['end'], anchors['paragraph_id'], anchors['text'], anchors['wikipedia_id']
                ):

                    context = paragraphs[p_id]
                    context = context.replace('\1', '').replace('\n', '')
                    context = context[:start] + '<NE> ' + context[start:end] + ' </NE>' + context[end:]

                    try:
                        description = collection.find_one(
                            {'wikipedia_id': wiki_id})['text']['paragraph'][1].replace('\1', '').replace('\n', '')
                    except Exception as e:
                        failed_counts += 1
                        continue

                    text = text.replace('\1', '').replace('\n', '')

                    f.write(f'{text}\1{context}\1{description}\n')

        logging.error("Failed to find {} documents".format(failed_counts))


def make_jsonl_to_csv(file_path):

    count_no_desc, count_with_desc = 0, 0
    with open(file_path, 'r') as f:
        with open(f'{file_path[:-5]}.csv', 'w') as g:
            writer = csv.writer(g, delimiter='\1')
            for line in f:
                data = json.loads(line)
                key, value = list(data.keys())[0], list(data.values())[0]
                if value['description'] == '':
                    count_no_desc += 1
                    continue
                count_with_desc += 1
                for context in value['contexts']:
                    title, context, description = \
                        value['label'].replace('\1', ''), context.replace('\1', ''), value['description'].replace('\1', '')
                    writer.writerow([title, context, description])

    print(f'No description: {count_no_desc}, with description: {count_with_desc}')
