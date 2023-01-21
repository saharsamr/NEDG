import pickle

from pymongo import MongoClient
import datasets
import random

from tqdm import tqdm
from collections import defaultdict
import logging

<<<<<<< Updated upstream
import json
import csv
import re

from utils.data_structures import is_array_of_empty_strings
from config import MIN_CONTEXT_LEN, MAX_CONTEXT_NUM, MASKING_PROBABILITY
=======
from utils.arrays import is_array_of_empty_strings
>>>>>>> Stashed changes


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
<<<<<<< Updated upstream

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

=======

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

>>>>>>> Stashed changes
                    f.write(f'{text}\1{context}\1{description}\n')

        logging.error("Failed to find {} documents".format(failed_counts))


def mask_ne(string, probability=MASKING_PROBABILITY):

    starts, ends = re.finditer('<NE>', string), re.finditer('</NE>', string)
    new_string, pre_end_idx = '', 0
    for start, end in zip(starts, ends):
        new_string += string[pre_end_idx:start.start()]
        entity = string[start.start():end.end()]
        tokens = entity.split(' ')
        new_tokens = []
        for i, t in enumerate(tokens):
            if random.random() > probability or i == 0 or i == len(tokens)-1:
                new_tokens.append(t)
            else:
                new_tokens.append('<MASK>')
        new_string += ' '.join(new_tokens)
        pre_end_idx = end.end()
    new_string += string[pre_end_idx:]

    return new_string


def make_jsonl_to_csv(file_path, concat_contexts=False):

    count_no_desc, count_with_desc = 0, 0
    with open(file_path, 'r') as f:
        with \
          open(f'{file_path[:-5]}_ne_with_context.csv', 'w') as ne_with_context, \
          open(f'{file_path[:-5]}_ne_no_context.csv', 'w') as ne_no_context, \
          open(f'{file_path[:-5]}_masked_ne_with_context.csv', 'w') as masked_ne_with_context:

            ne_with_context_writer = csv.writer(ne_with_context, delimiter='\1')
            ne_no_context_writer = csv.writer(ne_no_context, delimiter='\1')
            masked_ne_with_context_writer = csv.writer(masked_ne_with_context, delimiter='\1')

            for line in f:
                data = json.loads(line)
                key, value = list(data.keys())[0], list(data.values())[0]
                if value['description'] == '':
                    count_no_desc += 1
                    continue
                count_with_desc += 1

                contexts = [context.replace('\1', '') for context in value['contexts'] if len(context.split())]
                title, description = value['label'].replace('\1', ''), value['description'].replace('\1', '')

                if len(contexts) > MAX_CONTEXT_NUM:
                    contexts = random.sample(contexts, MAX_CONTEXT_NUM)

                if concat_contexts:
                    contexts = '<CNTXT>' + '</CNTXT><CNTXT>'.join(contexts) + '</CNTXT>'
                    ne_with_context_writer.writerow([title, contexts, description])
                    ne_no_context_writer.writerow([title, title, description])
                    masked_ne_with_context_writer.writerow([title, mask_ne(contexts), description])

                else:
                    for context in contexts:
                        ne_with_context_writer.writerow([title, context, description])
                        ne_no_context_writer.writerow([title, title, description])
                        masked_ne_with_context_writer.writerow([title, mask_ne(context), description])

    print(f'No description: {count_no_desc}, with description: {count_with_desc}')
