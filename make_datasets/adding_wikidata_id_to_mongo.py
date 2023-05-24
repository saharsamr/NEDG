from pymongo import MongoClient
from pymongo import InsertOne, DeleteMany, ReplaceOne, UpdateOne

import requests
from tqdm import tqdm
from collections import defaultdict

from multiprocessing import Pool
import os

from config import MONGODB_LINK, MONGODB_PORT, MONGODB_DATABASE, \
    MONGODB_COLLECTION, MONGODB_READ_BATCH_SIZE, MONGODB_WRITE_BATCH_SIZE, \
    MONGODB_PASSWORD, MONGODB_USERNAME


def batch_get_requests(article_titles):

    url = f'https://www.wikidata.org/w/api.php?action=wbgetentities&sites=enwiki&titles={"|".join(article_titles)}&format=json'
    res = requests.get(url)

    if res.status_code != 200:
        print(res.status_code)
        return None

    try:
        entities = res.json()['entities']
    except:
        return None

    article_titles_lower = [t.lower() for t in article_titles]
    batch_responses = defaultdict(dict)
    for entity_id, entity_value in entities.items():

        if 'missing' in entity_value:
            pass
        else:
            try:
                if entity_value['labels']['en']['value'].lower() in article_titles_lower:
                    batch_responses[entity_value['labels']['en']['value'].lower()] = entity_value
                else:
                    for alias in entity_value['aliases']['en']:
                        if alias['value'].lower() in article_titles_lower:
                            batch_responses[alias['value'].lower()] = entity_value
                            break
            except:
                pass

    return batch_responses


total_updates = 0
def process_batch(titles_and_map):
    global total_updates

    titles, title_id_map = titles_and_map

    wikidata_info = batch_get_requests(titles)
    updates = []
    if wikidata_info:
        for title, info in wikidata_info.items():
            updates.append(UpdateOne({'_id': title_id_map[title.lower()]}, {'$set': {"wikidata_info": info}}))
    if len(updates):
        collection.bulk_write(updates)
        total_updates += len(updates)
        print(total_updates)


print("Connecting to MongoDB...")
client = MongoClient(f'{MONGODB_LINK}:{MONGODB_PORT}/', username=MONGODB_USERNAME, password=MONGODB_PASSWORD)
db = client[MONGODB_DATABASE]
collection = db[MONGODB_COLLECTION]

pool = Pool(processes=os.cpu_count())

print("Scanning documents...")
documents_cursor = collection.find({'context_ids': {'$exists': True}, 'wikidata_info': {'$exists': False}},
                                   batch_size=MONGODB_READ_BATCH_SIZE)
total_count = collection.count_documents({'context_ids': {'$exists': True}, 'wikidata_info': {'$exists': False}})

updates, titles = [], []
title_id_map = {}
for doc in tqdm(documents_cursor, total=total_count):

    titles.append(doc['title'])
    title_id_map[doc['title'].lower()] = doc['_id']

pool.map(process_batch, [(titles[i:i+50], title_id_map) for i in range(0, len(titles), 50)])

pool.close()
pool.join()
documents_cursor.close()

