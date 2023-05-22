from pymongo import InsertOne, DeleteMany, ReplaceOne, UpdateOne
from tqdm import tqdm
import requests
from collections import defaultdict
from config import MONGODB_LINK, MONGODB_PORT, MONGODB_DATABASE, \
    MONGODB_COLLECTION, MONGODB_READ_BATCH_SIZE, MONGODB_WRITE_BATCH_SIZE, \
    MONGODB_PASSWORD, MONGODB_USERNAME
from pymongo import MongoClient
import pickle


misses = []


def batch_get_requests(article_titles):

    batch_responses = defaultdict(dict)
    url = 'https://www.wikidata.org/w/api.php?action=wbgetentities&sites=enwiki&titles='
    titles = '|'.join(article_titles)
    r = requests.get(url + titles + '&format=json')
    r = r.json()
    entities = r['entities']
    for entity_id, entity_value in entities.items():
        if 'missing' in entity_value:
            misses.append(entity_value['title'])
        else:
            if entity_value['labels']['en']['value'] not in article_titles:
                print('== wrong label ==')
            else:
                batch_responses[entity_value['labels']['en']['value']] = {
                    'wikidata_id': entity_id,
                    'description': entity_value['descriptions']
                }
    return batch_responses


print("Connecting to MongoDB...")
client = MongoClient(f'{MONGODB_LINK}:{MONGODB_PORT}/', username=MONGODB_USERNAME, password=MONGODB_PASSWORD)
db = client[MONGODB_DATABASE]
collection = db[MONGODB_COLLECTION]

documents_cursor = collection.find({'context_ids': {'$exists': True}}, batch_size=MONGODB_READ_BATCH_SIZE)
updates, titles = [], []
print("Scanning documents...")
total_count = collection.count_documents({'context_ids': {'$exists': True}})
for doc in tqdm(documents_cursor, total=total_count):

    if len(titles) > MONGODB_WRITE_BATCH_SIZE:
        wikidata_info = batch_get_requests(titles)
        for title, info in wikidata_info.items():
            updates.append(UpdateOne({'title': title}, {'$set': {"wikidata_info": info}}))

        collection.bulk_write(updates)
        updates, titles = [], []

    titles.append(doc['title'])

documents_cursor.close()

print(len(misses))
with open(f'./misses.pkl', 'wb') as f:
    pickle.dump(misses, f)
