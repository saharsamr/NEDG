from pymongo import InsertOne, DeleteMany, ReplaceOne, UpdateOne
from tqdm import tqdm
import requests
from config import MONGODB_LINK, MONGODB_PORT, MONGODB_DATABASE, \
    MONGODB_COLLECTION, MONGODB_READ_BATCH_SIZE, MONGODB_WRITE_BATCH_SIZE, \
    MONGODB_PASSWORD, MONGODB_USERNAME
from pymongo import MongoClient


def get_wikidata_info(article_title):

    url = f'https://www.wikidata.org/w/api.php?action=wbgetentities&sites=enwiki&titles={article_title}&format=json'
    headers = {
        "accept": "application/json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        try:
            response = response.json()
            entities = response['entities']
            for entity_id, entity_info in entities.items():
                return {
                    'entity_id': entity_id,
                    'description': entity_info['descriptions'],
                }
        except:
            print(article_title, ': Description not found.')
    return None


print("Connecting to MongoDB...")
client = MongoClient(f'{MONGODB_LINK}:{MONGODB_PORT}/', username=MONGODB_USERNAME, password=MONGODB_PASSWORD)
db = client[MONGODB_DATABASE]
collection = db[MONGODB_COLLECTION]

documents_cursor = collection.find({'context_ids': {'$exists': True}}, batch_size=MONGODB_READ_BATCH_SIZE)
updates = []
print("Scanning documents...")
for doc in tqdm(documents_cursor):

    title = doc['title']
    wikidata_info = get_wikidata_info(title)
    if wikidata_info:
        updates.append(UpdateOne({'_id': doc['_id']}, {'$set': {"wikidata_info": wikidata_info}}))

    if len(updates) > MONGODB_WRITE_BATCH_SIZE:
        collection.bulk_write(updates)
        updates = []

documents_cursor.close()
