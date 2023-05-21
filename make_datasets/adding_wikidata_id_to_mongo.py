from pymongo import InsertOne, DeleteMany, ReplaceOne, UpdateOne
from tqdm import tqdm
from collections import defaultdict
import requests
from config import MONGODB_LINK, MONGODB_PORT, MONGODB_DATABASE, \
    MONGODB_COLLECTION, MONGODB_READ_BATCH_SIZE, MONGODB_WRITE_BATCH_SIZE, \
    MONGODB_PASSWORD, MONGODB_USERNAME
from pymongo import MongoClient


def get_wikidata_id(article_title):
    url = f"https://query.wikidata.org/sparql?query=SELECT%20%3Fitem%20WHERE%20%7B%0A%20%20%20%20%3Fsitelink%20schema" \
          f"%3Aabout%20%3Fitem%20.%0A%20%20%20%20%3Fsitelink%20schema%3AinLanguage%20%22en%22%20.%0A%20%20%20" \
          f"%20FILTER%20(STRSTARTS(str(%3Fsitelink)%2C%20%22https%3A%2F%2Fen.wikipedia.org%2Fwiki%2F" \
          f"{article_title}%22))%0A%7D"

    headers = {
        "Accept": "application/json"
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    data = response.json()
    bindings = data["results"]["bindings"]
    if len(bindings) == 0:
        return None
    wikidata_id = bindings[0]["item"]["value"].split("/")[-1]
    return wikidata_id


print("Connecting to MongoDB...")
client = MongoClient(f'{MONGODB_LINK}:{MONGODB_PORT}/', username=MONGODB_USERNAME, password=MONGODB_PASSWORD)
db = client[MONGODB_DATABASE]
collection = db[MONGODB_COLLECTION]

total_documents = collection.count_documents({})

documents_cursor = collection.find(batch_size=MONGODB_READ_BATCH_SIZE)
anchor_to_ids = defaultdict(list)
updates = []
print("Scanning documents...")
for doc in tqdm(documents_cursor, total=total_documents):
    if len(doc['context_ids']) > 0:
        title = doc['title']
        wikidata_id = get_wikidata_id(title)
        if wikidata_id:
            updates.append(UpdateOne({'_id': doc['_id']}, {'$set': {"wikidata_id": wikidata_id}}))

        if len(updates) > MONGODB_WRITE_BATCH_SIZE:
            collection.bulk_write(updates)
            updates = []

documents_cursor.close()
