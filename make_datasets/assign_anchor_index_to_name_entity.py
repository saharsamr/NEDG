from pymongo import InsertOne, DeleteMany, ReplaceOne, UpdateOne
from tqdm import tqdm
from collections import defaultdict

from config import MONGODB_LINK, MONGODB_PORT, MONGODB_DATABASE, \
    MONGODB_COLLECTION, MONGODB_READ_BATCH_SIZE, MONGODB_WRITE_BATCH_SIZE, \
    MONGODB_PASSWORD, MONGODB_USERNAME
from pymongo import MongoClient


client = MongoClient(f'{MONGODB_LINK}:{MONGODB_PORT}/', username=MONGODB_USERNAME, password=MONGODB_PASSWORD)
db = client[MONGODB_DATABASE]
collection = db[MONGODB_COLLECTION]

total_documents = collection.count_documents({})

documents_cursor = collection.find(batch_size=MONGODB_READ_BATCH_SIZE)
anchor_to_ids = defaultdict(list)

for doc in tqdm(documents_cursor, total=total_documents):
    anchors = doc['anchors']
    id = doc['_id']
    for anchor, paragraph_index in anchors.items():
        # removing the "" from beginning and the end
        anchor = anchor[1:-1]
        anchor_to_ids[anchor].append(id)

documents_cursor.close()

updates = []
for anchor, ids in anchor_to_ids.items():
    updates.append(UpdateOne({'title': anchor}, {'$set': {"context_ids": ids}}))
    if len(updates) > MONGODB_WRITE_BATCH_SIZE:
        collection.bulk_write(updates)
        updates = []

