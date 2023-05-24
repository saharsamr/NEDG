from tqdm import tqdm
import pymongo
import re
import urllib.parse
from collections import defaultdict
from pymongo import InsertOne, DeleteMany, ReplaceOne, UpdateOne
from config import MONGODB_LINK, MONGODB_PORT, MONGODB_DATABASE, \
    MONGODB_COLLECTION, MONGODB_READ_BATCH_SIZE, MONGODB_WRITE_BATCH_SIZE, \
    MONGODB_PASSWORD, MONGODB_USERNAME


client = pymongo.MongoClient(f'{MONGODB_LINK}:{MONGODB_PORT}/', username=MONGODB_USERNAME, password=MONGODB_PASSWORD)
db = client[MONGODB_DATABASE]
collection = db[MONGODB_COLLECTION]

total_documents = collection.count_documents({})
documents_cursor = collection.find(batch_size=MONGODB_READ_BATCH_SIZE)
updates = []

for document in tqdm(documents_cursor, total=total_documents):

    dictionary = defaultdict(list)
    context = document["text"]
    paragraphs = context.split('\n')

    for i, par in enumerate(paragraphs):
        links = re.findall(r'&lt;a href=(.*?)&gt;', par)
        for link in links:
            decoded_link = urllib.parse.unquote(link)
            dictionary[decoded_link].append(i)
    updates.append(UpdateOne({'_id': document['_id']}, {'$set': {"anchors": dictionary}}))

    if len(updates) > MONGODB_WRITE_BATCH_SIZE:
        collection.bulk_write(updates)
        updates = []
