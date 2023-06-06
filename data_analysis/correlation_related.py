
from tqdm import tqdm
from collections import defaultdict

from config import MONGODB_LINK, MONGODB_PORT, MONGODB_DATABASE, \
    MONGODB_COLLECTION, MONGODB_READ_BATCH_SIZE, MONGODB_WRITE_BATCH_SIZE, \
    MONGODB_PASSWORD, MONGODB_USERNAME
from pymongo import MongoClient
import pickle


def entity_popularity_check(path):
    print("Connecting to MongoDB...")
    client = MongoClient(f'{MONGODB_LINK}:{MONGODB_PORT}/', username=MONGODB_USERNAME, password=MONGODB_PASSWORD)
    db = client[MONGODB_DATABASE]
    collection = db[MONGODB_COLLECTION]

    total_documents = collection.count_documents({"context_id": {"$exists": True}})

    documents_cursor = collection.find({"context_id": {"$exists": True}}, batch_size=MONGODB_READ_BATCH_SIZE)
    title_to_popularity = defaultdict(int)

    total_documents = collection.count_documents({"context_id": {"$exists": True}})

    print("Scanning documents...")
    for doc in tqdm(documents_cursor, total=total_documents):
        context_ids = doc['context_ids']
        title = doc['title']
        title_to_popularity[title] = len(context_ids)

    # save the dictionary in the path
    print("Saving dictionary to file...")
    with open(path, 'wb') as file:
        pickle.dump(title_to_popularity, file)

    print("Dictionary saved successfully.")

