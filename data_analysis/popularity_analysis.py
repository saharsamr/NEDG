from tqdm import tqdm
from pymongo import MongoClient
from collections import defaultdict
import pickle

from make_datasets.config import MONGODB_LINK, MONGODB_PORT, MONGODB_DATABASE, \
    MONGODB_COLLECTION, MONGODB_READ_BATCH_SIZE, MONGODB_PASSWORD, MONGODB_USERNAME
from data_analysis.config import ENTITY_POPULARITY_PATH


def find_entity_popularity(path):

    print("Connecting to MongoDB...")
    client = MongoClient(f'{MONGODB_LINK}:{MONGODB_PORT}/', username=MONGODB_USERNAME, password=MONGODB_PASSWORD)
    db = client[MONGODB_DATABASE]
    collection = db[MONGODB_COLLECTION]

    documents_cursor = collection.find({"context_ids": {"$exists": True}}, batch_size=MONGODB_READ_BATCH_SIZE)
    title_to_popularity = defaultdict(int)

    total_documents = collection.count_documents({"context_ids": {"$exists": True}})
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


find_entity_popularity(ENTITY_POPULARITY_PATH)
