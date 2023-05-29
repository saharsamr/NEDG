import os
import json
from pymongo import MongoClient
from tqdm import tqdm
from config import MONGODB_LINK, MONGODB_PORT, MONGODB_DATABASE, \
    MONGODB_COLLECTION, MONGODB_PASSWORD, MONGODB_USERNAME, WIKI_JSONS_PATH


client = MongoClient(f'{MONGODB_LINK}:{MONGODB_PORT}', username=MONGODB_USERNAME, password=MONGODB_PASSWORD)
db = client[MONGODB_DATABASE]
collection = db[MONGODB_COLLECTION]


for root, dirs, files in tqdm(os.walk(WIKI_JSONS_PATH)):

    for direc in tqdm(dirs):
        dir_path = os.path.join(root, direc)
        for filename in tqdm(os.listdir(dir_path)):
            filepath = os.path.join(dir_path, filename)
            file_articles = []

            with open(filepath, 'r') as f:
                for line in f.readlines():
                    data = json.loads(line)
                    file_articles.append(data)
            collection.insert_many(file_articles)
