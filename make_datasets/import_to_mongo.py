import os
import json
from pymongo import MongoClient
from tqdm import tqdm

# Establishing a connection with MongoDB
client = MongoClient('mongodb://localhost:27017/', username = 'user', password = 'pass')

# Creating a database
db = client['wikipedia']

# Creating a collection
collection = db['dump']

# Path to the directory containing the JSONL files
path = '../../wikipedia/text'

# Loop through each file in the directory
for root, dirs, files in tqdm(os.walk(path)):
    for direc in tqdm(dirs):
        # Path to the current subdirectory
        dir_path = os.path.join(root, direc)
        # print(f'Opening folder: {dir_path}')
        # Loop through each file in the directory
        for filename in tqdm(os.listdir(dir_path)):
            # Path to the JSONL file
            filepath = os.path.join(dir_path, filename)
            file_articles = []
            # Load the JSONL data from the file
            with open(filepath, 'r') as f:
                for line in f.readlines():
                    # Parse each JSON object from the line
                    data = json.loads(line)
                    file_articles.append(data)
                    # Insert the JSON object into the MongoDB collection
            collection.insert_many(file_articles)
