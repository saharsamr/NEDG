import os
import json
from pymongo import MongoClient

# Establishing a connection with MongoDB
client = MongoClient('mongodb://localhost:27017/')

# Creating a database
db = client['mydatabase']

# Creating a collection
collection = db['mycollection']

# Path to the directory containing the JSONL files
path = '../wikipedia/text/AA'

# Loop through each file in the directory
for filename in os.listdir(path):
    # Path to the JSONL file
    filepath = os.path.join(path, filename)

    # Load the JSONL data from the file
    with open(filepath, 'r') as f:
        for line in f:
            # Parse each JSON object from the line
            data = json.loads(line)

            # Insert the JSON object into the MongoDB collection
            collection.insert_one(data)