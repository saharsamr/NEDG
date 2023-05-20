from pymongo import MongoClient
from tqdm import tqdm
from collections import defaultdict

# Connect to MongoDB
client = MongoClient('localhost', 27017)
db = client['my_database']
collection = db['my_collection']

# Set batch size
batch_size = 2000

# Get total number of documents
total_documents = collection.count_documents({})

# Process documents in batches
for i in tqdm(range(0, total_documents, batch_size)):
    # Iterate over all documents in the collection
    documents = collection.find().skip(i).limit(batch_size)
    updates = []
    for doc in tqdm(documents):
        entity = doc['entity']
        anchors = doc['anchors']

        # Iterate over all anchors
        for anchor, i in anchors:
            # Find all documents that have this anchor as a name entity
            matching_docs = collection.find({'title': anchor})
            for matching_doc in matching_docs:
                matching_docs['paragraph_anchor_index'] = doc['id']
                collection.save(matching_docs)

