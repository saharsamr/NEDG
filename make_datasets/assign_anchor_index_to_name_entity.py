from pymongo import InsertOne, DeleteMany, ReplaceOne, UpdateOne
from tqdm import tqdm
from collections import defaultdict
# mycode.py
from config import MONGODB_LINK, MONGODB_PORT, MONGODB_DATABASE, MONGODB_COLLECTION, MONGODB_BATCH_SIZE
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient(MONGODB_LINK, MONGODB_PORT)
db = client[MONGODB_DATABASE]
collection = db[MONGODB_COLLECTION]

# Use the collection object to interact with the database
# Connect to MongoDB
# client = pymongo.MongoClient('mongodb://localhost:27017/', username='user', password='pass')
# Creating a database
# db = client['wikipedia']

# Creating a collection
# collection = db['dump']

# Set batch size
batch_size = 2 * MONGODB_BATCH_SIZE

# Get total number of documents
total_documents = collection.count_documents({})
# Create a cursor to iterate over documents in batches
documents_cursor = collection.find(batch_size=batch_size)

# Process documents in batches
anchor_to_ids = defaultdict(list)
for doc in tqdm(documents_cursor, total=total_documents):
    anchors = doc['anchors']
    id = doc['_id']
    # for i in tqdm(range(0, total_documents, batch_size)):
    # Iterate over all documents in the collection
    # documents = collection.find().skip(i).limit(batch_size)
    # for doc in tqdm(documents):
    #     anchors = doc['anchors']
    #     id = doc['_id']
    # Iterate over all anchors
    for anchor, paragraph_index in anchors.items():
        # removing the "" from beginning and the end
        anchor = anchor[1:-1]
        # updating the anchor to ids dict
        anchor_to_ids[anchor].append(id)
# Close the cursor
documents_cursor.close()

# Set batch size
batch_size = MONGODB_BATCH_SIZE
# for i in tqdm(range(0, total_documents, batch_size)):
#     documents = collection.find().skip(i).limit(batch_size)
j = 0
updates = []
# for document in tqdm(documents):
for anchor, ids in anchor_to_ids.items():
    updates.append(UpdateOne({'title': anchor}, {'$set': {"anchor_to_wikipage_ids": ids}}))
    j = j + 1
    if j > batch_size:
        j = 0
        collection.bulk_write(updates)
        updates = []
# Find all documents that have this anchor as a name entity
# matching_docs = collection.find({'title': anchor})
# for matching_doc in matching_docs:
#     matching_docs['paragraph_anchor_index'] = doc['id']
#     collection.save(matching_docs)
