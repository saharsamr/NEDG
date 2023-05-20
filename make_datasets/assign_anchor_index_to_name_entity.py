from pymongo import InsertOne, DeleteMany, ReplaceOne, UpdateOne
from tqdm import tqdm
from collections import defaultdict
# mycode.py
from config import MONGODB_LINK, MONGODB_PORT, MONGODB_DATABASE, MONGODB_COLLECTION, MONGODB_READ_BATCH_SIZE,MONGODB_WRITE_BATCH_SIZE, MONGODB_PASSWORD, MONGODB_USERNAME
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient(MONGODB_LINK, username=MONGODB_USERNAME, password=MONGODB_PASSWORD)
db = client[MONGODB_DATABASE]
collection = db[MONGODB_COLLECTION]

# Set batch size
batch_size = MONGODB_READ_BATCH_SIZE

# Get total number of documents
total_documents = collection.count_documents({})
# Create a cursor to iterate over documents in batches
documents_cursor = collection.find(batch_size=batch_size)

# Process documents in batches
anchor_to_ids = defaultdict(list)
for doc in tqdm(documents_cursor, total=total_documents):
    anchors = doc['anchors']
    id = doc['_id']
    for anchor, paragraph_index in anchors.items():
        # removing the "" from beginning and the end
        anchor = anchor[1:-1]
        # updating the anchor to ids dict
        anchor_to_ids[anchor].append(id)
# Close the cursor
documents_cursor.close()

# Set batch size
batch_size = MONGODB_WRITE_BATCH_SIZE
j = 0
updates = []
for anchor, ids in anchor_to_ids.items():
    updates.append(UpdateOne({'title': anchor}, {'$set': {"anchor_to_wikipage_ids": ids}}))
    if len(updates) > batch_size:
        collection.bulk_write(updates)
        updates = []

