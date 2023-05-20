from pymongo import InsertOne, DeleteMany, ReplaceOne, UpdateOne
from tqdm import tqdm
from collections import defaultdict

# Connect to MongoDB
client = pymongo.MongoClient('mongodb://localhost:27017/', username='user', password='pass')
# Creating a database
db = client['wikipedia']

# Creating a collection
collection = db['dump']

# Set batch size
batch_size = 4000

# Get total number of documents
total_documents = collection.count_documents({})

# Process documents in batches
anchor_to_ids = defaultdict(list)
for i in tqdm(range(0, total_documents, batch_size)):
    # Iterate over all documents in the collection
    documents = collection.find().skip(i).limit(batch_size)
    for doc in tqdm(documents):
        anchors = doc['anchors']
        id = doc['_id']
        # Iterate over all anchors
        for anchor, paragraph_index in anchors:
            # removing the "" from beginning and the end
            anchor = anchor[1:-1]
            # updating the anchor to ids dict
            anchor_to_ids[anchor].append(id)

# Set batch size
batch_size = 2000
for i in tqdm(range(0, total_documents, batch_size)):
    documents = collection.find().skip(i).limit(batch_size)
    updates = []
    for document in tqdm(documents):
        updates.append(UpdateOne({'_id': document['_id']}, {'$set': {"anchor_to_wikipage_ids": anchor_to_ids[document['title']]}}))

    collection.bulk_write(updates)
# Find all documents that have this anchor as a name entity
# matching_docs = collection.find({'title': anchor})
# for matching_doc in matching_docs:
#     matching_docs['paragraph_anchor_index'] = doc['id']
#     collection.save(matching_docs)

