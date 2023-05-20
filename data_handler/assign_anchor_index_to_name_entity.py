from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('localhost', 27017)
db = client['my_database']
collection = db['my_collection']

# Iterate over all documents in the collection
for doc in collection.find():
    entity = doc['entity']
    contexts = doc['contexts']
    anchors = doc['anchors']

    # Iterate over all anchors
    for anchor, i in anchors:
        # Find all documents that have this anchor as a name entity
        matching_docs = collection.find({'title': anchor})
        for matching_doc in matching_docs:
            matching_docs['paragraph_anchor_index'] = doc['id']
            collection.save(matching_docs)

