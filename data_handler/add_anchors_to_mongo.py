

import pymongo
import re
import urllib.parse

# Establishing a connection with MongoDB
client = pymongo.MongoClient('mongodb://localhost:27017/', username = 'user', password = 'pass')

# Creating a database
db = client['wikipedia']

# Creating a collection
collection = db['dump']

link_list = []
# open each json
for document in collection.find():
    # extract the context
    context = document["context"]
    # find anchors and add them to list
    links = re.findall(r'<a href=(.*?)&gl', context)
    # link_list.extend(links)
    link_list = list(set(links))  # remove duplicates
    # decode the links
    decoded_anchors = [urllib.parse.unquote(link) for link in link_list]
    # add the list to the mongo json
    collection.update_one({"_id": document["_id"]}, {"$set": {"anchors": decoded_anchors}})
