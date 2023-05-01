

import pymongo
import re
import urllib.parse
from collections import defaultdict
# Establishing a connection with MongoDB
client = pymongo.MongoClient('mongodb://localhost:27017/', username = 'user', password = 'pass')

# Creating a database
db = client['wikipedia']

# Creating a collection
collection = db['dump']

dictionary = defaultdict(list)
# open each json
for document in collection.find():
    # extract the text
    context = document["text"]
    paragraphs = context.split('\n')
    # find anchors and add them to list
    for i, par in enumerate(paragraphs):
        links = re.findall(r'<a href=(.*?)&gl', par)
        for link in links:
            decoded_link = urllib.parse.unquote(link)
            dictionary[decoded_link].append(i)
    # link_list.extend(links)
    # dictionary = list(set(links))  # remove duplicates

    # decode the links
    # decoded_anchors = [urllib.parse.unquote(link) for link in dictionary]
    # add the list to the mongo json
    collection.update_one({"_id": document["_id"]}, {"$set": {"anchors": dictionary}})
