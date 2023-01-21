from pymongo import MongoClient
from tqdm import tqdm
import logging


def test_anchor_existence():

    client = MongoClient('localhost', 27017)
    db = client.kilt_wikipedia
    collection = db.articles

    logging.info("Start Testing Anchors' Existence")
    failed_counts = 0
    for doc in tqdm(collection.find()):

        anchors = doc['anchors']
        for text, wiki_id in zip(anchors['text'], anchors['wikipedia_id']):
            try:
                description = collection.find_one({'wikipedia_id': wiki_id})
                if not description:
                    'failed, but none'
            except Exception as e:
                failed_counts += 1
                print(f'anchor with wikipediaID: {wiki_id} and text: {text} does not have any records available')\

    logging.error("Failed to find {} documents".format(failed_counts))
