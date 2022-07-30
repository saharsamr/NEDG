from pymongo import MongoClient
from nltk import tokenize
import datasets
from tqdm import tqdm
import logging


def import_to_mongo():

    logging.info("Connecting to MongoDB")
    client = MongoClient('localhost', 27017)
    db = client.kilt_wikipedia
    collection = db.articles

    logging.info("Importing data to MongoDB")
    data = datasets.load_dataset("kilt_wikipedia", streaming=True, split='full')
    for d in tqdm(data):

        new_anchor = {'paragraph_id': d['anchors']['paragraph_id'], 'start': d['anchors']['start'],
                      'end': d['anchors']['end'], 'text': d['anchors']['text'],
                      'wikipedia_title': d['anchors']['wikipedia_title'], 'wikipedia_id': d['anchors']['wikipedia_id']}
        new_record = {'wikipedia_id': d['wikipedia_id'], 'wikipedia_title': d['wikipedia_title'], 'text': d['text'],
                      'anchors': new_anchor}

        collection.insert_one(new_record)

    logging.info("Creating indices")
    collection.create_index([('wikipedia_id', 1)])


def create_dataset():

    client = MongoClient('localhost', 27017)
    db = client.kilt_wikipedia
    collection = db.articles

    with open('data.csv', 'w+') as f:

        logging.info("Creating dataset from knowledgebase")
        failed_counts = 0
        for doc in tqdm(collection.find()):

            paragraphs = doc['text']['paragraph']
            anchors = doc['anchors']

            for start, end, p_id, text, wiki_id in zip(
                    anchors['start'], anchors['end'], anchors['paragraph_id'], anchors['text'], anchors['wikipedia_id']
            ):

                context = paragraphs[p_id]
                context = context.replace('\1', '').replace('\n', '')
                context = context[:start] + '<NE> ' + context[start:end] + ' </NE>' + context[end:]

                # INFO: extract first sentence as the description
                try:
                    description = collection.find_one(
                        {'wikipedia_id': wiki_id})['text']['paragraph'][1].replace('\1', '').replace('\n', '')
                    description = tokenize.sent_tokenize(description)[0]
                except Exception as e:
                    failed_counts += 1
                    continue

                text = text.replace('\1', '').replace('\n', '')

                f.write(f'{text}\1{context}\1{description}\n')

        logging.error("Failed to find {} documents".format(failed_counts))
