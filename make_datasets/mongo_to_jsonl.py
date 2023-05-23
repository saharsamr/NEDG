from pymongo import MongoClient
from tqdm import tqdm
import nltk

from config import MONGODB_LINK, MONGODB_PORT, MONGODB_DATABASE, \
    MONGODB_COLLECTION, MONGODB_READ_BATCH_SIZE, MONGODB_PASSWORD, \
    MONGODB_USERNAME, MIN_CONTEXT_LENGTH, MAX_ENTITY_NAME_LENGTH


def remove_special_characters(text):

    special_chars_regexp = '[^a-z|A-Z| |\.|\!|\?|0-9|(|)|-|\'|`|\"]'
    text = text.replace(special_chars_regexp, '', regex=True)
    text = text.replace('^=+\s+.*\s+=+$', '', regex=True)
    text = text.replace('\1', '')
    return text


def get_wikipedia_description(text):

    text = text.replace('\n', '')
    sentences = nltk.sent_tokenize(text)
    if not len(sentences):
        return None
    return remove_special_characters(sentences[0])


def get_wikidata_description(wikidata_info):

    if wikidata_info and wikidata_info.get('descriptions'):
        description = wikidata_info['descriptions'].get('en')
        if description:
            return remove_special_characters(description['value'])
    return None


def is_proper_contexts(context):
    if context.contains('BULLET'):
        return True
    return len(context.split()) < MIN_CONTEXT_LENGTH


def has_long_entity_name(entity_name):
    return len(entity_name.split()) > MAX_ENTITY_NAME_LENGTH


def get_contexts(mongo_collection, title, context_ids):

    contexts = []
    linked_wiki_pages = mongo_collection.find({'_id': {'$in': context_ids}})
    for page in linked_wiki_pages:
        anchors = page['anchors']
        for anchor_name, par_ids in anchors.items():
            if anchor_name[1:-1] == title:
                paragraphs = page['text'].split('\n')
                for par_id in par_ids:
                    context = remove_special_characters(paragraphs[par_id])
                    if is_proper_contexts(context):
                        contexts.append(context)
                break

    return contexts


print("Connecting to MongoDB...")
client = MongoClient(f'{MONGODB_LINK}:{MONGODB_PORT}/', username=MONGODB_USERNAME, password=MONGODB_PASSWORD)
db = client[MONGODB_DATABASE]
collection = db[MONGODB_COLLECTION]

print("Scanning documents...")
documents_cursor = collection.find({'context_ids': {'$exists': True}, 'wikidata_info': {'$exists': False}},
                                   batch_size=MONGODB_READ_BATCH_SIZE)
total_count = collection.count_documents({'context_ids': {'$exists': True}, 'wikidata_info': {'$exists': False}})

with open('wiki_dump.jsonl', 'w+') as f:
    for doc in tqdm(documents_cursor, total=total_count):
        if not has_long_entity_name(doc['title']):
            doc_data = {
                'wikipedia_title': doc['title'].replace('\1', ''),
                'wikipedia_id': doc['id'],
                'wikipedia_description': get_wikipedia_description(doc['text']),
                'wikidata_description': get_wikidata_description(doc.get('wikidata_info')),
                'contexts': get_contexts(collection, doc['title'], doc['context_ids'])
            }