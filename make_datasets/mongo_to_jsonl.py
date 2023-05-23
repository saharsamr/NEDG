from pymongo import MongoClient
from collections import defaultdict
from tqdm import tqdm
import urllib.parse
import nltk
import json
import re

from config import MONGODB_LINK, MONGODB_PORT, MONGODB_DATABASE, \
    MONGODB_COLLECTION, MONGODB_READ_BATCH_SIZE, MONGODB_PASSWORD, \
    MONGODB_USERNAME, MIN_CONTEXT_LENGTH, MAX_ENTITY_NAME_LENGTH, WIKI_DUMP_JSONL_PATH


def remove_special_characters(text):

    text = re.sub(r'^=+\s+.*\s+=+$', '', text)
    text = re.sub(r'[^a-z|A-Z| |\.|\!|\?|0-9|(|)|-|\'|`|\"|<|>|\/|]', '', text)
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

    if 'BULLET' in context:
        return True
    return len(context.split()) < MIN_CONTEXT_LENGTH


def tag_entity_in_context_and_clean(context, entity_name):

    anchors = re.findall(r'&lt;a href=(.*?)&gt;(.*?)&lt;/a&gt;', context)
    for (anchor_link, anchor_name) in anchors:
        if urllib.parse.unquote(anchor_link)[1:-1] == entity_name:
            context = context.replace(f'&lt;a href={anchor_link}&gt;{anchor_name}&lt;/a&gt;',
                                      f'{"<NE>"+anchor_name+"</NE>"}')
        else:
            context = context.replace(f'&lt;a href={anchor_link}&gt;{anchor_name}&lt;/a&gt;', anchor_name)

    return context, '<NE>' in context and '</NE>' in context


def has_long_entity_name(entity_name):
    return len(entity_name.split()) > MAX_ENTITY_NAME_LENGTH


def aggregates_context_ids_and_query(mongo_collection, batch_data):

    context_ids_list = []
    for title, doc_info in batch_data.items():
        context_ids_list.extend(doc_info['context_ids'])

    contexts_docs = mongo_collection.find(
        {'_id': {'$in': list(set(context_ids_list))}},
        {'text': 1, 'anchors': 1, '_id': 1}
    )
    context_id_to_doc_map = {}
    for context_doc in contexts_docs:
        context_id_to_doc_map[context_doc['_id']] = context_doc

    title_to_contexts_map = defaultdict(dict)
    for title, doc_info in batch_data.items():
        for context_id in doc_info['context_ids']:
            title_to_contexts_map[title][context_id] = context_id_to_doc_map[context_id]

    return title_to_contexts_map


def get_batch_contexts(mongo_collection, docs_batch):

    title_to_contexts_map = aggregates_context_ids_and_query(mongo_collection, docs_batch)

    for title, doc_info in docs_batch.items():
        contexts = []
        linked_wiki_pages = title_to_contexts_map[title]
        for page_id, page_info in linked_wiki_pages.items():
            anchors = page_info['anchors']
            for anchor_name, par_ids in anchors.items():
                if anchor_name[1:-1] == title:
                    paragraphs = page_info['text'].split('\n')
                    for par_id in par_ids:
                        context = paragraphs[par_id]
                        if is_proper_contexts(context):
                            context, is_tagged = tag_entity_in_context_and_clean(context, title)
                            if is_tagged:
                                context = remove_special_characters(context)
                                contexts.append(context)
                            else:
                                print('entity not found!')
                    break

        docs_batch[title]['contexts'] = contexts

    return docs_batch


print("Connecting to MongoDB...")
client = MongoClient(f'{MONGODB_LINK}:{MONGODB_PORT}/', username=MONGODB_USERNAME, password=MONGODB_PASSWORD)
db = client[MONGODB_DATABASE]
collection = db[MONGODB_COLLECTION]

print("Scanning documents...")
documents_cursor = collection.find({'context_ids': {'$exists': True}}, batch_size=MONGODB_READ_BATCH_SIZE)
total_count = collection.count_documents({'context_ids': {'$exists': True}})


with open(WIKI_DUMP_JSONL_PATH, 'w+') as f:

    docs_batch = {}
    long_name_entities_count = 0
    for doc in tqdm(documents_cursor, total=total_count):
        if not has_long_entity_name(doc['title']):
            doc_data = {
                'wikipedia_title': doc['title'].replace('\1', ''),
                'wikipedia_id': doc['id'],
                'wikipedia_description': get_wikipedia_description(doc['text']),
                'wikidata_description': get_wikidata_description(doc.get('wikidata_info')),
                'context_ids': doc['context_ids']
            }
            docs_batch[doc['title']] = doc_data
        else:
            long_name_entities_count += 1

        if len(docs_batch) == MONGODB_READ_BATCH_SIZE:
            docs_batch = get_batch_contexts(collection, docs_batch)
            for doc_title, doc_info in docs_batch.items():
                print('writing to jsonl file.')
                if len(doc_data['contexts']):
                    f.write(json.dumps(doc_data)+'\n')

    print(f'found {long_name_entities_count} entities which their name '
          f'has more than {MAX_ENTITY_NAME_LENGTH} words or tokens.')
