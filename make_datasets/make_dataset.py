import pickle

from datasets import load_metric
from pymongo import MongoClient
import datasets
import random

from tqdm import tqdm
from collections import defaultdict
import logging

import pandas as pd
import numpy as np
import json
import csv
import re

from utils.data_structures import is_array_of_empty_strings
from utils.metrics import compute_bertscores_from_file
from config import MIN_CONTEXT_LEN, MAX_CONTEXT_NUM, MASKING_PROBABILITY


def import_to_mongo():

    logging.info("Connecting to MongoDB")
    client = MongoClient('localhost', 27017)
    db = client.kilt_wikipedia
    collection = db.articles

    logging.info("Importing data to MongoDB")
    data = datasets.load_dataset("kilt_wikipedia", streaming=True, split='full')
    categories = defaultdict(int)
    counter = 0
    for d in tqdm(data):

        if is_array_of_empty_strings(d['anchors']['wikipedia_id']):
            continue

        new_anchor = {'paragraph_id': d['anchors']['paragraph_id'], 'start': d['anchors']['start'],
                      'end': d['anchors']['end'], 'text': d['anchors']['text'],
                      'wikipedia_title': d['anchors']['wikipedia_title'], 'wikipedia_id': d['anchors']['wikipedia_id']}

        new_record = {
            'wikipedia_id': d['wikipedia_id'], 'wikipedia_title': d['wikipedia_title'], 'text': d['text'],
            'categories': d['categories'], 'wikidata_info': d['wikidata_info'], 'anchors': new_anchor
        }

        cats = d['categories'].split(',')
        for cat in cats:
            categories[cat] += 1

        counter += 1
        if counter % 1000:
            with open('categories.pkl', 'wb') as f:
                pickle.dump(categories, f)

        collection.insert_one(new_record)

    logging.info("Creating indices")
    collection.create_index([('wikipedia_id', 1)])

    with open('total_categories.pkl', 'wb') as f:
        pickle.dump(categories, f)


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

            if not is_array_of_empty_strings(anchors['wikipedia_id']):
                for start, end, p_id, text, wiki_id in zip(
                        anchors['start'], anchors['end'], anchors['paragraph_id'], anchors['text'], anchors['wikipedia_id']
                ):

                    context = paragraphs[p_id]
                    context = context.replace('\1', '').replace('\n', '')
                    context = context[:start] + '<NE> ' + context[start:end] + ' </NE>' + context[end:]

                    try:
                        description = collection.find_one(
                            {'wikipedia_id': wiki_id})['text']['paragraph'][1].replace('\1', '').replace('\n', '')
                    except Exception as e:
                        failed_counts += 1
                        continue

                    text = text.replace('\1', '').replace('\n', '')

                    f.write(f'{text}\1{context}\1{description}\n')

        logging.error("Failed to find {} documents".format(failed_counts))


def mask_ne(string, probability=MASKING_PROBABILITY):

    starts, ends = re.finditer('<NE>', string), re.finditer('</NE>', string)
    new_string, pre_end_idx = '', 0
    for start, end in zip(starts, ends):
        new_string += string[pre_end_idx:start.start()]
        entity = string[start.start():end.end()]
        tokens = entity.split(' ')
        new_tokens = []
        for i, t in enumerate(tokens):
            if random.random() > probability or i == 0 or i == len(tokens)-1:
                new_tokens.append(t)
            else:
                new_tokens.append('<MASK>')
        new_string += ' '.join(new_tokens)
        pre_end_idx = end.end()
    new_string += string[pre_end_idx:]

    return new_string


def make_jsonl_to_csv(file_path, concat_contexts=False):

    count_no_desc, count_with_desc = 0, 0
    with open(file_path, 'r') as f:
        with \
          open(f'{file_path[:-5]}_ne_with_context.csv', 'w') as ne_with_context, \
          open(f'{file_path[:-5]}_ne_no_context.csv', 'w') as ne_no_context, \
          open(f'{file_path[:-5]}_masked_ne_with_context.csv', 'w') as masked_ne_with_context:

            ne_with_context_writer = csv.writer(ne_with_context, delimiter='\1')
            ne_no_context_writer = csv.writer(ne_no_context, delimiter='\1')
            masked_ne_with_context_writer = csv.writer(masked_ne_with_context, delimiter='\1')

            for line in f:
                data = json.loads(line)
                key, value = list(data.keys())[0], list(data.values())[0]
                if value['description'] == '':
                    count_no_desc += 1
                    continue
                count_with_desc += 1

                contexts = [context.replace('\1', '') for context in value['contexts'] if len(context.split())]
                title, description = value['label'].replace('\1', ''), value['description'].replace('\1', '')

                if len(contexts) > MAX_CONTEXT_NUM:
                    contexts = random.sample(contexts, MAX_CONTEXT_NUM)

                if concat_contexts:
                    contexts = '<CNTXT>' + '</CNTXT><CNTXT>'.join(contexts) + '</CNTXT>'
                    ne_with_context_writer.writerow([title, contexts, description])
                    ne_no_context_writer.writerow([title, title, description])
                    masked_ne_with_context_writer.writerow([title, mask_ne(contexts), description])

                else:
                    for context in contexts:
                        ne_with_context_writer.writerow([title, context, description])
                        ne_no_context_writer.writerow([title, title, description])
                        masked_ne_with_context_writer.writerow([title, mask_ne(context), description])

    print(f'No description: {count_no_desc}, with description: {count_with_desc}')


def label_based_on_bertscores(file_path1, file_path2, output_file, delimiter='~'):

    contexts, preds_we, labels_we, bertscore_output_we = compute_bertscores_from_file(file_path1, delimiter=delimiter)
    _, preds_woe, labels_woe, bertscore_output_woe = compute_bertscores_from_file(file_path2, delimiter=delimiter)
    labels = labels_we

    with open(output_file, 'w+') as f:
        for l, c, bert_we, bert_woe, pred_we, pred_woe in \
          tqdm(zip(labels, contexts, bertscore_output_we, bertscore_output_woe, preds_we, preds_woe)):
            if not bert_we or not bert_woe:
                classification_label = 1
            else:
                classification_label = 1 if bert_we >= bert_woe else 0
            f.write(f'{l}\1{pred_we}\1{bert_we}\1{pred_woe}\1{bert_woe}\1{c}\1{classification_label}\n')


def make_classification_dataset(CPE_model_name, CME_model_name, CPE_input_file, CME_input_file, delimiter='~'):

    CPE_input_data = pd.read_csv(CPE_input_file, delimiter='\1', header=None,
                                 names=['title', 'context', 'description'])
    CPE_input_x, CPE_input_y = list(CPE_input_data['context']), list(CPE_input_data['description'])

    CME_input_data = pd.read_csv(CME_input_file, delimiter='\1', header=None,
                                 names=['title', 'context', 'description'])
    CME_input_x, CME_input_y = list(CME_input_data['context']), list(CME_input_data['description'])

    training_args = TrainingArguments(output_dir=OUTPUT_DIR)
    CPE_model = BART(
        training_args,
        CPE_input_x, CPE_input_y,
        CPE_input_x, CPE_input_y,
        CPE_input_x, CPE_input_y,
        load=True,
        model_name=CPE_model_name
    )
    CME_model = BART(
        training_args,
        CME_input_x, CME_input_y,
        CME_input_x, CME_input_y,
        CME_input_x, CME_input_y,
        load=True,
        model_name=CME_model_name
    )

    CPE_preds, CPE_inputs, CPE_labels = CPE_model.pred()
    CME_preds, CME_inputs, CME_labels = CME_model.pred()

    bertscore = load_metric('bertscore')
    CPE_bert = bertscore.compute(
        predictions=CPE_preds, references=CPE_labels, lang='en', model_type='bert-large-uncased'
    )['f1']
    CME_bert = bertscore.compute(
        predictions=CME_preds, references=CME_labels, lang='en', model_type='bert-large-uncased'
    )['f1']
    classification_labels = [1 if cpe >= cme else 0 for cpe, cme in zip(CPE_bert, CME_bert)]

    with open('test_classification_no_concat.csv', 'w') as f:
        for context_cpe, label_cpe, pred_cpe, bert_cpe, context_cme, label_cme, \
            pred_cme, bert_cme, class_label, masked_input, entity_name in \
          zip(
              CPE_inputs, CPE_labels, CPE_preds, CPE_bert,
              CME_inputs, CME_labels, CME_preds, CME_bert,
              classification_labels, CME_input_x, list(CPE_input_data['title'])
          ):

            f.write(f'{context_cpe.replace(delimiter, "")}{delimiter}{label_cpe}{delimiter}{pred_cpe}{delimiter}'
                    f'{bert_cpe}{delimiter}{context_cme.replace(delimiter, "")}{delimiter}{label_cme}{delimiter}'
                    f'{pred_cme}{delimiter}{bert_cme}{delimiter}{class_label}{delimiter}'
                    f'{masked_input.replace(delimiter, "")}{delimiter}{entity_name}\n')


def x(file_path1, file_path2, output_file, delimiter='~'):

    bertscore = load_metric('bertscore')

    data1 = pd.read_csv(file_path1, names=['context1', 'label1', 'pred1'], header=None, delimiter=delimiter)
    data2 = pd.read_csv(file_path2, names=['context2', 'label2', 'pred2'], header=None, delimiter=delimiter)

    contexts1, labels1, preds1 = data1['context1'].values, data1['label1'].values, data1['pred1'].values
    contexts2, labels2, preds2 = data2['context2'].values, data2['label2'].values, data2['pred2'].values

    bertscore_output1 = bertscore.compute(
        predictions=preds1, references=labels1, lang='en', model_type='bert-base-uncased'
    )['f1']
    bertscore_output2 = bertscore.compute(
        predictions=preds2, references=labels2, lang='en', model_type='bert-base-uncased'
    )['f1']

    data1['bertscore1'] = bertscore_output1
    data2['bertscore2'] = bertscore_output2

    data = pd.concat([data1, data2], axis=1)

    data['classification_label'] = np.where(data['bertscore1'] >= data['bertscore2'], 1, 0)

    data.to_csv(output_file, header=False, index=False, sep='\1')
