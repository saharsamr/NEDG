import csv
import json
import random
from config import FINAL_MIN_CONTEXT_LEN, MAX_CONTEXT_NUMBER, \
    TRAIN_JSONL_PATH, TEST_JSONL_PATH, VAL_JSONL_PATH,\
    CSVS_PATH, SOURCE_DEFINITION


def making_csv(jsonl_path, max_context_number, definition_source, data_split):

    csv_file_name = f'{CSVS_PATH}{max_context_number}_contexts_{definition_source}_{data_split}.csv'

    with open(jsonl_path, 'r') as jsonl_f, open(csv_file_name, 'w+') as csv_f:

        writer = csv.writer(csv_f, delimiter='\1')
        writer.writerow(['entity_name', 'contexts', 'entity_description'])

        for line in jsonl_f.readlines():
            json_obj = json.loads(line)

            entity_name = json_obj['wikipedia_title']
            if definition_source == 'wikipedia':
                entity_description = json_obj['wikipedia_description']
            elif definition_source == 'wikidata':
                entity_description = json_obj['wikidata_description']
            else:
                raise ValueError('Invalid value for source of definition: should be wikipedia or wikidata')

            contexts = json_obj['contexts']
            contexts = [context for context in contexts if len(context) >= FINAL_MIN_CONTEXT_LEN]
            if len(contexts):
                if len(contexts) > max_context_number:
                    contexts = random.sample(contexts, max_context_number)
                for context in contexts:
                    writer.writerow([entity_name, context, entity_description])


making_csv(TRAIN_JSONL_PATH, MAX_CONTEXT_NUMBER, SOURCE_DEFINITION, 'train')
making_csv(TEST_JSONL_PATH, MAX_CONTEXT_NUMBER, SOURCE_DEFINITION, 'test')
making_csv(VAL_JSONL_PATH, MAX_CONTEXT_NUMBER, SOURCE_DEFINITION, 'train')
