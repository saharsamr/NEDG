import csv
import json
import os
import random


#  each json in jsonl:
# {
# 'wikipedia_title': ...,
# 'wikipedia_id': ...,
# 'wikipedia_description': ...,
# 'wikidata_description': , ...,
# 'contexts': [context1, context2, ...]
# }
# wikipedia_or_wikidata_definition can have two values, wikipedia and wikidata
def making_csv(jsonl_path, max_context_number, definition_source, data_split):
    # the contexts may have comma, so for the separating in csv should use \1 as a separator
    # making the csv file name based on the max_context_number and wikipedia or wikidata
    csv_file_name = f'{max_context_number}_contexts_{definition_source}_{data_split}.csv'

    #  reading the jsonl from the server
    with open(jsonl_path, 'r') as f:
        jsonl = []
        for line in f.readlines():
            json_obj = json.loads(line)
            jsonl.append(json_obj)
    with open(csv_file_name, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\1')
        writer.writerow(['entity_name', 'contexts', 'entity_description'])
        for json_str in jsonl:
            json_obj = json.loads(json_str)
            # entity name, entity description, and contexts
            entity_name = json_obj['wikipedia_title']
            if definition_source == 'wikipedia':
                if json_obj['wikipedia_description'] is None or json_obj['wikipedia_description'] == '':
                    continue
                entity_description = json_obj['wikipedia_description']
            elif definition_source == 'wikidata':
                if json_obj['wikidata_description'] is None or json_obj['wikidata_description'] == '':
                    continue
                entity_description = json_obj['wikidata_description']
            else:
                raise ValueError('Invalid value for wikipedia_or_wikidata_definition')
            # if a json have more than max_context_number contexts, it should randomly choose max_context_number of them
            contexts = json_obj['contexts']
            if len(contexts) > max_context_number:
                contexts = random.sample(contexts, max_context_number)
            # inserting to csv
            # each one has 3 parameters:
            # entity_name, context, entity_description
            # Add each context as a separate row in the CSV file
            for context in contexts:
                writer.writerow([entity_name, context, entity_description])
