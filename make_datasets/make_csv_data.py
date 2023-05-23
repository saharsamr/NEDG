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
# wikipedia_or_wikidata_definition can have two values, pedia for wikipedia and data for wikidata
def making_csv(jsonl_path, max_context_number, wikipedia_or_wikidata_definition, test_or_train):
    #  reading the jsonl from the server
    with open(jsonl_path, 'r') as f:
        jsonl = list(f)
    rows = []
    for json_str in jsonl:
        json_obj = json.loads(json_str)
        # entity name, entity description, and contexts
        entity_name = json_obj['wikipedia_title']
        if wikipedia_or_wikidata_definition == 'pedia':
            entity_description = json_obj['wikipedia_description']
        elif wikipedia_or_wikidata_definition == 'data':
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
            rows.append([entity_name, context, entity_description])

    # the contexts may have comma, so for the separating in csv should use \1 as a separator
    # making the csv file name based on the max_context_number and wikipedia or wikidata
    csv_file_name = f'{max_context_number}_contexts_wiki{wikipedia_or_wikidata_definition}_{test_or_train}.csv'

    # save the csv by the name that has been created earlier
    with open(csv_file_name, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\1')
        writer.writerow(['entity_name', 'contexts', 'entity_description'])
        writer.writerows(rows)

    return csv_file_name
