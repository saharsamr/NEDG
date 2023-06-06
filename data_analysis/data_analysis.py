import json

import matplotlib.pyplot as plt
from pymongo import InsertOne, DeleteMany, ReplaceOne, UpdateOne


def count_words(sentence):
    words = sentence.split()
    return len(words)


def distribution_of_entity_context_number(path):
    with open(path, 'r') as f:
        number_of_contexts = []
        for line in f.readlines():
            data = json.loads(line)
            number_of_contexts.append(len(data['contexts']))
        plt.hist(number_of_contexts, bins=20)
        plt.title('Distribution of Number of Contexts per Entity')
        plt.xlabel('Number of Contexts')
        plt.ylabel('Frequency')
        plt.show()


def distribution_of_contexts_length_in_json(path):
    with open(path, 'r') as f:
        context_lengths = []
        for line in f.readlines():
            data = json.loads(line)
            for context in data['contexts']:
                context_lengths.append(count_words(context))
        plt.hist(context_lengths, bins=20)
        plt.title('Distribution of Context Lengths in JSONL File')
        plt.xlabel('Context Length')
        plt.ylabel('Frequency')
        plt.show()


def files_distribution_of_descriptions_length(path):
    with open(path, 'r') as f:
        wikipedia_description_lengths = []
        wikidata_description_lengths = []
        for line in f.readlines():
            data = json.loads(line)
            wikipedia_description_lengths.append(count_words(data['wikipedia_description']))
            wikidata_description_lengths.append(count_words(data['wikidata_description']))
        plt.hist(wikipedia_description_lengths, bins=20, alpha=0.5, label='Wikipedia')
        plt.hist(wikidata_description_lengths, bins=20, alpha=0.5, label='Wikidata')
        plt.title('Distribution of Description Lengths by Source')
        plt.xlabel('Description Length')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()


