import pandas as pd


def remove_short_contexts(data_file):

    data = pd.read_csv(
        data_file, delimiter='\1', on_bad_lines='skip', header=0, names=['word', 'context', 'description']
    )

    data['context_len'] = data.context.str.split().str.len()
    data.drop(data[data['context_len'] < 15].index, inplace=True)

    data.to_csv('data/data.csv', sep='\1', columns=['word', 'context', 'description'], header=False, index=False)


def remove_long_entities(data_file):

    data = pd.read_csv(
        data_file, delimiter='\1', on_bad_lines='skip', header=0, names=['word', 'context', 'description']
    )

    data['entity_len'] = data.word.str.split().str.len()
    data.drop(data[data['entity_len'] > 4].index, inplace=True)

    data.to_csv('data/data.csv', sep='\1', columns=['word', 'context', 'description'], header=False, index=False)


def remove_list_items(data_file):

    data = pd.read_csv(
        data_file, delimiter='\1', on_bad_lines='skip', header=0, names=['word', 'context', 'description']
    )

    data['has_bullet'] = data.context.str.contains('BULLET')
    data.drop(data[data['has_bullet'] == True].index, inplace=True)
    data['has_bullet'] = data.description.str.contains('BULLET')
    data.drop(data[data['has_bullet'] == True].index, inplace=True)

    data.to_csv('data/data.csv', sep='\1', columns=['word', 'context', 'description'], header=False, index=False)


def remove_special_characters(data_file):

    data = pd.read_csv(
        data_file, delimiter='\1', on_bad_lines='skip', header=0, names=['word', 'context', 'description']
    )
    special_chars_regexp = '[^a-z|A-Z| |\.|\!|\?|0-9|(|)|-|\'|`]'
    data['context'] = data['context'].str.replace(special_chars_regexp, ' ', regex=True)
    data['description'] = data['description'].str.replace(special_chars_regexp, ' ', regex=True)

    data.to_csv('data/data.csv', sep='\1', columns=['word', 'context', 'description'], header=False, index=False)
