import pandas as pd


def remove_short_contexts(data_file):

    data = pd.read_csv(
        data_file, delimiter='\1', on_bad_lines='skip', header=0, names=['word', 'context', 'description']
    )

    data['context_len'] = data.context.str.split().str.len()
    data.drop(data[data['context_len'] < 10].index, inplace=True)

    data.to_csv('data/data.csv', sep='\1', columns=['word', 'context', 'description'], header=False, index=False)


def remove_list_item_context(data_file):

    data = pd.read_csv(
        data_file, delimiter='\1', on_bad_lines='skip', header=0, names=['word', 'context', 'description']
    )

    data['has_bullet'] = data.context.str.split().str.startswith('BULLET')
    data.drop(data[data['has_bullet'] == True].index, inplace=True)
    data['has_bullet'] = data.description.str.startswith('BULLET')
    data.drop(data[data['has_bullet'] == True].index, inplace=True)

    data.to_csv('data/data.csv', sep='\1', columns=['word', 'context', 'description'], header=False, index=False)

