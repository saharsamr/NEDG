from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types

import pandas as pd


class WikiDataset(lit_dataset.Dataset):

    def __init__(self, file_path):

        data = pd.read_csv(file_path, delimiter='\1', header=None, names=['title', 'context', 'description'])
        self._examples = [{
            "title": row['title'],
            "context": row['context'],
            "description": row['description']
        } for _, row in data.iterrows()]

    def spec(self) -> lit_types.Spec:

        return {
            "title": lit_types.TextSegment(),
            "context": lit_types.TextSegment(),
            "description": lit_types.TextSegment()
        }