from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import types as lit_types

import pandas as pd
import pickle


class WikiDataset(lit_dataset.Dataset):

    def __init__(self, file_path):

        data = pd.read_csv(file_path, delimiter='\1')
        with open('temp.pkl', 'rb') as f:
            main_data = pickle.load(f)
        data = data[data['entity_name'].isin(main_data['title'].values)]
        data = data.sample(frac=0.5, random_state=42)
        self._examples = [{
            "title": row['entity_name'],
            "context": row['contexts'],
            "description": row['entity_description']
        } for _, row in data.iterrows()]

    def spec(self) -> lit_types.Spec:

        return {
            "title": lit_types.TextSegment(),
            "context": lit_types.TextSegment(),
            "description": lit_types.TextSegment()
        }