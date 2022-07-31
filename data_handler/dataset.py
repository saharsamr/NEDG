from torch.utils.data import Dataset
import pandas as pd


class WikiDataset(Dataset):

    def __init__(self, path):

        self.data = pd.read_csv(
            path, delimiter='\1', on_bad_lines='skip', header=0, columns=['word', 'context', 'description']
        )

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        item = self.data.iloc[idx]
        context, description = item['context'], item['description']
        # TODO: remove the replacements
        context = context.replace('<NE>', '<NE> ').replace('</NE>', ' </NE>')

        return context, description



