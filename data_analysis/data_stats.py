from GNED.config import *
from GNED.data_handler.wiki_dataset import WikiDataset
from data_analysis.data_plots import number_of_tokens_histogram
from transformers import BartTokenizerFast
import numpy as np
import pandas as pd
from tqdm import tqdm
from make_datasets.config import *
import pickle


if __name__ == "__main__":

    tokenizer = BartTokenizerFast.from_pretrained(
        MODEL_GENERATION_NAME, padding=True, truncation=True,
    )
    tokenizer.add_special_tokens({'additional_special_tokens': ADDITIONAL_SPECIAL_TOKENS})

    train = pd.read_csv(TRAIN_GENERATION_FILE, delimiter='\1').sample(frac=0.2, random_state=42)
    test = pd.read_csv(TEST_GENERATION_FILE, delimiter='\1').sample(frac=0.2, random_state=42)
    val = pd.read_csv(VALID_GENERATION_FILE, delimiter='\1').sample(frac=0.2, random_state=42)

    data = pd.concat([train, test, val])
    print(len(data))
    data = data.dropna()
    print(len(data))
    x, y = list(data['contexts']), list(data['entity_description'])

    dataset = WikiDataset(tokenizer, x, y, mask_entity=False)

    lengths = [(len(sample['input_ids']), len(sample['labels'])) for sample in tqdm(dataset)]
    context_lengths = [sample[0] for sample in lengths]
    description_lengths = [sample[1] for sample in lengths]

    input_mean_length = np.mean(context_lengths)
    input_std_length = np.std(context_lengths)
    output_mean_length = np.mean(description_lengths)
    output_std_length = np.std(description_lengths)
    print(f'Contexts Mean Length (In Tokens): {input_mean_length}, Standard Deviation: {input_std_length}')
    print(f'Descriptions Mean Length (In Tokens): {output_mean_length}, Standard Deviation: {output_std_length}')

    with (open('contexts_token_count.pkl', 'wb') as f1,
          open(f'{SOURCE_DEFINITION}_descriptions_token_count.pkl', 'wb') as f2):
        pickle.dump(context_lengths, f1)
        pickle.dump(description_lengths, f2)

    number_of_tokens_histogram(
        context_lengths, 'Contexts\' Token Count Histogram',
        'Contexts\' Token Count', input_std_length+3.5*input_std_length)
    number_of_tokens_histogram(
        description_lengths, f'Wikidata Descriptions\' Token Count Histogram',
        'Descriptions\' Token Count', output_mean_length+3.5*output_std_length)
