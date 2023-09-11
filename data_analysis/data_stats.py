from GNED.config import *
from GNED.data_handler.wiki_dataset import WikiDataset
from data_analysis.data_plots import number_of_tokens_histogram
from transformers import BartTokenizerFast
import numpy as np
from tqdm import tqdm
import json
from make_datasets.config import *
import pickle


def read_json_lines(file_path):

    x, y = [], []
    with open(file_path, 'r') as json_f:
        for line in tqdm(json_f.readlines()):
            json_obj = json.loads(line)
            if not len(json_obj['contexts']):
                continue
            x.append(json_obj['contexts'])
            y.append(json_obj[f'{SOURCE_DEFINITION}_description'])

    return x, y


if __name__ == "__main__":

    tokenizer = BartTokenizerFast.from_pretrained(
        MODEL_GENERATION_NAME, padding=True, truncation=True,
    )
    tokenizer.add_special_tokens({'additional_special_tokens': ADDITIONAL_SPECIAL_TOKENS})

    train_x, train_y = read_json_lines(TRAIN_JSONL_PATH)
    test_x, test_y = read_json_lines(TEST_JSONL_PATH)
    val_x, val_y = read_json_lines(VAL_JSONL_PATH)

    x = train_x + test_x + val_x
    y = train_y + test_y + val_y

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

    with open('contexts_token_count.pkl', 'wb') as f1, open('descriptions_token_count.pkl', 'wb') as f2:
        pickle.dump(context_lengths, f1)
        pickle.dump(description_lengths, f2)

    number_of_tokens_histogram(
        context_lengths, 'Contexts\' Token Count Histogram', 'Contexts\' Token Count')
    number_of_tokens_histogram(
        description_lengths, f'{SOURCE_DEFINITION}Descriptions\' Token Count Histogram', 'Descriptions\' Token Count')
