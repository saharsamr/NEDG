from GNED.config import *
from GNED.data_handler.wiki_dataset import WikiDataset
from transformers import BartTokenizerFast
import pandas as pd
import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    tokenizer = BartTokenizerFast.from_pretrained(
        MODEL_GENERATION_NAME, model_max_length=INPUT_GENERATION_MAX_LENGTH, padding=True, truncation=True,
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
    input_mean_length = np.mean([len(sample['input_ids']) for sample in tqdm(dataset)])
    output_mean_length = np.mean([len(sample['labels']) for sample in tqdm(dataset)])

    print(input_mean_length, output_mean_length)
