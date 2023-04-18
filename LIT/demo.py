import pandas as pd
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
from lit_nlp.api import types as lit_types
from lit_nlp import dev_server

# Load your named entity to definition dataset into a pandas DataFrame
df = pd.read_csv("path/to/your/dataset.csv")

# Initialize the BART tokenizer and model
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')

# Define the maximum input and output sequence lengths
MAX_INPUT_SEQ_LENGTH = 600
MAX_OUTPUT_SEQ_LENGTH = 20

# Define the input and output specs for LIT
input_spec = {'context': lit_types.TextSegment()}
output_spec = {'definition': lit_types.TextSegment()}

# Convert the DataFrame to a LIT dataset
examples = lit_dataset.Dataset.from_dataframe(df)


# Define the function to generate predictions for LIT
def predict_fn(inputs):
    named_entity = inputs['context']
    input_ids = tokenizer.encode(named_entity, add_special_tokens=False, truncation=True,
                                 max_length=MAX_INPUT_SEQ_LENGTH)
    input_ids = torch.tensor(input_ids).unsqueeze(0)
    output_ids = model.generate(input_ids, max_length=MAX_OUTPUT_SEQ_LENGTH, num_beams=5, no_repeat_ngram_size=2)
    output_str = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return {'definition': output_str}


# Wrap the model and prediction function in a LIT model
lit_model = lit_model.Model(
    input_spec=input_spec,
    output_spec=output_spec,
    predict_fn=predict_fn
)

# Start the LIT server
dev_server.serve(
    lit_model,
    datasets={'my_dataset': examples},
    port=5432  # Set the port number to the desired value
)
