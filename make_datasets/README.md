# Make Necessary Datasets
We need to gather a dataset consisting of entity descriptions along with their respective contexts. To achieve this, we used the **Wikipedia** and **Wikidata** encyclopedias as our sources. We then fine-tuned the **BART** model, introduced in [this paper](https://aclanthology.org/2020.acl-main.703/), and employed the predictions from the fine-tuned models to create the classification dataset. Further information is provided below.

## Description Generation Datasets:
By running **./make_dataset.sh** file, the following steps will be done (which can take up to several hours!):
```console
./make_datasets/make_dataset.sh
```
- Downloading the latest Wikipedia dump
- Using the [WikiExtractor](https://github.com/attardi/wikiextractor), every page from the dump will be saved as a JSON file within specific directories
- The pages extracted from the previous stage will be imported into a MongoDB instance
- A MongoDB instance will be set up using the provided docker-compose.yaml file for future use
- For each entry in [MongoDB](https://www.mongodb.com), the anchors will be located using the annotations available in the Wikipedia dump, resulting in a list of anchor names and their corresponding paragraph indices for each entry
- For each MongoDB entry a query will be executed to identify the IDs of entries containing an anchor to the specific entry, creating a list of page IDs with paragraphs that can serve as the context for the initial entry
- For entries with identified contexts, a request will be sent to the Wikidata API to retrieve the corresponding Wikidata information
- Entries with contexts will be exported to a JSONL file, from which the contexts for each entry will be extracted using the document IDs containing the context and the previously created anchor list
- The data will be split into three different sets, train, test, and validation, based on entity names and the definition source, Wikipedia or Wikidata, to prevent data leakage
- Converting JSONL files to CSV format

The parameters in the config.py file can be altered. The role of each parameter can be found in the table below. This table is limited to the scope of the description generation dataset.


<p align="center">

|          Feature               | Description                                                       |
| -------------------------------|-------------------------------------------------------------------|
| MONGODB_LINK                   | Link for connecting to the MongoDB Instance                       |
| MONGODB_PORT                   | Port of the MongoDB instance                                      |
| MONGODB_DATABASE               | The database name that the downloaded dump will be added to       |
| MONGODB_COLLECTION             | The collection name that the downloaded dump will be added to     |
| MONGODB_READ_BATCH_SIZE        | The batch size for MongoDB queries                                |
| MONGODB_WRITE_BATCH_SIZE       | The batch size for updating MongoDB entries                       |
| MONGODB_USERNAME               | Username for connecting to the database/collection on MongoDB     |
| MONGODB_PASSWORD               | Password for connecting to the database/collection on MongoDB     |
| WIKI_JSONS_PATH                | Path to JSON files created by WikiExtractor                       |
| MAX_ENTITY_NAME_LENGTH         | The upper limit for the length of entity names in terms of words  |
| MIN_CONTEXT_LENGTH             | The lower limit for the length of contexts in terms of words      |
| FINAL_MIN_CONTEXT_LENGTH       | The lower limit for the length of contexts after preprocessing    |
| WIKI_DUMP_JSONL_PATH           | The path to export MongoDB entries as a JSONL                     |
| TRAIN_SHARE                    | The share of the train split                                      |
| TEST_SHARE                     | The share of the test split                                       |
| VALID_SHARE                    | The share of the validation split                                 |
| SOURCE_DEFINITION              | Indicating the description source ('wikipedia' or 'wikidata')     |
| TRAIN_JSONL_PATH               | Path to the JSONL file of the train split                         |
| TEST_JSONL_PATH                | Path to the JSONL file of the test split                          |
| VALID_JSONL_PATH               | Path to the JSONL file of the validation split                    |
| CSVS_PATH                      | Path to export CSV files from JSONL files                         |
| MAX_CONTEXT_NUMBER             | Number of contexts to be used for each entity                     |
  
</p>

## Classification Dataset
The objective of the classifier is to select the superior description provided by either the CPE or CME models. To accomplish this, each data split must first be separately fed to the CPE and CME models. The resulting descriptions will then be compared to the golden description, and their BertScore, introduced in [this paper](https://openreview.net/forum?id=SkeHuCVFDr) will be calculated. If the CPE's description achieves a higher BertScore, the label will be set to 1, and vice versa.
We use the same train, test, and validation sets as in the description generation dataset, so there will be no data leakage. 
This code can be run by the following command:
```console
python -m make_datasets/make_classification_dataset.py
```
The parameters that can be modified for making the classification dataset are summarized in the table below.

<p align="center">
  
|          Feature                | Description                                                      |
| -------------------------------|-------------------------------------------------------------------|
| LOGGING_DIR                    | The path to the directory to put the logs.                        |
| CPE_MODEL_NAME                 | Path to the trained CPE model                                     |
| CME_MODEL_NAME.                | Path to the trained CME model                                     |
| TRAIN_CSV_FILE_PATH            | The path to the CSV file of the train split                       |
| TEST_CSV_FILE_PATH             | The path to the CSV file of the test split                        |
| VAL_CSV_FILE_PATH              | The path to the CSV file of the validation split                  |
| TRAIN_CLASSIFICATION_PATH      | Path of the file that the train split of classification data is going to be saved|
| TEST_CLASSIFICATION_PATH       | Path of the file that the test split of classification data is going to be saved|
| VAL_CLASSIFICATION_PATH.       | Path of the file that the validation split of classification data is going to be saved|

</p>
  
