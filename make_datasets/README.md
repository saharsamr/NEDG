# Make Necessary Datasets
We must collect a dataset of entity descriptions and some contexts containing those entities. For that, we used the **Wikipedia** and **Wikidata** encyclopedias. After fine-tuning **BART**, introduced in [this paper](https://aclanthology.org/2020.acl-main.703/), we used the prediction of fine-tuned models to make the classification dataset. More details can be found below.

## Description Generation Datasets:
By running **./make_dataset.sh** file, the following steps will be done (which can take up to several hours!):
```console
./make_datasets/make_dataset.sh
```
- Downloading the latest Wikipedia dump
- Using the [WikiExtractor](https://github.com/attardi/wikiextractor), each page in the dump will be stored as a JSON file in some directories
- The extracted pages of the previous step will be imported to a MongoDB instance
- A MongoDB instance using the provided docker-compose.yaml file will be up for future uses
- For each entry in [MongoDB](https://www.mongodb.com), the anchors will be found using the annotations available in the Wikipedia dump so that each entry will have a list of anchor names and their corresponding paragraph index.
- For each entry in MongoDB, a query will run to find the id of entries that have an anchor to that specific entry; so each entry will have a list containing the id of the pages that contain a paragraph capable of being used as the context of the first entry
- For each entry that some contexts have been found, a request will be sent to Wikidata API so that the Wikidata information of those are retrieved
- Entries having contexts will be exported to a JSONL file in which we will extract the contexts for each entry using the id of documents containing its context and the anchor list of each entry that was created in the previous steps
- The will be split into three different sets, train, test, and validation, based on entity names and the source of the definition, Wikipedia or Wikidata, so that we can prevent data leakage
- Converting JSONL files to CSV

The parameters in the config.py file can be altered. The role of each parameter can be found in the table below. This table is limited to the scope of the description generation dataset.


<p align="center">

|          Feature                | Description                                                       |
| -------------------------------|-------------------------------------------------------------------|
| MONGODB_LINK                   | Link for connecting to the MongoDB Instance                       |
| MONGODB_PORT                   | Port of the MongoDB instance                                      |
| MONGODB_DATABASE               | The database name that the downloaded dump will be added to        |
| MONGODB_COLLECTION             | The collection name that the downloaded dump will be added to      |
| MONGODB_READ_BATCH_SIZE        | The batch size for MongoDB queries                                |
| MONGODB_WRITE_BATCH_SIZE       | The batch size for updating MongoDB entries                       |
| MONGODB_USERNAME               | Username for connecting to the database/collection on MongoDB     |
| MONGODB_PASSWORD               | Password for connecting to the database/collection on MongoDB     |
| WIKI_JSONS_PATH                | Path to JSON files created by WikiExtractor                       |
| MAX_ENTITY_NAME_LENGTH         | The upper limit for the length of entity names in terms of words   |
| MIN_CONTEXT_LENGTH             | The lower limit for the length of contexts in terms of words       |
| FINAL_MIN_CONTEXT_LENGTH       | The lower limit for the length of contexts after preprocessing     |
| WIKI_DUMP_JSONL_PATH           | The path to export MongoDB entries                                |
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
The classifier aims to select the better description provided by CPE or CME models. For that, first, each data split should be fed to CPE and CME separately, then the generated descriptions will be compared with the golden description, and their BertScore will be computed. If the CPE's description gains a better BertScore, the label would be 1 and vice-versa. 
We use the same train, test, and validation sets as in the description generation dataset, so there will be no data leakage. 
This code can be run by the following command:
```console
python -m make_datasets/make_classification_dataset.py
```
The parameters that can be altered for making the classification dataset are summarized in the table below.

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
  