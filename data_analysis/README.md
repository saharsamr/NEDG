# Analysis
There are two types of analysis available for now:
1. Comparing the CPE and CME performance on different subsets of data selected based on CPE's Bertscore
2. Comparing the CPE and CME performance on different subsets of data selected based on entities' popularity

For running the first analysis, the following command should be used:
```console
python -m data_analysis.CPE_errors
```
The second analysis can be run using:
```console
python -m data_analysis.popularity_analysis
```
Some parameters need to be set before running the codes:
| Feature                   | Description                                                                                                                                                                     |
|---------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| JSONL_PATH                | Path to the JSONL file, which contains entities and their context. This file will be used to extract the entity's popularity based on the number of contexts for each entity. |
| ENTITY_POPULARITY_PATH    | Path to the pickle file that the popularity dictionary will be saved for the future.                                                                                           |
| CLASSIFICATION_RESULT_PATH | Path to the pickle file containing the data frame that was built after feeding the test split to the classifier. This will be used to access the CPE, CME, and hybrid models' descriptions for further analysis. |
