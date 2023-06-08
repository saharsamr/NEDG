# Training and Evaluating Models
Three models should be trained and evaluated, the CPE, CME, and BERT-based classifier. The generative models, CPE and CME, are based on BART architecture. 

The **generation_main.py** file consists of codes for training and evaluating generative models, while the **classifier_main.py** contains the training and evaluating steps for the classification model.

First, some parameters should be set in the **config.py** file for training each model, especially the TASK parameter, indicating whether the generation or classification parts should be run. All parameters will be described in detail in the following. After setting the parameters, the following command should be run:
```console
python -m GNED.main
```
These are the parameters shared between the generation and classification parts.
|      Parameter    |                                    Description                                    |
|-------------------|-----------------------------------------------------------------------------------|
| TASK              | Indicating the task that is going to be run. It can be "GENERATION" and "CLASSIFICATION" |
| LOGGING_DIR       | Path to the directory the logs are going to be saved                              |
| OUTPUT_DIR        | Path to the directory where the models and test results are going to be saved      |
| WARMUP_STEPS      | Number of steps before starting schedulers                                       |
| WEIGHT_DECAY      | The weight-decay parameters to be used by the optimizer                          |
| LEARNING_RATE     | The starting value of the learning-rate parameter                                |
| DEFINITION_SOURCE | Source of definition to be used for training models. It can be "wikidata" or "wikipedia" |
| EPOCHS            | Number of epochs for fine-tuning models                                          |

## Generative Models
The parameters related to training, evaluating, and saving the generative models are described in the table below. These can be used for both CPE and CME models.
| Parameter                          | Description                                                                                        |
|-----------------------------------|----------------------------------------------------------------------------------------------------|
| MASK_ENTITY                       | Whether the entity names are going to be masked in the input of the following model                |
| TRAIN_GENERATION_BATCH_SIZE       | Batch size to be used for training generative models                                               |
| VALID_GENERATION_BATCH_SIZE       | Batch size to be used with the validation data during training generative models                  |
| TEST_GENERATION_BATCH_SIZE        | Batch size to be used for testing generative models                                                |
| INPUT_GENERATION_MAX_LENGH        | Maximum length of the input sequence. Inputs longer than this specified value will be truncated   |
| OUTPUT_GENERATION_MAX_LENGHT      | Maximum length of output that generative models are allowed to generate                           |
| OUTPUT_GENERATION_MIN_LENGTH      | Minimum length of output that generative models are allowed to generate                           |
| MODEL_GENERATION_NAME             | The BART architecture is going to be used as the base model and to be fine-tuned                   |
| DATA_GENERATION_FOLDER            | Path to the folder containing data splits                                                         |
| TRAIN_GENERATION_FILE             | Path to train data for generative models                                                          |
| VALID_GENERATION_FILE             | Path to validation data for generative models                                                     |
| TEST_GENERATION_FILE              | Path to test data for generative models                                                           |
| LOAD_GENERATION_MODEL             | Whether to load a previously fine-tuned model                                                     |
| MODEL_GENERATION_PATH             | Path to the model should be loaded before starting the fine-tuning. It needs the LOAD_GENERATION_MODEL to be set to True |
| EVALUATE_GENERATION               | Whether to evaluate the fine-tuned models on the test split                                        |
| PRED_GENERATION_FILE_PATH         | Path to file that the evaluation results should be saved. It needs the EVALUATE_GENERATION to be set to True |

## Classification Model
The parameters related to training, evaluating, and saving the classification model are described in the table below.
| Parameter                            | Description                                                                                        |
|--------------------------------------|----------------------------------------------------------------------------------------------------|
| TRAIN_CLASSIFICATION_BATCH_SIZE       | Batch size to be used during training for the classification task                                  |
| VALID_CLASSIFICATION_BATCH_SIZE       | Batch size to be used with validation data during training for the classification task             |
| TEST_CLASSIFICATION_BATCH_SIZE        | Batch size used while testing the model using the test split for the classification task           |
| INPUT_CLASSIFICATION_MAX_LENGTH       | Maximum length of the input. Inputs longer than the specified value will be truncated by the tokenizer |
| MODEL_CLASSIFICATION_NAME             | The BERT architecture is going to be used as the base model and to be fine-tuned                   |
| DATA_CLASSIFICATION_FOLDER            | Path to the folder containing data splits for the classification task                              |
| TRAIN_CLASSIFICATION_FILE             | Path to train data for the classification task                                                     |
| VALID_CLASSIFICATION_FILE             | Path to validation data for the classification task                                                |
| TEST_CLASSIFICATION_FILE              | Path to test data for the classification task                                                      |
| LOAD_CLASSIFICATION_MODEL             | Whether to load a previously fine-tuned model before starting fine-tuning                          |
| MODEL_CLASSIFICATION_PATH             | Path to the model that should be loaded. It needs the LOAD_CLASSIFICATION_MODEL to be set to True  |
| EVALUATE_CLASSIFICATION               | Whether to evaluate the classifier using the test split                                            |
| PRED_CLASSIFICATION_FILE_PATH         | Path to save the classification result on the test split. It needs the EVALUATE_CLASSIFICATION to be set to True |


