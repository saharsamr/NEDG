# A Model for Dynamic Utilization of Entity Name and Context to Enhance Context-Aware ‌Named Entity Description Generation

This repository contains the code and resources for the paper titled "Enhancing Context-Aware ‌Named Entity Description Generation: A Model for Dynamic Utilization of Entity Name and Context". The paper introduces a novel approach for generating descriptions of named entities, addressing the challenges posed by the evolving and ambiguous nature of these entities. The proposed framework uses conditional generation models with mix training data enabling the model to dynamically ignore the entity name to improve the quality and alignment of the generated descriptions with the given context.

## Repository Structure
This repository is organized as follows:

1. **make_datasets:** This directory contains the code for creating the necessary datasets for the generation and classification parts of the hybrid framework. For more information, visit this [link](https://github.com/saharsamr/NED/tree/master/make_datasets#readme).

2. **GNED:** The GNED directory contains the code for training, evaluating, and testing all three models: Context with the Plain Entity (CPE), Context with Masked Entity (CME), and the classifier. These models form the core components of the hybrid framework. For more information, visit this [link](https://github.com/saharsamr/NED/tree/master/GNED#readme).

3. **data_analysis:** The data_analysis directory includes the code for comparing the performance of the CPE and CME models on the worst results of the CPE, along with other analysis tasks. For more information, visit this [link](https://github.com/saharsamr/NED/tree/master/data_analysis#readme).

Each directory mentioned above has its own README file with detailed instructions on how to use the code and resources provided.

## Usage
To use this repository, follow the instructions in each directory's README file to set up and run the code for the specific task you are interested in. Make sure to install any required dependencies and download the necessary datasets as instructed.

## License
This repository is licensed under the [MIT License](https://github.com/git/git-scm.com/blob/main/MIT-LICENSE.txt).

## Contributions
Contributions to this repository are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## Contact
For any further questions or inquiries, please contact [Sahar](sahar.rajabi@ut.aci.ir) or [Kiana](kghezelbash@aut.ac.ir).

Enjoy exploring the hybrid framework for generating named entity descriptions!
