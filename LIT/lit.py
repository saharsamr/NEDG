from lit_nlp import dev_server
from lit_nlp import server_flags

import sys
from typing import Optional, Sequence

from absl import app
from absl import flags
from absl import logging

from LIT.lit_datasets import WikiDataset
from LIT.lit_models import BartModel
from LIT.comparision_models import ModelComparison

FLAGS = flags.FLAGS


def get_wsgi_app() -> Optional[dev_server.LitServerType]:
    FLAGS.set_default("server_type", "default")
    FLAGS.set_default("host", "0.0.0.0")
    FLAGS.set_default("demo_mode", True)

    unused = flags.FLAGS(sys.argv, known_only=True)
    if unused:
        logging.info(
            "quickstart_sst_demo:get_wsgi_app() called with unused "
            "args: %s", unused)

    return main([])


# filtered data returned based on being under threshold
def filter_data(test_data, bert_score_threshold):
    filtered_data = []
    for example in test_data:
        bert_score = example['metrics']['bert_score']
        if bert_score < bert_score_threshold:
            filtered_data.append(example)
    return filtered_data


def filter_data_callback(dataset, payload):
    bert_score_threshold = float(payload['bert_score_threshold'])
    test_data = dataset.get_all_examples('test')
    filtered_data = filter_data(test_data, bert_score_threshold)
    return {'data': filtered_data}


def main(argv: Sequence[str]) -> Optional[dev_server.LitServerType]:
    datasets = {
        'wiki_dataset': WikiDataset('data/wikipedia/1_contexts_wikidata_test.csv')
    }
    models = {
        # "bart_CME": BartModel('../results/NoConcatCME', 'facebook/bart-large-cnn', mask_entity=True),
        # "bart_CPE": BartModel('../results/NoConcatCPE', 'facebook/bart-large-cnn')
        "model_comparison": ModelComparison('results/1-context-1epoch-wikidata-CPE',
                                            'results/1-context-1epoch-wikidata-CME',
                                            'facebook/bart-large-cnn')
    }

    lit_demo = dev_server.Server(models, datasets, **server_flags.get_flags())
    return lit_demo.serve()


if __name__ == "__main__":
    app.run(main)
