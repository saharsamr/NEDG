from lit_nlp import dev_server
from lit_nlp import server_flags

import sys
from typing import Optional, Sequence

from absl import app
from absl import flags
from absl import logging

from lit_datasets import WikiDataset
from lit_models import BartModel
from comparision_models import ModelComparison


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


def main(argv: Sequence[str]) -> Optional[dev_server.LitServerType]:

    datasets = {
        'wiki_dataset': WikiDataset('../data/HumanNoConcat/test_human_ne_with_context.csv')
    }
    models = {
        # "bart_CME": BartModel('../results/NoConcatCME', 'facebook/bart-large-cnn', mask_entity=True),
        # "bart_CPE": BartModel('../results/NoConcatCPE', 'facebook/bart-large-cnn')
        "model_comparison": ModelComparison('../results/NoConcatCPE', '../results/NoConcatCME', 'facebook/bart-large-cnn')
    }

    lit_demo = dev_server.Server(models, datasets, **server_flags.get_flags())
    return lit_demo.serve()


if __name__ == "__main__":
    app.run(main)
