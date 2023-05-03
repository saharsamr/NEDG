from lit_nlp.components.metrics import SimpleMetrics
from lit_nlp.api import types as lit_types

from typing import Sequence, Optional, Dict, Text
import numpy as np

from datasets import load_metric


class BertScore(SimpleMetrics):

    def __init__(self):

        super().__init__()
        self.metric = load_metric('bertscore')

    def is_compatible(self, model, dataset) -> bool:
        print('======================================================')
        print('======================================================')
        print('======================================================')

        # is_pred_compatible = isinstance(pred_spec, lit_types.GeneratedText)
        # is_parent_compatible = isinstance(parent_spec, lit_types.TextSegment)
        #
        # return is_pred_compatible and is_parent_compatible
        return True

    def meta_spec(self):
        return {
            "description": lit_types.TextSegment(),
            "output_text": lit_types.GeneratedText(parent="description")
        }

    def compute(self,
                labels: Sequence[lit_types.TextSegment],
                preds: Sequence[lit_types.GeneratedText],
                label_spec: lit_types.TextSegment,
                pred_spec: lit_types.GeneratedText,
                config: Optional[lit_types.JsonDict] = None) -> Dict[Text, float]:

        bertscore_output = self.metric.compute(
            predictions=preds, references=labels, lang='en', model_type='bert-large-uncased'
        )
        return {'bert-f1': np.mean(bertscore_output['f1'])}
