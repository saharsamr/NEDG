from lit_nlp.api import types as lit_types
from typing import Any, Optional, Sequence, Callable

from lit_nlp.api import components as lit_components
from lit_nlp.api import dataset as lit_dataset
from lit_nlp.api import model as lit_model
import numpy as np
from absl import logging

from datasets import load_metric

JsonDict = lit_types.JsonDict
IndexedInput = lit_types.IndexedInput
LitType = lit_types.LitType
Spec = lit_types.Spec
MetricsDict = dict[str, float]


def map_pred_keys(
        data_spec: Spec, model_output_spec: Spec,
        predicate: Callable[[LitType, Optional[LitType]], bool]) -> dict[str, str]:
    """Returns a map of compatible output fields and their parent input fields."""
    ret = {}
    for pred_key, pred_spec in model_output_spec.items():
        parent_key: Optional[str] = getattr(pred_spec, 'parent', None)
        if parent_key is None:
            logging.info("Skipping '%s': No parent provided.", pred_key)
            continue

        parent_spec: Optional[LitType] = data_spec.get(parent_key)
        if predicate(pred_spec, parent_spec):
            ret[pred_key] = parent_key
        else:
            logging.info("Skipping '%s': incompatible parent '%s'.", pred_key,
                         parent_key)
            continue
    return ret


def nan_to_none(metrics: dict[str, float]) -> dict[str, Optional[float]]:
    return {k: (v if not np.isnan(v) else None) for k, v in metrics.items()}


class BertScore(lit_components.Metrics):

    def __init__(self):
        super(BertScore, self).__init__()
        self.bertscore = load_metric('bertscore')

    def is_compatible(self, model: lit_model.Model,
                      dataset: lit_dataset.Dataset) -> bool:
        return True

    def meta_spec(self):
        return {
            'bertscore-f1': lit_types.MetricResult(
                best_value=lit_types.MetricBestValue.HIGHEST,
                description='Similarity between the reference and the generated output'
                            ', calculated using BERT-embeddings'),
        }

    def run(
            self,
            inputs: Sequence[JsonDict],
            model: lit_model.Model,
            dataset: lit_dataset.Dataset,
            model_outputs: Optional[list[JsonDict]] = None,
            config: Optional[JsonDict] = None) -> list[JsonDict]:

        if model_outputs is None:
            model_outputs = list(model.predict(inputs))

        spec = model.spec()
        field_map = map_pred_keys()

        ret = []
        for pred_key, label_key in field_map.items():
            labels = [ex[label_key] for ex in inputs]
            preds = [mo[pred_key] for mo in model_outputs]
            # Compute metrics, as dict(str -> float)
            metrics = self.compute(
                labels,
                preds,
                label_spec=dataset.spec()[label_key],
                pred_spec=spec.output[pred_key],
                config=config.get(pred_key) if config else None)
            # Format for frontend.
            ret.append({
                'pred_key': pred_key,
                'label_key': label_key,
                'metrics': nan_to_none(metrics)
            })
        return ret

    def run_with_metadata(
            self,
            indexed_inputs: Sequence[IndexedInput],
            model: lit_model.Model,
            dataset: lit_dataset.IndexedDataset,
            model_outputs: Optional[list[JsonDict]] = None,
            config: Optional[JsonDict] = None) -> list[JsonDict]:
        inputs = [inp['data'] for inp in indexed_inputs]
        return self.run(inputs, model, dataset, model_outputs, config)

    def is_field_compatible(
            self,
            pred_spec: lit_types.LitType,
            parent_spec: Optional[lit_types.LitType]) -> bool:
        return True

    def compute(
            self,
            labels: Sequence[Any],
            preds: Sequence[Any],
            label_spec: lit_types.LitType,
            pred_spec: lit_types.LitType,
            config: Optional[JsonDict] = None) -> MetricsDict:

        bertscore_output = self.bertscore.compute(
            predictions=preds, references=labels, lang='en', model_type='bert-large-uncased'
        )
        return {'bertscore-f1': np.mean(bertscore_output['f1'])}

    def compute_with_metadata(
            self,
            labels: Sequence[Any],
            preds: Sequence[Any],
            label_spec: lit_types.LitType,
            pred_spec: lit_types.LitType,
            indices: Sequence[lit_types.ExampleId],
            metas: Sequence[JsonDict],
            config: Optional[JsonDict] = None) -> MetricsDict:

        del indices, metas  # unused by Metrics base class
        return self.compute(labels, preds, label_spec, pred_spec, config)