from functools import partial
from typing import Dict

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, \
    jaccard_score, cohen_kappa_score, matthews_corrcoef, fbeta_score, hamming_loss, zero_one_loss

from ..api.typing import Metric, MetricOrName

__METRICS: Dict[str, Metric] = {
    "accuracy": accuracy_score,
    "balanced_accuracy": balanced_accuracy_score,
    "bal_accuracy": balanced_accuracy_score,
    "bal_acc": balanced_accuracy_score,
    "f1": f1_score,
    "f1_micro": partial(f1_score, average='micro'),
    "f1_macro": partial(f1_score, average='macro'),
    "f1_weighted": partial(f1_score, average='weighted'),
    "precision": precision_score,
    "precision_micro": partial(precision_score, average='micro'),
    "precision_macro": partial(precision_score, average='macro'),
    "precision_weighted": partial(precision_score, average='weighted'),
    "recall": recall_score,
    "recall_micro": partial(recall_score, average='micro'),
    "recall_macro": partial(recall_score, average='macro'),
    "recall_weighted": partial(recall_score, average='weighted'),
    "jaccard": jaccard_score,
    "jaccard_micro": partial(jaccard_score, average='micro'),
    "jaccard_macro": partial(jaccard_score, average='macro'),
    "jaccard_weighted": partial(jaccard_score, average='weighted'),
    "cohen_kappa": cohen_kappa_score,
    "matthews_correlation": matthews_corrcoef,
    "fbeta": fbeta_score,
    "fbeta_micro": partial(fbeta_score, average='micro'),
    "fbeta_macro": partial(fbeta_score, average='macro'),
    "fbeta_weighted": partial(fbeta_score, average='weighted'),
    "hamming": hamming_loss,
    "zero_one": zero_one_loss
}

__ALIAS: Dict[str, str] = {
    "acc": "accuracy",
    "bal_accuracy": "balanced_accuracy",
    "bal_acc": "balanced_accuracy",
    "f1_score": "f1",
    "prec": "precision",
    "prec_micro": "precision_micro",
    "prec_macro": "precision_macro",
    "prec_weighted": "precision_weighted",
    "recl": "recall",
    "recl_micro": "recall_micro",
    "recl_macro": "recall_macro",
    "recl_weighted": "recall_weighted",
    "jacc": "jaccard",
    "jacc_micro": "jaccard_micro",
    "jacc_macro": "jaccard_macro",
    "jacc_weighted": "jaccard_weighted",
    "cohen": "cohen_kappa",
    "matthews": "matthews_correlation",
    "matthews_coef": "matthews_correlation",
    "matthews_correllation_coef": "matthews_correlation",
    "matthews_corrcoef": "matthews_correlation",
    "fbeta_score": "fbeta",
    "hamming_loss": "hamming",
    "zero_one_loss": "zero_one"
}


def resolve_metric(metric: MetricOrName, **metric_kwargs) -> Metric:
    if isinstance(metric, str):
        cleaned_metric = metric.lower().replace('-', '_')

        if cleaned_metric not in __METRICS and cleaned_metric not in __ALIAS:
            raise ValueError(f"Metric '{metric}' could not be resolved.")
        if cleaned_metric in __ALIAS:
            cleaned_metric = __ALIAS[cleaned_metric]

        return resolve_metric(__METRICS[cleaned_metric], **metric_kwargs)
    else:
        if len(metric_kwargs) > 0:
            metric = partial(metric, **metric_kwargs)

        return metric
