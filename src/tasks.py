"""Definitions of possible tasks."""
import abc
import torch
import src.evaluation_metrics as metrics_module


class Task(abc.ABC):
    @property
    @abc.abstractmethod
    def loss(self):
        pass

    @property
    @abc.abstractmethod
    def n_outputs(self):
        pass

    @property
    @abc.abstractmethod
    def metrics(self):
        pass


class BinaryClassification(Task):
    @property
    def loss(self):
        return torch.nn.BCEWithLogitsLoss()

    @property
    def n_outputs(self):
        return 1

    @property
    def metrics(self):
        # TODO: Extend by further metrics
        return {
            'auprc': metrics_module.auprc,
            'auroc': metrics_module.auroc,
        }

class MulticlassClassification(Task):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        if self.n_classes < 3:
            import warnings
            warnings.warn('Current dataset is not multiclass, however Multiclass Task is selected!')

    @property
    def loss(self):
        return torch.nn.CrossEntropyLoss()
    
    @property
    def n_outputs(self):
        return self.n_classes

    @property
    def metrics(self):
        return {
            'auroc_weighted': metrics_module.auroc_weighted,
            'balanced_accuracy': metrics_module.balanced_accuracy,
            'accuracy': metrics_module.accuracy
        }

