"""Module containing convinience wrapped evaluation metrics."""
import numpy as np
from functools import partial

from sklearn.metrics import average_precision_score, roc_auc_score,\
    balanced_accuracy_score, accuracy_score 

def to_one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def score_to_pred(score):
    """converts np score array (samples, classes) to array of predictions (samples,)"""
    return np.argmax(score,axis=-1)

def prediction_wrapper(metric, **kwargs):
    """
    By default our methods output logits and scores, however certain metrics require label predictions as input.
    Here, we convert scores to predictions and apply the metric to the predicted labels.
    """
    def wrapped(y_true, y_score):
        if y_score.ndim > 1:
            if y_score.ndim > 2:
                raise NotImplementedError(f'Currently only 2-dimensional scores implemented, but {y_score.ndim} provided.')
            y_pred = score_to_pred(y_score)
        else:
            y_pred = y_score
        return metric(y_true, y_pred, **kwargs)
    return wrapped

def batched_softmax(x):
    "util function to convert logits / unnormalized scores (as np arrays of shape: [samples, classes] ) to softmax probabilities"
    num = np.exp(x - np.max(x, axis=-1, keepdims=True)) 
    denom = np.sum(num, axis=1)
    denom = np.tile(denom, [x.shape[-1],1]).T
    return num / denom

def probability_wrapper(metric, **kwargs):
    """
    By default our methods output logits and scores, however certain metrics require normalized probability scores (suming to 1).
    Here, we convert scores to probabilities and apply the metric to the transformed scores.
    """
    def wrapped(y_true, y_score):
        y_probs = batched_softmax(y_score) 
        return metric(y_true, y_probs, **kwargs)
    return wrapped

 
#standard metrics for binary setting:
auroc = roc_auc_score
auprc = average_precision_score

#metrics extensions to multi-class:
auroc_weighted = probability_wrapper(roc_auc_score, average='weighted', multi_class='ovo') #wrapper converts scores to probabilities, 
# which is required for multiclass auroc implementation 
balanced_accuracy = prediction_wrapper(balanced_accuracy_score)
accuracy = prediction_wrapper(accuracy_score)


