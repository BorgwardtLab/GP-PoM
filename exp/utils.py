import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch

dataset_to_classes = {
    'PenDigits': 10,
    'LSST': 14,
    'CharacterTrajectories': 20 
}

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def augment_labels(labels, n_samples):
    """Expand labels for multiple MC samples in the GP Adapter.

    Args:
         Takes tensor of size [n]

    Returns:
        expanded tensor of size [n_mc_samples, n]

    """
    return labels.expand(labels.shape[0], n_samples).transpose(1, 0)


def plot_losses(losses, losses_std=None, save_file=None):
    """Plot a dictionary with per epoch losses.

    Args:
        losses: Mean of loss per epoch
        losses_std: stddev of loss per epoch

    """
    for key, values in losses.items():
        if losses_std is not None:
            plt.errorbar(range(len(values)), values, yerr=losses_std[key], label=key)
        else:
            plt.plot(range(len(values)), values, label=key)
    plt.xlabel('# epochs')
    plt.ylabel('loss')
    plt.legend()
    if save_file:
        plt.savefig(save_file, dpi=200)
        plt.close()


def convert_to_base_type(value):
    """Convert a value into a python base datatype.

    Args:
        value: numpy or torch value

    Returns:
        Python base type
    """
    if isinstance(value, (torch.Tensor, np.generic)):
        return value.item()
    else:
        return value


def execute_callbacks(callbacks, hook, local_variables):
    stop = False
    for callback in callbacks:
        # Convert return value to bool --> if callback doesn't return
        # anything we interpret it as False
        stop |= bool(getattr(callback, hook)(**local_variables))
    return stop


def compute_loss(d, data_format, device, model, loss_fn, callbacks, imputation_params):
    # if we use mc sampling, expand labels to match multiple predictions
    y_true = d['label']
    valid_lengths = d['valid_lengths'].to(device)
    #handling data_format-specific cases:  
    if data_format == 'GP':
        n_mc_smps = imputation_params['n_mc_smps']
        max_root = imputation_params['max_root']
        if n_mc_smps > 1:
            y_true = augment_labels(d['label'], n_mc_smps)
        # GP format of data:
        inputs = d['inputs'].to(device)
        indices = d['indices'].to(device)
        test_inputs = d['test_inputs'].to(device)
        test_indices = d['test_indices'].to(device)
        values = d['values'].to(device)
    elif data_format in ('zero', 'linear', 'forwardfill', 'causal', 'indicator'):
        #in case we use other imputation scheme, we feed the irregular time steps to the model (via values)
        values = torch.cat([d['values'],d['time']], dim=2) #the time is treated as additional channel
        values = values.to(device)
    else:
        raise ValueError('Not understood data_format: {}'.format(data_format))

    execute_callbacks(callbacks, 'on_batch_begin', locals())

    #model.train()

    if data_format == 'GP':
        with gpytorch.settings.fast_pred_var(), gpytorch.settings.max_root_decomposition_size(max_root), gpytorch.settings.max_cholesky_size(20): #use this to strictly enforce lanczos
            logits = model(inputs, indices, values, test_inputs, test_indices, valid_lengths)
    elif data_format in ('zero', 'linear', 'forwardfill', 'causal', 'indicator'):
        logits = model(values, valid_lengths)
    else:
        raise ValueError('Not understood data_format: {}'.format(data_format))

    y_true = y_true.flatten().to(device)
    if logits.shape[1] == 1:
        logits = logits.squeeze(-1)
    #in case of multi-class setting (using CrossEntropyLoss), make sure we use long()
    if 'CrossEntropyLoss' in str(loss_fn):
        y_true = y_true.long()
    loss = loss_fn(logits, y_true)
    return loss, logits, y_true
