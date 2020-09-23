"""Module to train a model with a dataset configuration."""

import gpytorch
import os
from sacred import Experiment, SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch

sys.path.append(os.getcwd())

from exp.callbacks import Callback, Progressbar, LogDatasetLoss, LogTrainingLoss
from exp.format import get_input_transform, get_collate_fn, get_subsampler, get_imputation_scheme
from exp.ingredients import dataset_config, model_config
from exp.utils import count_parameters, plot_losses, execute_callbacks, compute_loss, dataset_to_classes


# Testing to overwrite cudnn backend:
torch.backends.cudnn.benchmark = False

# Workaround for sacred read-only error
SETTINGS.CONFIG.READ_ONLY_CONFIG = False
SETTINGS['CAPTURE_MODE'] = 'sys' #workaround for sdtout timeout

EXP = Experiment('training', ingredients=[model_config.ingredient, dataset_config.ingredient])
EXP.captured_out_filter = apply_backspaces_and_linefeeds

@EXP.config
def cfg():
    n_epochs = 50
    batch_size = 32
    virtual_batch_size = None
    learning_rate = 5e-4
    weight_decay = 1e-4
    early_stopping = 20
    data_format = 'GP'
    device = 'cuda:0'
    quiet = False
    evaluation = {'active': False, 'evaluate_on': 'validation'}
    imputation_params = {'n_mc_smps': 10, #number of monte carlo samples
                         'max_root': 25, #max_root_decomposition_size for MGP lanczos iters
                         'grid_spacing': 1. # determines n_hours between query points
                        }                              
    subsampler_name = 'MissingAtRandomSubsampler' 
    subsampler_parameters = {'probability': 0.1}
    num_workers=1
    drop_last=False
    n_params_limit=1.5e6

# Named configs for Subsampling schemes (only for UEA)
@EXP.named_config
def MissingAtRandomSubsampler():
    subsampler_name = 'MissingAtRandomSubsampler' 
    subsampler_parameters = {
        'probability': 0.5
    }

@EXP.named_config
def LabelBasedSubsampler():
    subsampler_name = 'LabelBasedSubsampler' 
    subsampler_parameters = {
        'probability_ranges': [0.4, 0.6]
    }


# Named configs for setting imputation scheme of train module:
@EXP.named_config
def zero():
    data_format = 'zero'

@EXP.named_config
def forwardfill():
    data_format = 'forwardfill'

@EXP.named_config
def causal():
    data_format = 'causal'

@EXP.named_config
def indicator():
    data_format = 'indicator'

@EXP.named_config
def linear():
    data_format = 'linear'

# Named configs for running repetitions with fixed seeds
@EXP.named_config
def rep1():
    seed = 249040430

@EXP.named_config
def rep2():
    seed = 621965744

@EXP.named_config
def rep3():
    seed = 771860110

@EXP.named_config
def rep4():
    seed = 775293950

@EXP.named_config
def rep5():
    seed = 700134501


class NewlineCallback(Callback):
    """Add newline between epochs for better readability."""
    def on_epoch_end(self, **kwargs):
        print()

def train_loop(model, dataset, data_format, loss_fn, collate_fn, n_epochs, batch_size, virtual_batch_size,
               learning_rate, imputation_params, weight_decay=1e-4, device='cuda:0', callbacks=None, num_workers=0, drop_last=False):
    if callbacks is None:
        callbacks = []

    virtual_scaling = 1
    if virtual_batch_size is not None:
        virtual_scaling = virtual_batch_size / batch_size

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True,
                                               pin_memory=True, num_workers=num_workers, drop_last=drop_last)
    n_instances = len(dataset)
    n_batches = len(train_loader)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    epoch = 1
    for epoch in range(1, n_epochs + 1):
        if execute_callbacks(callbacks, 'on_epoch_begin', locals()):
            break
        optimizer.zero_grad()
        for batch, d in enumerate(train_loader):
            model.train()
            loss, _, _ = compute_loss(d, data_format, device, model, loss_fn, callbacks, imputation_params)

            loss = loss / virtual_scaling
            loss.backward()
            if (batch + 1) % virtual_scaling == 0:
                optimizer.step()
                optimizer.zero_grad()
                execute_callbacks(callbacks, 'on_batch_end', locals())

        if execute_callbacks(callbacks, 'on_epoch_end', locals()):
            break
    execute_callbacks(callbacks, 'on_train_end', locals())

@EXP.automain
def train(n_epochs, batch_size, virtual_batch_size, learning_rate, weight_decay, early_stopping, data_format,
          imputation_params, device, quiet, evaluation, subsampler_name, subsampler_parameters, num_workers,
          drop_last, n_params_limit, _run, _log, _seed, _rnd, dataset):
    """Sacred wrapped function to run training of model."""

    torch.manual_seed(_seed)

    try:
        rundir = _run.observers[0].dir
    except IndexError:
        rundir = None

    # Check if virtual batch size is defined and valid:
    if virtual_batch_size is not None:
        if virtual_batch_size % batch_size != 0:
            raise ValueError(f'Virtual batch size {virtual_batch_size} has to be a multiple of batch size {batch_size}')  
    
    # In case we have a subsampling scenario (UEA), stack all input transforms:
    if subsampler_name is not None:
        if subsampler_name == 'LabelBasedSubsampler':
            # chicken and egg problem: we need n_classes of dataset to specify the subsampling 
            # to initialize the dataset, currently solved with a dictionary that maps the current dataset
            # name to its number of classes 
            subsampler_parameters['n_classes'] = dataset_to_classes[dataset['parameters']['dataset_name']]
            print(f'Subsampler_parameters: {subsampler_parameters}') 
        input_transform = [
            get_subsampler(subsampler_name, subsampler_parameters),
            get_imputation_scheme(data_format),
            get_input_transform(data_format, imputation_params['grid_spacing']) 
        ]
    else:
        # Using a non-UEA dataset (where we only need the data format input transform 
        # (as imputation is already handled in the dataset class)
        input_transform = get_input_transform(data_format, imputation_params['grid_spacing'])

    # Get data, sacred does some magic here so we need to hush the linter
    # pylint: disable=E1120,E1123
    train_dataset = dataset_config.get_instance(split='training', transform=input_transform, data_format=data_format)
    validation_dataset = dataset_config.get_instance(split='validation', transform=input_transform, data_format=data_format)
    test_dataset = dataset_config.get_instance(split='testing', transform=input_transform, data_format=data_format)
    
    # Determine number of input dimensions as GP-Sig models requires this parameter for initialisation
    n_input_dims = train_dataset.measurement_dims
    out_dimension = train_dataset.n_classes
    collate_fn = get_collate_fn(data_format, n_input_dims) 
    
    # In case we use indicator imputation, double the input dim for model:
    if data_format == 'indicator':
        n_input_dims *= 2
    
    # Get model, sacred does some magic here so we need to hush the linter
    # pylint: disable=E1120
    
    model = model_config.get_instance(n_input_dims, out_dimension, device=device)
    n_params = count_parameters(model) 
    print(f'Number of trainable Parameters: {n_params}')
    if n_params > n_params_limit: 
        raise ValueError(f'Number of parameters {n_params} exceeds upper limit {n_params_limit} ')
        
    model.to(device)
   
    # Safety guard, ensure that if mc_sampling is inactive that n_mc_smps are 1 to
    # prevent unwanted label augmentation
    if not hasattr(model, 'sampling_type'):
        imputation_params['n_mc_smps'] = 1     
    elif model.sampling_type != 'monte_carlo':
        imputation_params['n_mc_smps'] = 1 

    loss_fn = train_dataset.task.loss 
    print(f'Using the following loss fn: {str(loss_fn)}')

    callbacks = [
        LogTrainingLoss(_run, print_progress=quiet),
        LogDatasetLoss('validation', validation_dataset, data_format, collate_fn, loss_fn, _run, imputation_params, batch_size,  
                       early_stopping=early_stopping, save_path=rundir, device=device, print_progress=True, num_workers=num_workers, drop_last=drop_last),
        LogDatasetLoss('testing', test_dataset, data_format, collate_fn, loss_fn, _run, imputation_params, batch_size, save_path=rundir, 
                       device=device, print_progress=False, num_workers=num_workers, drop_last=drop_last)
    ]
    if quiet:
        # Add newlines between epochs
        callbacks.append(NewlineCallback())
    else:
        callbacks.append(Progressbar())

    train_loop(model, train_dataset, data_format, loss_fn, collate_fn, n_epochs, batch_size, virtual_batch_size,
               learning_rate, imputation_params, weight_decay, device, callbacks, num_workers, drop_last)

    if rundir:
        # Save model state (and entire model)
        print('Loading model checkpoint prior to evaluation...')
        state_dict = torch.load(os.path.join(rundir, 'model_state.pth'))
        model.load_state_dict(state_dict)
    model.eval()

    logged_averages = callbacks[0].logged_averages
    logged_stds = callbacks[0].logged_stds
    loss_averages = {key: value for key, value in logged_averages.items() if 'loss' in key}
    loss_stds = {key: value for key, value in logged_stds.items() if 'loss' in key}
    if rundir:
        plot_losses(loss_averages, loss_stds, save_file=os.path.join(rundir, 'batch_monitoring.png'))
    monitoring_measures = callbacks[1].logged_averages
    print(monitoring_measures)
    monitoring_measures.update(loss_averages)
    print(monitoring_measures)
    if rundir:
        plot_losses(monitoring_measures, save_file=os.path.join(rundir, 'epoch_monitoring.png'))

    result = {key: values[-1] for key, values in logged_averages.items()}

    if evaluation['active']:
        evaluate_on = evaluation['evaluate_on']
        if evaluate_on == 'validation':
            eval_measures = callbacks[1].eval_measures
        else:
            eval_measures = callbacks[2].eval_measures
        
        result.update(eval_measures)

    return result


