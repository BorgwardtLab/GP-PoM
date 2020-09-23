"""Callbacks for training loop."""
from collections import defaultdict
import gpytorch
import numpy as np
import os
from sklearn.metrics import roc_auc_score as auc
from sklearn.metrics import average_precision_score as auprc
import torch
from tqdm import tqdm

from exp.utils import augment_labels, convert_to_base_type, compute_loss


# Hush the linter, child callbacks will always have different parameters than
# the overwritten method of the parent class. Further kwargs will mostly be an
# unused parameter due to the way arguments are passed.
# pylint: disable=W0221,W0613
class Callback:
    """Callback for training loop."""

    def on_epoch_begin(self, **local_variables):
        """Call before an epoch begins."""

    def on_epoch_end(self, **local_variables):
        """Call after an epoch is finished."""

    def on_batch_begin(self, **local_variables):
        """Call before a batch is being processed."""

    def on_batch_end(self, **local_variables):
        """Call after a batch has be processed."""

    def on_train_end(self, **local_variables):
        """Call after training is finished."""


class Progressbar(Callback):
    """Callback to show a progressbar of the training progress."""

    def __init__(self):
        """Show a progressbar of the training progress."""
        self.total_progress = None
        self.epoch_progress = None

    def on_epoch_begin(self, n_epochs, n_instances, **kwargs):
        """Initialize the progressbar."""
        if self.total_progress is None:
            self.total_progress = tqdm(position=0, total=n_epochs, unit='epochs')
        self.epoch_progress = tqdm(position=1, total=n_instances, unit='instances')

    def _description(self, loss):
        description = f'Loss: {loss:3.3f}'
        return description

    def on_batch_end(self, batch_size, loss, virtual_batch_size, **kwargs):
        """Increment progressbar and update description."""
        if virtual_batch_size is not None:
            batch_size = virtual_batch_size
        self.epoch_progress.update(batch_size)
        description = self._description(loss)
        self.epoch_progress.set_description(description)

    def on_epoch_end(self, epoch, n_epochs, **kwargs):
        """Increment total training progressbar."""
        self.epoch_progress.close()
        self.epoch_progress = None
        self.total_progress.update(1)
        if epoch == n_epochs:
            self.total_progress.close()


class LogTrainingLoss(Callback):
    """Logging of loss during training into sacred run."""

    def __init__(self, run, print_progress=False):
        """Create logger callback.

        Log the training loss using the sacred metrics API.

        Args:
            run: Sacred run
        """
        self.run = run
        self.print_progress = print_progress
        self.epoch_losses = None
        self.logged_averages = defaultdict(list)
        self.logged_stds = defaultdict(list)
        self.iterations = 0

    def _description(self):
        all_keys = self.logged_averages.keys()
        elements = []
        for key in all_keys:
            last_average = self.logged_averages[key][-1]
            last_std = self.logged_stds[key][-1]
            elements.append(f'{key}: {last_average:3.3f} +/- {last_std:3.3f}')
        return ' '.join(elements)

    def on_epoch_begin(self, **kwargs):
        self.epoch_losses = defaultdict(list)

    def on_batch_end(self, loss, **kwargs):
        loss = convert_to_base_type(loss)
        self.iterations += 1
        self.epoch_losses['training.loss'].append(loss)
        self.run.log_scalar('training.loss.batch', loss, self.iterations)

    def on_epoch_end(self, epoch, **kwargs):
        for key, values in self.epoch_losses.items():
            mean = np.mean(values)
            std = np.std(values)
            self.run.log_scalar(key + '.mean', mean, self.iterations)
            self.logged_averages[key].append(mean)
            self.run.log_scalar(key + '.std', std, self.iterations)
            self.logged_stds[key].append(std)
        self.epoch_losses = defaultdict(list)
        if self.print_progress:
            print(f'Epoch {epoch}:', self._description())


class LogDatasetLoss(Callback):
    """Logging of loss and other eval measures during and after training into sacred run."""

    def __init__(self, dataset_name, dataset, data_format, collate_fn, loss_fn, run, 
                 imputation_params, batch_size=64, early_stopping=None, save_path=None,
                 device='cpu', print_progress=True, num_workers=4, drop_last=False):
        """Create logger callback.

        Log the training loss using the sacred metrics API.

        Args:
            dataset_name: Name of dataset
            dataset: Dataset to use
            collate_fn: dataset-specific collate function (depends on input dimension)
            loss_fn: loss function object to use
            run: Sacred run
            print_progress: Print evaluated loss
            batch_size: Batch size
            max_root: max_root_decomposition_size (rank param for GP)
            n_mc_smps: number of mc samples (1 if no additional sampling used)
            early_stopping: if int the number of epochs to wait befor stopping
                training due to non-decreasing loss, if None dont use
                early_stopping
            save_path: Where to store model weigths
        """
        self.prefix = dataset_name
        self.dataset = dataset
        self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, collate_fn=collate_fn,
                                                       pin_memory=True, num_workers=num_workers, drop_last=drop_last)
        self.data_format = data_format
        self.loss_fn = loss_fn
        self.run = run
        self.print_progress = print_progress
        self.imputation_params = imputation_params
        self.early_stopping = early_stopping
        self.save_path = save_path
        self.device = device
        self.iterations = 0
        self.patience = 0
        self.best_loss = np.inf
        self.logged_averages = defaultdict(list) #we now add logging of all epoch-wise evaluation metrics (auprc, auroc etc) during training
    
    def _compute_eval_measures(self, model, full_eval=False):
        losses = defaultdict(list)
        model.eval()

        if full_eval:
            y_true_total = []
            y_score_total = []

        for d in self.data_loader:
            loss, logits, y_true = compute_loss(d, self.data_format, self.device, model, self.loss_fn,
                                                callbacks=[], imputation_params=self.imputation_params)
            loss = convert_to_base_type(loss)
            losses['loss'].append(loss)

            if full_eval:
                with torch.no_grad():
                    y_true = y_true.detach().cpu().numpy()
                    y_score = logits.detach().cpu().numpy() 
                    y_true_total.append(y_true)
                    y_score_total.append(y_score)
        return_dict = {}
        
        average_loss = np.mean(losses['loss'])

        return_dict['loss'] = average_loss  
        if full_eval: 
            y_true_total = np.concatenate(y_true_total)
            y_score_total = np.concatenate(y_score_total)
            if y_score_total.shape[-1] == 1:
                y_score_total = y_score_total.squeeze(-1)
            for measure_name, measure in self.dataset.task.metrics.items():
                return_dict[measure_name] = measure(y_true_total, y_score_total)
        return return_dict
 
    def _progress_string(self, epoch, losses):
        progress_str = " ".join([
            f'{self.prefix}.{key}: {value:.3f}'
            for key, value in losses.items()
        ])
        return f'Epoch {epoch}: ' + progress_str

    def on_batch_end(self, **kwargs):
        self.iterations += 1

    def on_epoch_begin(self, model, epoch, **kwargs):
        """Store the loss on the dataset prior to training."""
        if epoch == 1:  # This should be prior to the first training step
            losses = self._compute_eval_measures(model)
            if self.print_progress:
                print(self._progress_string(epoch - 1, losses))

            for key, value in losses.items():
                self.run.log_scalar(
                    f'{self.prefix}.{key}',
                    value,
                    self.iterations
                )

    def on_epoch_end(self, model, epoch, **kwargs):
        """Score evaluation metrics at end of epoch."""
        losses = self._compute_eval_measures(model, full_eval = True)
        for key, value in losses.items():
            self.logged_averages[key].append(value)
    
        if self.prefix != 'testing':
            print(self._progress_string(epoch, losses))
        for key, value in losses.items():
            self.run.log_scalar(
                f'{self.prefix}.{key}',
                value,
                self.iterations
            )
        if self.early_stopping is not None:
            if losses['loss'] < self.best_loss:
                self.best_loss = losses['loss']
                if self.save_path is not None:
                    save_path = os.path.join(self.save_path, 'model_state.pth')
                    print('Saving model to', save_path)
                    torch.save(
                        model.state_dict(),
                        save_path
                    )
                self.patience = 0
            else:
                self.patience += 1

            if self.early_stopping <= self.patience:
                print(
                    'Stopping training due to non-decreasing '
                    f'{self.prefix} loss over {self.early_stopping} epochs'
                )
                return True

    def on_train_end(self, model, epoch, **kwargs):
        """Score evaluation metrics at end of training."""
        self.eval_measures = self._compute_eval_measures(model, full_eval=True)
        for key, value in self.eval_measures.items():
            self.run.log_scalar(
                f'{self.prefix}.{key}',
                value,
                self.iterations
            )
