import torch.nn as nn
import gpytorch

from src.models.mgp import GPAdapter
from src.models.rnn_models import GRU, LSTM

class GPRNNModel(nn.Module):
    """
    GP Adapter combined with a RNN
    """

    def __init__(self, n_input_dims, out_dimension, sampling_type, n_mc_smps, n_devices, output_device, sig_depth=2,
                 kernel='rbf', mode='normal', keops=False, hidden_size=32, rnn_type='gru'):
        super(GPRNNModel, self).__init__()
        
        #safety guard:
        self.sampling_type = sampling_type
        if self.sampling_type == 'moments':
            n_mc_smps = 1 
            # the classifier receives mean and variance of GPs posterior
            clf_input_dims = 2*n_input_dims
        else:
            clf_input_dims = n_input_dims

        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(output_device, non_blocking=True)
        if rnn_type == 'gru':
            clf_class = GRU
        elif rnn_type == 'lstm':
            clf_class = LSTM
        else:
            raise ValueError('No valid RNN type provided [gru, lstm]')
        
        clf = clf_class(out_channels=out_dimension,
                        input_size=clf_input_dims,
                        hidden_size=hidden_size
        ) 
        
        self.model = GPAdapter(clf,
                               None,
                               n_mc_smps,
                               sampling_type,
                               likelihood,
                               n_input_dims + 1,
                               n_devices,
                               output_device,
                               kernel,
                               mode,
                               keops
        )

    def forward(self, *data):
        return self.model(*data)

