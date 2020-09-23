"""
Offers various kinds of imputation.

The convention throughout is that the input is a dictionary 'batch' with three keys, 'time', 'values', 'labels'.
    (a) batch['time'] should be a tensor of shape (batch, stream, 1)
    (b) batch['values'] should be a tensor of shape (batch, stream, channels)
    (c) batch['labels'] should be a tensor of shape (batch, 1)

Missing values should be represented as a NaN.

Padding to a common length should be done with zeros. (Not with NaNs!!!)
"""


import torch


def zero_imputation(batch):
    """Simple zero imputation; just replaces every NaN with zero.

    Note that this corresponds to mean imputation if all of the inputs are standardized first.

    Returns:
        A Python dictionary with three keys, 'time', 'values', 'labels'.
            (a) batch['time'] will be a tensor of shape (batch, stream, 1)
            (b) batch['values'] will be a tensor of shape (batch, stream, channels)
            (c) batch['labels'] will be a tensor of shape (batch, 1)
    """

    imputed_values = batch['values'].clone()
    imputed_values[torch.isnan(imputed_values)] = 0

    batch = dict(batch)  # copy
    batch['values'] = imputed_values
    return batch


def linear_imputation(batch):
    """Imputes by linearly interpolating. Note that as a result, this is noncausal.

    This won't fill in any NaN values that are at the start or the end, i.e. those NaNs without observations on *both*
    sides. As a result you may wish to run forward and backward fills, or zero imputation, afterwards.

    Returns:
        A Python dictionary with three keys, 'time', 'values', 'labels'.
            (a) batch['time'] will be a tensor of shape (batch, stream, 1)
            (b) batch['values'] will be a tensor of shape (batch, stream, channels)
            (c) batch['labels'] will be a tensor of shape (batch, 1)
    """

    time = batch['time']  # of shape (batch, stream, 1)
    values = batch['values']  # of shape (batch, stream, channels)

    time_slices = time.unbind(1)
    value_slices = values.unbind(1)

    # Record the first time/value combinations
    prev_before_value = value_slices[0]
    prev_before_time = time_slices[0].expand_as(prev_before_value)
    before_times = [prev_before_time]
    before_values = [prev_before_value]
    for time_slice, value_slice in zip(time_slices[1:], value_slices[1:]):
        # At each subsequent time step:
        #   If there's a NaN for a particular entry:
        #       Record the last time/value combination that we had
        #   Else:
        #       Record the new time/value combination for where we are
        no_update = torch.isnan(value_slice)
        before_time = torch.where(no_update, prev_before_time, time_slice)
        before_value = torch.where(no_update, prev_before_value, value_slice)
        before_times.append(before_time)
        before_values.append(before_value)
        prev_before_time = before_time
        prev_before_value = before_value
    # In this way, for each batch, stream, channel and channel index, then before_times and before_values are recording
    # the last time and value for which an entry in that batch element and channel was non-NaN

    # Now repeat going the other way, to determine the next (in the future) time and value for which an entry is non-NaN
    prev_after_value = value_slices[-1]
    prev_after_time = time_slices[-1].expand_as(prev_after_value)
    after_times = [prev_after_time]
    after_values = [prev_after_value]
    for time_slice, value_slice in zip(time_slices[-2::-1], value_slices[-2::-1]):
        no_update = torch.isnan(value_slice)
        after_time = torch.where(no_update, prev_after_time, time_slice)
        after_value = torch.where(no_update, prev_after_value, value_slice)
        after_times.append(after_time)
        after_values.append(after_value)
        prev_after_time = after_time
        prev_after_value = after_value
    after_times.reverse()
    after_values.reverse()

    before_times = torch.stack(before_times, dim=1)
    before_values = torch.stack(before_values, dim=1)
    after_times = torch.stack(after_times, dim=1)
    after_values = torch.stack(after_values, dim=1)
    # Each of these is now a tensor of shape (batch, stream, channel), for each element recording the most recent and
    # most soon times and values for which a batch/channel combination will be non-NaN

    # Then it's just simple arithmetic to do linear interpolation
    time_reciprocal = (after_times - before_times).reciprocal()
    before_coeff = (after_times - time) * time_reciprocal
    after_coeff = (time - before_times) * time_reciprocal
    imputed_values = before_values * before_coeff + after_values * after_coeff
    # In the case that we actually have data at a point then ironically the above formula gives a NaN.
    imputed_values = torch.where(torch.isnan(imputed_values), values, imputed_values)

    batch = dict(batch)  # copy
    batch['values'] = imputed_values
    return batch


def forward_fill_imputation(batch):
    """Simple forward-fill imputation. Every NaN value is replaced with the most recent non-NaN value. (And is left at
    NaN if there are no preceding non-NaN values.)

    Returns:
        A Python dictionary with three keys, 'time', 'values', 'labels'.
            (a) batch['time'] will be a tensor of shape (batch, stream, 1)
            (b) batch['values'] will be a tensor of shape (batch, stream, channels)
            (c) batch['labels'] will be a tensor of shape (batch, 1)
    """

    values = batch['values']  # tensor of shape (batch, stream, channels)

    # Now forward-fill impute the missing values
    imputed_values = values.clone()
    value_slices = iter(imputed_values.unbind(dim=1))
    prev_value_slice = next(value_slices)
    for value_slice in value_slices:
        nan_mask = torch.isnan(value_slice)
        value_slice.masked_scatter_(nan_mask, prev_value_slice.masked_select(nan_mask))
        prev_value_slice = value_slice

    batch = dict(batch)  # copy
    batch['values'] = imputed_values
    return batch


def backward_fill_imputation(batch):
    """Simple backward-fill imputation. Every NaN value is replaced with the next-to-occur non-NaN value. (And is left
    as NaN if there are no upcoming non-NaN values.)

    Returns:
        A Python dictionary with three keys, 'time', 'values', 'labels'.
            (a) batch['time'] will be a tensor of shape (batch, stream, 1)
            (b) batch['values'] will be a tensor of shape (batch, stream, channels)
            (c) batch['labels'] will be a tensor of shape (batch, 1)
    """

    time = batch['time']
    values = batch['values']

    final_time_index = time.argmax(dim=1)

    imputed_values = values.clone()
    value_slices = iter(reversed(list(enumerate(imputed_values.unbind(dim=1)))))
    _, prev_value_slice = next(value_slices)
    for index, value_slice in value_slices:
        nan_mask = torch.isnan(value_slice) & (index < final_time_index)
        value_slice.masked_scatter_(nan_mask, prev_value_slice.masked_select(nan_mask))
        prev_value_slice = value_slice

    batch = dict(batch)  # copy
    batch['values'] = imputed_values
    return batch


def causal_imputation(batch):
    """Performs causal imputation on the batch. It's a bit like forward-fill imputation, except that changes explicitly
    occur instantaneously.

    Suppose we have a sequence of observations, (t_1, x_1, y_1, z_1), ..., (t_n, x_n, y_n, z_n), where t_i are the
    timestamps, and x_i, y_i, z_i are three channels that are observed. The different timestamps mean that this data is
    potentially irregularly sampled. Furthermore each x_i, y_i or z_i may be NaN, to represent no observation in that
    channel at that time, so the data may potentially also be partially observed. Suppose for example that the t_i, x_i
    pairs look like this:

    t_1 t_2 t_3
    x_1 NaN x_3

    (Where t_2 is presumably included because there is an observation for y_2 or z_2, but we don't show that here.)

    Then the causal imputation scheme first does simple forward-fill imputation:

    t_1 t_2 t_3
    x_1 x_1 x_3

    and then duplicates and interleaves the time and channel observations, to get:

    t_1 t_2 t_2 t_3 t_3
    x_1 x_1 x_1 x_1 x_3

    The forward-fill imputation preserves causality. The interleaving of time and channel changes means that when a
    change does occur, it does so instantaneously. In the example above, x changes from x_1 to x_3 without the value of
    t increasing.

    So for example if multiple channels are present:

    t_1 t_2 t_3 t_4
    x_1 NaN x_3 x_4
    NaN y_2 NaN y_4

    this becomes:

    t_1 t_2 t_2 t_3 t_3 t_4 t_4
    x_1 x_1 x_1 x_1 x_3 x_3 x_4
    NaN NaN y_2 y_2 y_2 y_2 y_4

    Note that we don't try to impute any NaNs at the start. You may wish to back-fill those.

    Returns:
        A Python dictionary with three keys, 'time', 'values', 'labels'.
            (a) batch['time'] will be a tensor of shape (batch, 2 * stream - 1, 1)
            (b) batch['values'] will be a tensor of shape (batch, 2 * stream - 1, channels)
            (c) batch['labels'] will be a tensor of shape (batch, 1)
    """

    time = batch['time']

    # Start off by forward-fill imputing the missing values
    imputed_values = forward_fill_imputation(batch)['values']

    # For the times, we want to repeat every time twice, and then drop the first (repeated) time.
    imputed_time = time.repeat_interleave(2, dim=1)
    imputed_time = imputed_time[:, 1:]

    # For the values, we want to repeat every value twice, and then drop the last (repeated) value.
    # This is a bit finickity because of the zero-padding of the shorter batch elements.
    imputed_values = imputed_values.repeat_interleave(2, dim=1)
    final_time_index = time.argmax(dim=1)
    final_time_index *= 2
    final_time_index += 1
    final_time_index = final_time_index.unsqueeze(1).expand(imputed_values.size(0), 1, imputed_values.size(2))
    imputed_values.scatter_(1, final_time_index, 0)
    imputed_values = imputed_values[:, :-1]

    batch = dict(batch)  # copy
    batch['time'] = imputed_time
    batch['values'] = imputed_values
    return batch


def indicator_imputation(batch):
    """Simple indicator "imputation"; mark the missing values in a separate channel, and set the missing value to zero.

    Returns:
        A Python dictionary with three keys, 'time', 'values', 'labels'.
            (a) batch['time'] will be a tensor of shape (batch, stream, 1)
            (b) batch['values'] will be a tensor of shape (batch, stream, 2 * channels)
            (c) batch['labels'] will be a tensor of shape (batch, 1)
    """

    values = batch['values']
    indicators = torch.isnan(values)

    batch_, stream, channels = values.shape
    imputed_values = torch.empty(batch_, stream, 2 * channels, dtype=values.dtype, device=values.device)
    imputed_values_no_indicator = imputed_values[:, :, :channels]
    imputed_values_no_indicator.copy_(values)
    imputed_values_no_indicator[indicators] = 0
    imputed_values[:, :, channels:].copy_(indicators)

    batch = dict(batch)  # copy
    batch['values'] = imputed_values
    return batch


class ImputationStrategy:
    """
    Main class for encapsulating different imputation strategies and
    making them mesh well with a dataset class.
    """

    def __init__(self, strategy='zero', ensure_zero_imputation=True):
        '''
        Creates a new imputation scheme based on a pre-defined strategy
        that can be applied to individual instances of a data set class
        on demand.

        Parameters
        ----------

            strategy: One value of ['zero', 'linear', 'forward_fill',
                      'backward_fill', 'causal', 'indicator', 'GP']. This
                      determines the imputation strategy. For GP we want 
                      no preprocessed imputation, so we set the strategy 
                      to 'inactive' in this case.

            ensure_zero_imputation: If set, will always apply zero-based
            imputation after any scheme, thus ensuring that no NaNs will
            remain in the data (except for inactive GP mode!).
            
            CAVE: the variable ensure_zero_imputation is by default True
            and for readibility not included in the cache file path. When 
            setting it to false, all the cached files have to be recomputed! 
        '''
        def inactive(x):
            return x

        strategy_to_fn = {
            'zero': zero_imputation,
            'linear': linear_imputation,
            'forwardfill': forward_fill_imputation,
            'backwardfill': backward_fill_imputation,
            'causal': causal_imputation,
            'indicator': indicator_imputation,
            'inactive' : inactive  #here we don't want any preprocessing imputation
        }

        # Report available strategies in order to make this class
        # configurable from outside.
        self.available_strategies = sorted(strategy_to_fn.keys())
        if strategy == 'GP':
            strategy = 'inactive' #map the GP format to an inactive imputation (for preprocessing) 
        self.strategy = strategy
        self.strategy_fn = strategy_to_fn[strategy]

        self.ensure_zero_imputation = ensure_zero_imputation

    def __repr__(self):
        '''
        Returns a string-based representation of the class, which will
        be useful when creating output filenames.
        '''

        return __name__ + '_' + self.strategy

    def __call__(self, instance, index):

        # Apply conversions to tensors because the imputation strategies
        # require this. This should be a no-op for tensors.
        instance['time'] = torch.Tensor(instance['time']).unsqueeze(0)
        instance['values'] = torch.Tensor(instance['values']).unsqueeze(0)
        instance['label'] = torch.Tensor(instance['label']).unsqueeze(0)

        instance = self.strategy_fn(instance) #the strategies are implemented for torch tensors
        #however, to speed up, we apply them now as a prepro step still in numpy format (reformat again here)

        if self.ensure_zero_imputation and self.strategy != 'inactive':
            instance = zero_imputation(instance)

        #Reformat to numpy (since we impute still before data loader as prepro step in numpy )
        instance = {key: value.squeeze(0).numpy() for key,value in instance.items()}
        #instance['time'] = instance['time'].squeeze(-1) #to stay consistent with other formats        

        return instance
