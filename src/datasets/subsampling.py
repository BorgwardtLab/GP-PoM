'''
Subsampling methods for time series. Implements several ways of
enforcing missingness. Each of these will merely update tensors
to include NaN values---data will *not* be removed.
'''

import numpy as np

import warnings


def _mask_tensor(X, p, random_state):
    '''
    Masks values in a tensor with a given probability. The masking will
    set certain values to `np.nan`, making it easy to ignore them for a
    downstream processing task. Given an unravelled tensor of size $m$,
    this function will mask up to $p \cdot m$ of the entries.

    Parameters
    ----------

        X: Input tensor or `np.ndarray`.
        p: Probability for masking any one entry in the tensor.

        random_state: Random number generator for performing the actual
        sampling prior to the masking. This should be specified from an
        external class instance.
    '''

    # Full tensor size, i.e. the product of instances and channels
    # of the time series.
    n = np.prod(X.shape)

    # Get the number of samples that we need to mask in the end;
    # we fully ignore interactions between different channels.
    m = int(np.floor(n * p))

    # Nothing to do here, move along!
    if m == 0:
        warnings.warn(
                f'The current subsampling threshold will *not* result '
                f'in any instances being subsampled. Consider using a '
                f'larger probability than {p}.'
        )
    else:

        # Get the indices that we are masking in the original time
        # series and update `X` accordingly.
        indices = random_state.choice(n, m, replace=False)
        X.ravel()[indices] = np.nan

    return X


class MissingAtRandomSubsampler:
    '''
    Performs MAR (missing at random) subsampling using a pre-defined
    threshold.
    '''

    def __init__(self, probability=0.1, random_seed=2020):
        '''
        Creates a new instance of the sampler object.

        Parameters
        ----------

            probability: Subsampling probability
            random_seed: Random seed to use for the subsampling
        '''

        self.probability = probability
        self.random_seed = random_seed

        assert 0.0 <= self.probability <= 1.0

    def __call__(self, instance, index):
        '''
        Applies the MAR subsampling to a given instance. The input label
        is optional because it will be ignored.

        Parameters
        ----------

            index: Index of the instance, with respect to some outer
            counter. This is required to ensure reproducibility.

            instance: An instance of a data set, as supplied by the
            `UEADataset` class. Requires the existence of `dict` or
            a `dict`-like object, containing the keys `values`, for
            the tensor values, and `label` for the label.

        '''

        instance['values'] = _mask_tensor(
            instance['values'],
            self.probability,
            np.random.RandomState(self.random_seed + index)
        )

        return instance

    def __repr__(self):
        '''
        Returns a string-based representation of the class, which will
        be useful when creating output filenames.
        '''

        r = f'{self.__class__.__name__}_{self.probability:.2f}'
        return r.replace('.', '_')


class LabelBasedSubsampler:
    '''
    Performs subsampling conditional on class labels by using
    a pre-defined set of thresholds for each class.
    '''

    def __init__(self, n_classes, probability_ranges, random_seed=2020):
        '''
        Creates a new instance of the sampler object.

        Parameters
        ----------

            n_classes: The number of classes for the data set. This
            directly influences the number of probabilities for the
            subsampling procedure.

            probability_ranges: Tuple, specifying a lower and upper
            range for the per-class subsample probabilities.

            random_seed: Random seed to use for the subsampling
        '''

        random_state = np.random.RandomState(random_seed)

        prob_l = probability_ranges[0]
        prob_r = probability_ranges[1]

        assert prob_l <= prob_r

        assert 0.0 <= prob_l <= 1.0
        assert 0.0 <= prob_r <= 1.0

        # Generate dropout probabilities for each of the classes. This
        # assumes that labels are forming a contiguous range between 0
        # and `n_classes`.
        self.probabilities = random_state.uniform(
            prob_l, prob_r, n_classes
        )

        self.random_seed = random_seed

    def __call__(self, instance, index):
        '''
        Applies the MAR subsampling to a given instance. The input label
        is optional because it will be ignored.

        Parameters
        ----------

            index: Index of the instance, with respect to some outer
            counter. This is required to ensure reproducibility.

            instance: An instance of a data set, as supplied by the
            `UEADataset` class. Requires the existence of `dict` or
            a `dict`-like object, containing the keys `values`, for
            the tensor values, and `label` for the label.

        '''

        # This call looks idiosyncratic because the UEA reader class
        # wraps labels into an additional dimension.
        label = int(instance['label'][0])

        # Get probability for the particular instance.
        p = self.probabilities[label]

        instance['values'] = _mask_tensor(
            instance['values'],
            p,
            np.random.RandomState(self.random_seed + index)
        )

        return instance

    def __repr__(self):
        '''
        Returns a string-based representation of the class, which will
        be useful when creating output filenames.
        '''

        p = '_'.join(f'{prob:.2f}' for prob in self.probabilities)
        r = f'{self.__class__.__name__}_{p}'

        return r.replace('.', '_')

