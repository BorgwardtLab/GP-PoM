#Imports for UEA dataset class:
import pickle
import os
import pandas as pd
import numpy as np
from tqdm import trange

from .dataset import Dataset
from .utils import DATA_DIR
import uea_ucr_datasets # >>>>> requires the input data in ~/.data/UEA_UCR <<<<<<

from src.tasks import MulticlassClassification 
from .benchmarks_utils import Normalizer
from .utils import DATA_DIR

DATASET_BASE_PATH = os.path.join(DATA_DIR, 'UEA')

class UEADataReader():
    """UEA Data Reader to read and return instances of given split."""
    def __init__(self, dataset_name, split, out_path):
        uea_ucr_datasets.list_datasets()
        if split == 'testing':
            data = uea_ucr_datasets.Dataset(dataset_name, train=False)
            self.X, self.y = _to_array(data)
        elif split in ['training', 'validation']:
            validation_split_file = os.path.join(out_path, 'validation_split_file.pkl')
            data = uea_ucr_datasets.Dataset(dataset_name, train=True)
            X, y = _to_array(data)
            if not os.path.isfile(validation_split_file):
                print('Generating stratified training/validation split...')
                #now create the splits:
                from sklearn.model_selection import StratifiedShuffleSplit
                sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                #X_dummy = np.zeros([len(y),2])
                sss.get_n_splits(X, y) #for simpler NaN handling, we use dummy data for splitting,
                #as only labels are relevant

                training_indices, validation_indices = next(sss.split(X, y))
                split_dict = {'training': training_indices,
                              'validation': validation_indices
                }
                #save the split ids
                if not os.path.exists(out_path):
                    os.makedirs(out_path, exist_ok=True)
                with open(validation_split_file, 'wb') as f:
                    pickle.dump(split_dict, f )#protocol=pickle.HIGHEST_PROTOCOL)
            else:
                print('Loading stratified training/validation split.')
                with open(validation_split_file, 'rb') as f:
                    split_dict = pickle.load(f)
            indices = split_dict[split]
            self.X = X[indices] #subsetting the split
            self.y =  y[indices]
        else:
            raise ValueError('Provided split not available.',
                             'Use any of [training, validation, testing]'
                            )
    def read_example(self, index):
        return {'X': self.X[index], 'y': self.y[index]}
    def get_number_of_examples(self):
        return len(self.X)


class UEADataset(Dataset):
    """UEA Dataset Class to load any UEA dataset."""

    #Here, the normalizer config path is still abstract, to be formatted later:
    normalizer_config = os.path.join(
         os.path.dirname(__file__), #hard code this line with cwd when working interactively
        'resources',
        '{}Dataset_normalization.json'
    )

    def __init__(self,
        dataset_name,
        split,
        transform=None,
        data_path=DATASET_BASE_PATH,
        use_disk_cache=False,
        data_format='GP'):
        """Initialize dataset.

        Args:
            -dataset_name: Name of UEA dataset to load.
                [ PenDigits, .. ]
            -split: Name of split. One of `training`, `validation`, `testing`.
            -data_path: Path to data. Default:
                {project_root_dir}/data/UEA
            -use_disk_cache: writes imputations to disk for faster training
            -data_format: this is a dummy argument to conform with other dataset classes! 
                For the UEA datasets which involve more transforms (subsampling, imputation), 
                those transforms are handled outside of the dataset class in the input transforms
        """
        self.data_path = os.path.join(data_path, dataset_name)
        self.split = split

        #self.dataset = uea_ucr_datasets.Dataset(dataset_name, train=True)
        self.reader = UEADataReader(dataset_name, split, self.data_path)
        self.normalizer_config = self.normalizer_config.format(dataset_name)
        self.normalizer = Normalizer()
        self._set_properties()

        if not os.path.exists(self.normalizer_config):
            print(f'Normalizer config {self.normalizer_config} not found!')
            print('Generating normalizer config...')
            if split != 'training':
                # Only allow to compute normalization statics on training split
                raise ValueError(
                    'Not allowed to compute normalization data '
                    'on other splits than training.'
                )
            for i in trange(len(self)):
                instance = self.reader.read_example(i)['X']
                self.normalizer._feed_data(instance)
            self.normalizer._save_params(self.normalizer_config)
        else:
            self.normalizer.load_params(self.normalizer_config)

        self.use_disk_cache = use_disk_cache
        self.transforms_not_to_cache =  ['gpytorch', 'no_transform'] # dataset independent transforms
        # Sets a default transformation that only returns the first
        # argument. This ensures that later on, if no transform has
        # been set, only the *instance* is returned.
        self.maybe_transform = transform if transform else lambda a: a
        # if using disk_cache: split the transforms into those that are cached and those that are not
        # this step is useful to ensure that all methods (GP or not) use the exact same subsampling        
        if use_disk_cache:
            self.maybe_transform_to_cache = [t for t in self.maybe_transform if all(
                [word not in str(t) for word in self.transforms_not_to_cache]) ]        
            self.maybe_transform_not_to_cache = [t for t in self.maybe_transform if any(
                [word in str(t) for word in self.transforms_not_to_cache]) ]        

    def _set_properties(self):
        self.has_unaligned_measurements = False
        self.statics = None
        self.n_statics = 0
        instance = self.reader.read_example(0)
        self._measurement_dims = instance['X'].shape[1]

    @property
    def n_classes(self):
        """
        N of different classes, determining classifier output dimension (if binary, use 1) otherwise n_classes
        """
        return self.n_class_types() 

    def n_class_types(self):
        """
        Given single-label settings, how many classes / class manifestations of this label
        """
        distinct_labels = np.unique(self.reader.y)
        return len(distinct_labels)

    @property
    def measurement_dims(self):
        return self._measurement_dims

    def __len__(self):
        return self.reader.get_number_of_examples()

    def __getitem__(self, index):

        # Check for the existence of data files or create them if they
        # do not exist.
        if self.use_disk_cache:

            # Create proper mode string depending on the transformations
            # that are set. We use this to check the disk cache, and skip
            # transforms that were requested not to be cached
            mode = '__'.join(repr(t) for t in self.maybe_transform_to_cache)

            path = os.path.join(
                self.data_path,
                self.split,
                mode
            )

            os.makedirs(path, exist_ok=True)

            cached_file = os.path.join(path, str(index) + '.pkl')

            if os.path.exists(cached_file):
                with open(cached_file, 'rb') as f:
                    instance = pickle.load(f)
                
                #if available, apply remaining transforms which were not cached:
                if len(self.maybe_transform_not_to_cache) > 0:
                    for transform in self.maybe_transform_not_to_cache:
                        instance = transform(instance)
                return instance

            # File does not exist; perform all transformations that are required to be cached
            # and write it out later.
            else:
                instance = self._read_and_process_instance(index, self.use_disk_cache)

                with open(cached_file, 'wb') as f:
                    pickle.dump(instance, f)
                
                #if available, apply remaining transforms which were not cached:
                if len(self.maybe_transform_not_to_cache) > 0:
                    for transform in self.maybe_transform_not_to_cache:
                        instance = transform(instance)

                return instance

        # Just load the existing file. When not using disk_cache, all transforms are performed 
        # inside _read_and_process_instance
        else:
            return self._read_and_process_instance(index, self.use_disk_cache)

    def _read_and_process_instance(self, index, use_disk_cache=False):
        '''
        Internal reading and processing function. Loads a single example
        from a raw dataset and applies all [optional] transformations.
        
        Parameters
        ----------

            - index: Index of the instance to load. Should be supplied by
            the caller, e.g. by `__getitem__`.
            - use_disk_cache: if True, only the transforms which ought to be cached are applied
                              while the remaining transforms are applied in __getitem__ after loading
                              if False, all transforms in maybe_transform are applied here
        '''

        instance = self.reader.read_example(index)
        features = self.normalizer.transform(instance['X'])
        label = instance['y']
        # Add an additional dimension to the label if it is only a scalar.
        # This makes it confirm more to the treatment of multi-class
        # targets.
        label = [label]
        label = np.array(label, dtype=np.float32)
        time = np.array(np.arange(features.shape[0]), dtype=np.float32)[:, None]
        features = np.array(features, dtype=np.float32)

        # Instance that will be returned to the user or transformed
        # depending on the availability of transformations.
        instance = {
            'time': time,
            'values': features,
            'label': label
        }

        # Check if multiple transformations are present and apply them
        # in the correct order.
        if type(self.maybe_transform) is list:

            # Note that the index is supplied multiple times because we
            # do not make any assumptions about the underlying functor.
            for transform in self.maybe_transform:
                if any(word in str(transform) for word in self.transforms_not_to_cache):
                    if not use_disk_cache:
                        instance = transform(instance)
                    else:
                        #apply these transforms only in __getitem__ to circumvent caching of them
                        pass
                else:
                    instance = transform(instance, index)
        
        else: #in case maybe_transform consist of only 1 transform
            if any(word in str(self.maybe_transform) for word in self.transforms_not_to_cache):
                if not use_disk_cache:
                    instance = self.maybe_transform(instance)
                else:
                    #apply these transforms only in __getitem__ to circumvent caching of them
                    pass
            else:
                instance = self.maybe_transform(instance, index)

        return instance
    
    @property
    def task(self):
        return MulticlassClassification(self.n_classes)

def _to_array(data):
        """
        Util function to convert iterable dataset to X,y arrays for stratified splitting, only used once for defining splits, so no need for speed
        """
        X = []; y = []
        for instance_x, instance_y in data:
            X.append(instance_x)
            y.append(instance_y)
        X = np.array(X)
        y = np.array(y)
        return X, y
