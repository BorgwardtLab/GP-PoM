# noqa: ignore: D101,D102
import os

import pandas as pd
import pickle
import numpy as np
from tqdm import trange

from src.imputation import (zero_imputation,
                            linear_imputation,
                            forward_fill_imputation,
                            causal_imputation,
                            indicator_imputation)
import torch

from src.tasks import BinaryClassification
from .dataset import Dataset
from .benchmarks_utils import Normalizer
from .utils import DATA_DIR
import os

DATASET_BASE_PATH = os.path.join(DATA_DIR, 'physionet_2012')


class PhysionetDataReader():
    valid_columns = [
        # Statics
        'Age', 'Gender', 'Height', 'ICUType', 'Weight',
        # Time series variables
        'ALP', 'ALT', 'AST', 'Albumin', 'BUN', 'Bilirubin',
        'Cholesterol', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose',
        'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'MAP', 'MechVent', 'Mg',
        'NIDiasABP', 'NIMAP', 'NISysABP', 'Na', 'PaCO2', 'PaO2', 'Platelets',
        'RespRate', 'SaO2', 'SysABP', 'Temp', 'TroponinI', 'TroponinT',
        'Urine', 'WBC', 'pH'
    ]

    # Need to drop these samples as they contain no time series information and
    # only statics (which we dont support).
    blacklisted_records = [
        140501, 150649, 140936, 143656, 141264, 145611, 142998, 147514, 142731,
        150309, 155655, 156254,

        #outlier sample with much more observations:
        135365
    ]

    def __init__(self, data_path, endpoint_file):
        self.data_path = data_path
        self.split_file_name = os.path.split(endpoint_file)[1]
        endpoint_data = pd.read_csv(endpoint_file, header=0, sep=',')
        # Drop backlisted records
        self.endpoint_data = endpoint_data[
            ~endpoint_data['RecordID'].isin(self.blacklisted_records)]
        
        self.feature_transform = PhysionetFeatureTransform()

    def convert_string_to_decimal_time(self, values):
        return values.str.split(':').apply(
            lambda a: float(a[0]) + float(a[1])/60
        )

    def read_example(self, index, mode='GP', normalizer=None, overwrite=False):
        """
        mode: [GP, zero, linear, forwardfill, causal, indicator ] whereas default (GP) refers to raw mode without imputation 
                new format: returning [times, features, label] 
        """
        def read_raw_example(index):
            example_row = self.endpoint_data.iloc[index, :]
            record_id = example_row['RecordID']
            data = self.read_file(str(record_id))
            data['Time'] = self.convert_string_to_decimal_time(data['Time'])
            #return {'X': data, 'y': example_row['In-hospital_death']}
            label = example_row['In-hospital_death'] 
            time, features = self.feature_transform(data)            
            return time, features, label
        
        if mode == 'GP': #GP refers to raw mode where nothing is changed
            if normalizer is not None:
                time, features, label = read_raw_example(index)
                return time, normalizer.transform(features), label
            else:
                return read_raw_example(index)            
        elif mode in ['zero', 'linear', 'forwardfill', 'causal', 'indicator']:
            #check for imputed data path:
            if 'val_' in self.split_file_name: #doing validation 
                outpath = os.path.join( os.path.split(self.data_path)[0], 'validation') 
            else:
                outpath = self.data_path
            imputed_path = os.path.join(outpath, mode + '_imputations')
            imputed_file = os.path.join(imputed_path, str(index) + '.pkl')
            if os.path.exists(imputed_file) and not overwrite:
                # read imputed data
                with open(imputed_file, 'rb') as f:
                    data_dict = pickle.load(f)
            else:
                # create imputed data and save it
                imputation_dict = {
                    'zero':         zero_imputation,
                    'linear':       linear_imputation,
                    'forwardfill':  forward_fill_imputation, 
                    'causal':       causal_imputation, 
                    'indicator':    indicator_imputation 
                }
                imputation_fn = imputation_dict[mode]
                time, features, label = read_raw_example(index)
                if normalizer is not None:
                    features = normalizer.transform(features)
                data_dict = {'time': time, 'values': features, 'label': label} 
                # convert to torch tensor for imputation fn compatibility:
                tensor_dict = {key: torch.tensor(value).unsqueeze(0) for key,value in data_dict.items()}
                # ensure time has proper format as in batch:
                tensor_dict['time'] = tensor_dict['time'].unsqueeze(-1)

                # reformat for imputation (batch dict of tensor)
                # do imputation
                imputed_dict = zero_imputation( 
                        imputation_fn(tensor_dict) 
                ) 
                #reformat to numpy:
                data_dict = {key: value.squeeze(0).numpy() for key,value in imputed_dict.items()}
                data_dict['time'] = data_dict['time'].squeeze(-1) #to stay consistent with other formats
                
                # save imputation
                if not os.path.exists(imputed_path):
                    os.makedirs(imputed_path, exist_ok=True)
                with open(imputed_file, 'wb') as f:
                    pickle.dump(data_dict, f) 
            return data_dict['time'], data_dict['values'], data_dict['label'] 
        else:
            raise ValueError('mode not among available ones: [GP, zero, linear, forwardfill, causal, indicator]')


    def read_file(self, record_id):
        filename = os.path.join(self.data_path, record_id + '.txt')
        df = pd.read_csv(filename, sep=',', header=0)
        # Sometimes the same value is observered twice fot the same time, in
        # this case simply take the first occurance.
        duplicated_entries = df[['Time', 'Parameter']].duplicated()
        df = df[~duplicated_entries]
        pivoted = df.pivot(index='Time', columns='Parameter', values='Value')
        return pivoted.reindex(columns=self.valid_columns).reset_index()

    def get_number_of_examples(self):
        return len(self.endpoint_data)


class PhysionetFeatureTransform():
    ignore_columns = ['Time', 'Age', 'Gender', 'Height', 'ICUType', 'Weight']

    def __call__(self, dataframe):
        times = dataframe['Time'].values
        values = dataframe[[
            col for col in dataframe.columns if col not in self.ignore_columns]].values
        return times, values


class Physionet2012Dataset(Dataset):
    """ Dataset of the PhysioNet 2012 Computing in Cardiology challenge.
        As this dataset is irregularly spaced, we assume that only one input transform is applied 
        (to_gpytorch_format or no_transform) as subsampling etc is not needed.
    """

    normalizer_config = os.path.join(
        os.path.dirname(__file__),
        'resources',
        'Physionet2012Dataset_normalization.json'
    )

    def __init__(self, split, data_format, transform=None, data_path=DATASET_BASE_PATH, overwrite=False):
        """Initialize dataset.

        Args:
            split: Name of split. One of `training`, `validation`, `testing`.
            data_format: which format to return, [GP, zero, linear, forwardfill, causal, indicator]
            data_path: Path to data. Default:
                {project_root_dir}/data/physionet_2012
            overwrite: recomputing imputation dumps (useful for debugging)

        """
        self.data_format = data_format
        self.data_path = data_path
        split_dir, split_file = self._get_split_path(split)
        self.overwrite = overwrite
        self.reader = PhysionetDataReader(split_dir, split_file)
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
                time, features, label = self.reader.read_example(i) #read raw data wihout imputation!
                self.normalizer._feed_data(features)
            self.normalizer._save_params(self.normalizer_config)
        else:
            self.normalizer.load_params(self.normalizer_config)

        self.maybe_transform = transform if transform else lambda a: a

    def _get_split_path(self, split):
        split_paths = {
            'training': (
                os.path.join(self.data_path, 'train'),
                os.path.join(self.data_path, 'train_listfile.csv')
            ),
            'validation': (
                os.path.join(self.data_path, 'train'),
                os.path.join(self.data_path, 'val_listfile.csv')
            ),
            'testing': (
                os.path.join(self.data_path, 'test'),
                os.path.join(self.data_path, 'test_listfile.csv')
            )
        }
        return split_paths[split]

    def _set_properties(self):
        self.has_unaligned_measurements = True
        self.statics = None
        self.n_statics = 0
        times, features, label = self.reader.read_example(0)
        self._measurement_dims = features.shape[1]

    @property
    def n_classes(self):
        return 1

    @property
    def measurement_dims(self):
        return self._measurement_dims

    def __len__(self):
        return self.reader.get_number_of_examples()

    def __getitem__(self, index):
        t, features, label = self.reader.read_example(index, mode=self.data_format, normalizer=self.normalizer, overwrite=self.overwrite)
 
        #t, features = self.feature_transform(instance['X'])
        #features = self.normalizer.transform(features)
        if self.n_classes == 1:
            # Add an additional dimension to the label if it is only a scalar.
            # This makes it confirm more to the treatment of multi-class
            # targets.
            label = [label]
        label = np.array(label, dtype=np.float32)
        time = np.array(t, dtype=np.float32)[:, None]
        features = np.array(features, dtype=np.float32)
        return self.maybe_transform(
            {'time': time, 'values': features, 'label': label})

    @property
    def task(self):
        return BinaryClassification()
