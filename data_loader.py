import logging
import os
from typing import List

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Compose

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.addHandler(logging.StreamHandler())

load_dotenv('envs/.env')

CMAPSS_DATA_DIRECTORY = os.getenv("CMAPSS_DATA_DIRECTORY")

unit_settings = ['Unit Number', 'Cycles']
operational_settings = ['Altitude', 'TRA', 'Mach']
sensor_names = ['Total temperature at fan inlet, T2 (◦R)',
                'Total temperature at LPC outlet, T24 (◦R)',
                'Total temperature at HPC outlet, T30 (◦R)',
                'Total temperature at LPT outlet, T50 (◦R)',
                'Pressure at fan inlet, P2 (psia)',
                'Total pressure in bypass-duct, P15 (psia)',
                'Total pressure at HPC outlet, P30 (psia)',
                'Physical fan speed, Nf (rpm)',
                'Physical core speed, Nc (rpm)',
                'Engine pressure ratio (P50/P2)',
                'Static pressure at HPC outlet, Ps30 (psia)',
                'Ratio of fuel flow to Ps30, phi (pps/psi)',
                'Corrected fan speed rpm',
                'Corrected core speed rpm',
                'Bypass Ratio',
                'Burner fuel-air ratio',
                'Bleed Enthalpy',
                'Demanded fan speed rpm',
                'Demanded corrected fan speed rpm',
                'HPT coolant bleed, W31 (lbm/s)',
                'LPT coolant bleed (lbm/s)']


def load_datasets(simulation_type: str, validation_size=None):
    train_path = f"{CMAPSS_DATA_DIRECTORY}/train_{simulation_type}.txt"
    test_x_path = f"{CMAPSS_DATA_DIRECTORY}/test_{simulation_type}.txt"
    test_y_path = f"{CMAPSS_DATA_DIRECTORY}/RUL_{simulation_type}.txt"

    feature_names = unit_settings + operational_settings + sensor_names
    df_train = pd.read_csv(train_path, sep='\s+', header=None, names=feature_names)
    df_train['RUL'] = \
        pd.merge(df_train[['Unit Number']], df_train.groupby('Unit Number')['Cycles'].max(), on='Unit Number',
                 how='inner')[
            'Cycles'] - df_train['Cycles']

    df_test_x = pd.read_csv(test_x_path, sep='\s+', header=None, names=feature_names)
    df_test_y = pd.read_csv(test_y_path, header=None).reset_index().rename({'index': 'Unit Number', 0: 'RUL'}, axis=1)
    df_test = pd.merge(df_test_x, df_test_y, how="inner", on="Unit Number")

    if validation_size:
        df_train, df_val = train_test_split(df_train, test_size=validation_size)
        return df_train, df_val, df_test

    return df_train, df_test

def generate_classification_dataset(df_, rul_cutoff_):
    df_x_ = df_[sensor_names]
    df_y_ = df_['is_faulty'] = np.where(df_['RUL'] < rul_cutoff_, 1, 0)
    return df_x_, df_y_

def generate_regression_dataset(df_):
    return df_[sensor_names], df_['RUL']

def load_dataset(simulation_type: str, train=True):
    feature_names = unit_settings + operational_settings + sensor_names

    if train:
        train_path = f"{CMAPSS_DATA_DIRECTORY}/train_{simulation_type}.txt"
        df_train = pd.read_csv(train_path, sep='\s+', header=None, names=feature_names)
        df_train['RUL'] = \
            pd.merge(df_train[['Unit Number']], df_train.groupby('Unit Number')['Cycles'].max(), on='Unit Number',
                     how='inner')['Cycles'] - df_train['Cycles']
        return df_train
    else:
        test_x_path = f"{CMAPSS_DATA_DIRECTORY}/test_{simulation_type}.txt"
        test_y_path = f"{CMAPSS_DATA_DIRECTORY}/RUL_{simulation_type}.txt"
        df_test_x = pd.read_csv(test_x_path, sep='\s+', header=None, names=feature_names)
        df_test_y = pd.read_csv(test_y_path, header=None).reset_index().rename({'index': 'Unit Number', 0: 'RUL'},
                                                                               axis=1)
        df_test = pd.merge(df_test_x, df_test_y, how="inner", on="Unit Number")
        return df_test


def get_transform(train):
    transforms = [ToTensor()]
    # if train:
    #     transforms.append(...other transforms...)
    return Compose(transforms)


class TurbofanDataset(Dataset):
    """
    Dataset class for loading preprocessed training & testing data
    Specify a simulation_type, e.g. FD001 and a set of engine IDs. These IDs are from the clustering step
    At each iteration, the dataset returns the sensor data + per-engine
    """

    def __init__(self, simulation_type: str, unit_numbers: List[int], train: bool, scaler=None, transforms=None,
                 as_classification=False, rul_cutoff_cycles=50):
        """
        Load data into object
        :param simulation_type:
        :param unit_numbers:
        :param train:
        :param transforms:
        :param as_classification: If true, use the RUL threshold to split samples into healthy vs. unhealthy
        """
        self.simulation_type = simulation_type
        self.transforms = transforms
        self.feature_cols = operational_settings + sensor_names
        self.classification = as_classification

        data = load_dataset(simulation_type, train)
        data = data[data["Unit Number"].isin(unit_numbers)]

        assert len(np.unique(data["Unit Number"])) == len(unit_numbers)

        if as_classification:
            # Each unit starts degrading at an arbitrary time, so a single threshold won't work
            # Take the max RUL and set the last 50% of cycle data as UNHEALTHY, then shuffle
            # TODO: Figure out a better way to calculate failure state
            # rul_cutoff = pd.merge(data[["Unit Number"]], data.groupby(
            #     "Unit Number").max() * rul_threshold, how="inner", on="Unit Number")["Cycles"]
            self.target_col = "is_faulty"

            data[self.target_col] = 0
            # If an engine's RUL < RUL_cutoff, then consider it an imminent failure
            data.loc[data["RUL"].values < rul_cutoff_cycles, self.target_col] = 1
        else:
            self.target_col = "RUL"

        if train:
            min_max_scaler = MinMaxScaler()  # feature_range=(-1, 1))
            data[self.feature_cols] = min_max_scaler.fit_transform(data[self.feature_cols])
            self.scaler = min_max_scaler
        else:
            if scaler is not None:
                data[self.feature_cols] = scaler.transform(data[self.feature_cols])

        self.engine_data = data[self.feature_cols+[self.target_col]]

    def get_stats(self):
        target_vals = np.unique(self.engine_data[self.target_col], return_counts=True)
        stats = f"Target: {target_vals}"
        return stats

    def __getitem__(self, idx):
        datum = self.engine_data.iloc[idx]
        sample = Tensor(np.array(datum[self.feature_cols].values))[None, ...]

        if self.classification:
            target = torch.tensor([[datum[self.target_col]]])
        else:
            target = Tensor(np.array([datum[self.target_col]]))

        if self.transforms is not None:
            sample = self.transforms(sample)
            target = self.transforms(target)

        return sample, target

    def __len__(self):
        return len(self.engine_data)
