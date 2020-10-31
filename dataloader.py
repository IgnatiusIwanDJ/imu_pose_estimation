#!/usr/bin/env python3
import glob
import csv
from torch.utils.data import Dataset
import json
import torch

class ImuPoseDataset(Dataset):
    """Dataset containing IMU pose"""

    def __init__(self, files, transform=None, include_null=False, return_old_data=False):
        """
        Args:
            files (list): list containing file names.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            include_null (bool): Optional, if want to load data will null class
        """
        self.train_data = []
        self.labels = []
        for file in files:
            with open(file) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    if include_null:
                        self.train_data.append([float(item) for item in row[0:21]])
                        self.labels.append(int(row[21]))
                    elif int(row[21]) != 0:
                        self.train_data.append([float(item) for item in row[0:21]])
                        self.labels.append(int(row[21])-1)

        self.transform = transform
        self.return_old_data = return_old_data

    def __len__(self):
        return len(self.train_data)

    def print_value(self,idx):
        print('Chest \t\t\tLeft Angkle')
        print('{:.2f} \t{:.2f} \t{:.2f} \t{:.2f} \t{:.2f} \t{:.2f} \t{:.2f} \t{:.2f} \t{:.2f} \t{:.2f} \t{:.2f} \t{:.2f}'.format(*self.train_data[idx]))
        print('Label \t\t\tRight Angkle')
        print('{} \t\t\t{:.2f} \t{:.2f} \t{:.2f} \t{:.2f} \t{:.2f} \t{:.2f} \t{:.2f} \t{:.2f} \t{:.2f}'.format(self.labels[idx],*self.train_data[idx][12:]))
        print('')

    def __getitem__(self, idx):
        data = self.train_data[idx]
        label = self.labels[idx]

        # convert to tensor
        data = torch.tensor(data)
        label = torch.tensor(label)

        # normalize value
        origin_data = torch.unsqueeze(data, 0)

        data_std = origin_data.std(dim=1, keepdim=True)
        data_mean = origin_data.mean(dim=1, keepdim=True)
        data = (origin_data - data_mean) / data_std

        if self.transform:
            data = self.transform(data)
        if self.return_old_data:
            return data, label, origin_data
        else:
            return data, label
