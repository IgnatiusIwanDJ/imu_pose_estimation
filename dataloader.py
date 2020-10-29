import glob
import csv
from torch.utils.data import Dataset

class ImuPoseDataset(Dataset):
    """Dataset containing IMU pose"""

    def __init__(self, files, transform=None):
        """
        Args:
            files (list): list containing file names.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.train_data = []
        self.labels = []
        for file in files:
            with open(file) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=',')
                for row in csv_reader:
                    if int(row[21]) != 0:
                        self.train_data.append([float(item) for item in row[0:21]])
                        self.labels.append(int(row[21]))
        self.transform = transform

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

        if self.transform:
            data = self.transform(data)

        return data, label
