#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F

class CnnModel(nn.Module):
    """CNN model using 1D convolution"""
    def __init__(self, input_size, num_classes):
        super(CnnModel,self).__init__()

        # first layer
        self.conv1 = nn.Conv1d(input_size, 64, 5)
        self.conv2 = nn.Conv1d(64, 128, 5)
        self.conv3 = nn.Conv1d(128, 128, 5)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        # second layer
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(128 * 9, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128 * 9)
        x = self.dropout1(x)
        x = self.dropout2(F.relu(self.fc1(x)))
        out = self.fc2(x)

        return out
