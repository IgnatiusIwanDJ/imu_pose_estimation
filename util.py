#!/usr/bin/env python3
import os
import torch
import syft as sy

def connect_to_workers(hook,n_workers):
    return [
        sy.VirtualWorker(hook, id=f"worker{i+1}")
        for i in range(n_workers)
    ]

def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch'], checkpoint['max_epoch'], checkpoint['best_valid_loss']

def save_ckp(state, checkpoint_dir):
    torch.save(state, checkpoint_dir)

def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
