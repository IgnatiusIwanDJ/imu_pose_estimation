#!/usr/bin/env python3
import os
import numpy as np
import datetime
import random
import glob
import math
import syft as sy
import torch
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau
from util import check_mkdir, load_ckp, save_ckp, connect_to_workers
from dataloader import ImuPoseDataset
from model import CnnModel
import sklearn.metrics as metrics
from tensorboardX import SummaryWriter

INPUT_SIZE = 1
"""
If you are using null class, change the NUM_CLASSES to 7
"""
NUM_CLASSES = 6
LABEL = ['Standing still', 'Sitting and relaxing', 'Lying down', 'Walking', 'Climbing', 'Running']

def get_dataloader(train_files,valid_files,batch_size,workers,include_null_class,use_cuda = False):

    train_dataset = ImuPoseDataset(files=train_files,include_null = include_null_class)
    valid_dataset = ImuPoseDataset(files=valid_files,include_null = include_null_class)
    if include_null_class:
        global NUM_CLASSES, LABEL
        NUM_CLASSES = 7
        LABEL = ['No Activity','Standing still', 'Sitting and relaxing', 'Lying down', 'Walking', 'Climbing', 'Running']

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    federated_train_loader = sy.FederatedDataLoader( train_dataset
                                                .federate(workers), # <-- we distribute the dataset across all the workers, it's now a FederatedDataset
                                                batch_size=batch_size,
                                                drop_last=True,
                                                shuffle=True,
                                                **kwargs)

    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, drop_last=True,shuffle=True,**kwargs)

    return federated_train_loader,valid_loader

def get_args():
    parser = argparse.ArgumentParser(description='Training Pose Estimation')
    parser.add_argument('-i','--input_folder', type=str,default='data/Preprocessed', help='Training image path')
    parser.add_argument('--validation_size', type=float,default=0.2, help='Validation size')
    parser.add_argument('-w','--num_workers', type=int,default=2, help='Virtual worker for federated learning')
    parser.add_argument('-s','--save_folder', type=str,default='', help='Save model path')
    parser.add_argument('-e','--epoch', type=int, default=30, help='Epoch for training')
    parser.add_argument('-b','--batch_size', type=int, default=30, help='Batch size')
    parser.add_argument('-lr','--learning_rate', type=float, default=0.01, help='Learning rate for training')
    parser.add_argument('--learning_patience', type=int, default=10, help='Loss not decreasing patience before changing learning rate')
    parser.add_argument('-c','--checkpoint', type=str, default='', help='Checkpoint file from last training')
    parser.add_argument('--include_null_class', action="store_true", help='Include Null classes')
    parser.add_argument('-n','--saved_model_name', type=str, default='pose_estimator', help='Name for model result')
    return parser.parse_args()

def train(args, model, criterion, optimizer, train_loader, gpu_found = False, valid_loader = None):
    # track change in validation loss
    valid_loss_min = np.Inf
    start_epoch = 1
    time_today = str(datetime.date.today())
    clock = datetime.datetime.now().strftime('%H-%M-%S')

    # check if CUDA is available
    device = torch.device("cuda" if gpu_found else "cpu")
    model = model.to(device)
    # move tensors to GPU if CUDA is available
    if gpu_found:
        device_used = "Train on GPU"
    else:
        device_used = "Train on CPU"

    # parse args for training
    max_epoch = args['epoch']
    batch_size = args['batch_size']
    learning_rate = args['learning_rate']
    learning_patience = args['learning_patience']
    checkpoint = args['checkpoint']
    model_save = args['saved_model_name']
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=learning_patience, min_lr=1e-10, verbose=True)

    # parse args utility
    save_folder = args['save_folder']
    # if provided checkpoint is valid
    if os.path.isfile(checkpoint):
        print('found checkpoint using checkpoint')
        ckpth_path = checkpoint
        model, optimizer, start_epoch, max_epoch_save, valid_loss_min = load_ckp(ckpth_path, model, optimizer)
        max_epoch = max_epoch_save
    else:
        check_mkdir( os.path.join('log/checkpoint', time_today))
        ckpth_path = os.path.join('log/checkpoint', time_today, clock)
        check_mkdir(ckpth_path)
        ckpth_path = os.path.join('log/checkpoint', time_today, clock,'checkpoint.pt')
    # folder to save model
    if save_folder != '' and os.path.exists(save_folder):
        model_path = save_folder
        ckpth_path = os.path.join(save_folder,'checkpoint.pt')
    else:
        check_mkdir(os.path.join('model/trained_model', time_today))
        model_path = os.path.join('model/trained_model', time_today, clock)
    check_mkdir(model_path)

    # start logging tensorboard
    check_mkdir(os.path.join('log/train_log', time_today))
    log_path = os.path.join('log/train_log', time_today, clock)
    check_mkdir(log_path)
    writer = SummaryWriter(log_path, comment = "{}_Start at{}_LR_{}_BATCH_{}".format(time_today,clock,learning_rate,batch_size))

    # draw model graph
    data, label = next(iter(train_loader))
    writer.add_graph(model, data.get().to(device))

    # start training
    print('--------------------------------------------------------------------------------------------')
    print('Train params: Epochs:{}, Batch Size: {}, Learning Rate: {}'.format(max_epoch,batch_size,learning_rate))
    print('\t\tCheckpoint saved to: {}'.format(ckpth_path))
    print('\t\tSaving Model to: {}/{}.pt'.format(model_path,model_save))
    print('\t\tStart at Epoch: {}'.format(start_epoch))
    print(device_used)
    print('--------------------------------------------------------------------------------------------')

    for epoch in range(start_epoch, max_epoch+1):

        # keep track of validation loss
        valid_loss = []
        train_loss = 0.0

        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # send the model to the right location
            model.send(data.location)
            # move tensors to GPU if CUDA is available
            data, target = data.to(device), target.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # get the model back
            model.get()
            # get the loss back
            loss = loss.get()
            train_loss += loss.item()

            if batch_idx % 500 == 499:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * batch_size, len(train_loader) * batch_size,
                100. * batch_idx / len(train_loader), train_loss / batch_idx ))

        ######################
        # validate the model #
        ######################
        all_predictions = np.array([])
        all_target = np.array([])

        model.eval()
        for data, target in valid_loader:
            # move tensors to GPU if CUDA is available
            data, target = data.to(device), target.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # update average validation loss
            valid_loss.append(loss.item())
            # calculate metrics
            top_p, top_class = output.topk(1, dim=1)
            eval_pred = top_class.view(-1).long().cpu().numpy()
            eval_target = target.long().cpu().numpy()

            all_predictions = np.append(all_predictions, eval_pred)
            all_target =  np.append(all_target, eval_target)

        # calculate average losses
        valid_loss = np.mean(valid_loss)
        train_loss = train_loss / len(train_loader)
        # calculate metrics
        f1_total = metrics.f1_score(all_target, all_predictions, average='weighted')
        acc_total = metrics.accuracy_score(all_target, all_predictions)
        precision_total = metrics.precision_score(all_target, all_predictions, average='weighted')
        recall_total = metrics.recall_score(all_target, all_predictions, average='weighted')

        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("valid_loss", valid_loss, epoch)
        writer.add_scalar("Accuracy", acc_total, epoch)
        writer.add_scalar("F1 Score", f1_total, epoch)
        writer.add_scalar("Recall", recall_total, epoch)
        writer.add_scalar("Precision", precision_total, epoch)

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))
        # update scheduler
        scheduler.step(valid_loss)

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            print('F1-score: {:.6f}\t Accuracy:{:.6f}\t Precission:{:.6f}\t Recall:{:.6f}'.format(
            f1_total, acc_total, precision_total, recall_total))
            writer.add_text('model_epoch_{}_train_loss_{}_val_loss_{}'.format(epoch,train_loss,valid_loss),
            'F1: {:.4f} Acc: {:.4f} Prec: {:.4f} Rec: {:.4f}'.format(f1_total, acc_total, precision_total, recall_total))
            torch.save(model.state_dict(), model_path+'/{}.pt'.format(model_save))
            valid_loss_min = valid_loss

        # checkpoint params
        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'max_epoch': max_epoch,
            'best_valid_loss': valid_loss_min,
        }
        # save checkpoint
        save_ckp(checkpoint, ckpth_path)

    writer.close()
    print('training end....')


if __name__ == '__main__':
    hook = sy.TorchHook(torch)
    args = get_args()
    # look for gpu
    gpu_found = torch.cuda.is_available()
    # data
    workers = connect_to_workers(hook,n_workers = args.num_workers)
    files = glob.glob(args.input_folder+"/*.csv")
    count_valid = math.ceil(args.validation_size*len(files))
    random.shuffle(files)
    valid_files,train_files = files[0:count_valid],files[count_valid:]
    train_loader, valid_loader = get_dataloader(train_files,valid_files,args.batch_size,workers,args.include_null_class,gpu_found)
    # args, model, workers
    model = CnnModel(INPUT_SIZE,NUM_CLASSES)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr = args.learning_rate)

    train(vars(args),model, criterion, optimizer,train_loader,gpu_found,valid_loader)
