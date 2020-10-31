#!/usr/bin/env python3
import os
import torch
import glob
import argparse
import datetime
import sklearn.metrics as metrics
from dataloader import ImuPoseDataset
from model import CnnModel

INPUT_SIZE = 1
"""
If you are using null class, change the NUM_CLASSES to 7
"""
NUM_CLASSES = 6
LABEL = ['Standing still', 'Sitting and relaxing', 'Lying down', 'Walking', 'Climbing', 'Running']

def get_args():
    parser = argparse.ArgumentParser(description='Testing trained Pose Estimator')
    parser.add_argument('-f','--test_folder', type=str,default='test', help='Test dataset location')
    parser.add_argument('-s','--save_result', type=str,default='', help='file name to save json NOTE: input location/filename with .json extension')
    parser.add_argument('-m','--model_path', type=str, default='model/pose_estimator.pt', help='Model file location to load')
    parser.add_argument('-b','--batch_size', type=int, default=1, help='batch size for test')
    parser.add_argument('--include_null_class', action="store_true", help='Include Null classes')
    return parser.parse_args()

def test(test_loader, model, use_gpu, args):
    # parse args
    accuracy = 0.0
    f1score = 0.0
    recall = 0.0
    precision = 0.0

    model_path = args['model_path']
    save_result = args['save_result']
    batch_size = args['batch_size']

    time_today = str(datetime.date.today())
    clock = datetime.datetime.now().strftime('%H-%M-%S')

    # check if CUDA is available
    device = torch.device("cuda" if use_gpu else "cpu")
    model = model.to(device)

    if save_result == '':
        filename = 'result_{}.json'.format(os.path.splitext(os.path.basename(model_path))[0])
    else:
        filename = '{}.json'.format(save_result)

    model.eval()
    json_data = {}

    print('Evaluating model, please wait...')
    # iterate over test data
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # move tensors to GPU if CUDA is available
            data, target = data.to(device), target.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            top_p, top_class = output.topk(1, dim=1)
            eval_pred = top_class.view(-1).long().cpu().numpy()
            eval_target = target.long().cpu().numpy()
            #equals = top_class == target.view(*top_class.shape).long()
            accuracy += metrics.accuracy_score(eval_target, eval_pred)
            f1score += metrics.f1_score(eval_target, eval_pred, average='weighted')
            recall += metrics.recall_score(eval_target, eval_pred, average='weighted')
            precision += metrics.precision_score(eval_target, eval_pred, average='weighted')

            if batch_idx % 500 == 499:
                print('Evaluating [{}/{} ({:.0f}%)]'.format(
                batch_idx * batch_size, len(test_loader) * batch_size,
                100. * batch_idx / len(test_loader)))
            # iterate over batch

        # calculate metrics
        f1_total = f1score/(len(test_loader))
        acc_total = accuracy/(len(test_loader))
        precision_total = precision/(len(test_loader))
        recall_total = recall/(len(test_loader))

        print("--------------------------------------------")
        print("Model {} Performance:".format(model_path))
        print("--------------------------------------------")
        print("F1 Score: {}".format(f1_total))
        print("Accuracy: {}".format(acc_total))
        print("Precission: {}".format(precision_total))
        print("Recall: {}".format(recall_total))
        print("--------------------------------------------")


if __name__ == '__main__':
    args = get_args()
    # look for gpu
    gpu_found = torch.cuda.is_available()
    # data
    files = glob.glob(args.test_folder+"/*.csv")
    test_dataset = ImuPoseDataset(files = files, include_null = args.include_null_class)
    if args.include_null_class:
        NUM_CLASSES = 7
        LABEL = ['No Activity','Standing still', 'Sitting and relaxing', 'Lying down', 'Walking', 'Climbing', 'Running']

    kwargs = {'num_workers': 1, 'pin_memory': True} if gpu_found else {}
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = args.batch_size, shuffle=True, **kwargs)

    # args, model, workers args.include_null_class
    model = CnnModel(INPUT_SIZE,NUM_CLASSES)
    model.load_state_dict(torch.load(args.model_path))

    test(test_loader, model, gpu_found, vars(args))
