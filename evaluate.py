#!/usr/bin/env python3
import os
import torch
import numpy as np
import glob
import json
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

def test(test_loader, args, use_gpu=False):
    # parse args
    all_predictions = np.array([])
    all_target = np.array([])

    model_path = args['model_path']
    save_result = args['save_result']
    batch_size = args['batch_size']
    include_null_class = args['include_null_class']

    time_today = str(datetime.date.today())
    clock = datetime.datetime.now().strftime('%H-%M-%S')

    # check if CUDA is available
    device = torch.device("cuda" if use_gpu else "cpu")
    model = CnnModel(INPUT_SIZE,NUM_CLASSES)
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))

    if save_result == '':
        filename = 'result_{}.json'.format(os.path.splitext(os.path.basename(model_path))[0])
    else:
        filename = '{}.json'.format(save_result)

    model.eval()
    json_data = {}
    json_index = 0

    print('--------------------------------------------------------------------------------------------')
    print('Evaluating Model with Batch Size: {}'.format(batch_size))
    print('Data for testing: {}'.format(len(test_loader)*batch_size))
    print('--------------------------------------------------------------------------------------------')

    print('Evaluating model, please wait...')
    # iterate over test data
    with torch.no_grad():
        for batch_idx, (data, target, old_data) in enumerate(test_loader):
            # move tensors to GPU if CUDA is available
            data, target = data.to(device), target.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            top_p, top_class = output.topk(1, dim=1)
            eval_pred = top_class.view(-1).long().cpu().numpy()
            eval_target = target.long().cpu().numpy()
            #equals = top_class == target.view(*top_class.shape).long()
            all_predictions = np.append(all_predictions, eval_pred)
            all_target =  np.append(all_target, eval_target)

            if batch_idx % 500 == 499:
                print('Evaluating [{}/{} ({:.0f}%)]'.format(
                batch_idx * batch_size, len(test_loader) * batch_size,
                100. * batch_idx / len(test_loader)))
            # iterate over batch
            for index,item in enumerate(old_data):
                row = item.cpu().numpy()[0]
                # build json
                json_data[json_index] = []
                if include_null_class:
                    json_data[json_index].append({
                    'chest_acc': [str(row[0]),str(row[1]),str(row[2])],
                    'left_ankle_acc': [str(row[3]),str(row[4]),str(row[5])],
                    'left_ankle_gyro': [str(row[6]),str(row[7]),str(row[8])],
                    'left_ankle_mag': [str(row[9]),str(row[10]),str(row[11])],
                    'right_ankle_acc': [str(row[12]),str(row[13]),str(row[14])],
                    'right_ankle_gyro': [str(row[15]),str(row[16]),str(row[17])],
                    'right_ankle_mag': [str(row[18]),str(row[19]),str(row[20])],
                    'actual_label': str(eval_target[index]),
                    'predicted_label': str(eval_pred[index])
                    })
                else:
                    json_data[json_index].append({
                    'chest_acc': [str(row[0]),str(row[1]),str(row[2])],
                    'left_ankle_acc': [str(row[3]),str(row[4]),str(row[5])],
                    'left_ankle_gyro': [str(row[6]),str(row[7]),str(row[8])],
                    'left_ankle_mag': [str(row[9]),str(row[10]),str(row[11])],
                    'right_ankle_acc': [str(row[12]),str(row[13]),str(row[14])],
                    'right_ankle_gyro': [str(row[15]),str(row[16]),str(row[17])],
                    'right_ankle_mag': [str(row[18]),str(row[19]),str(row[20])],
                    'actual_label': str(eval_target[index]+1),
                    'predicted_label': str(eval_pred[index]+1)
                    })
                json_index +=1

        # calculate metrics
        f1_total = metrics.f1_score(all_target, all_predictions, average='weighted')
        acc_total = metrics.accuracy_score(all_target, all_predictions)
        precision_total = metrics.precision_score(all_target, all_predictions, average='weighted')
        recall_total = metrics.recall_score(all_target, all_predictions, average='weighted')

        print("--------------------------------------------")
        print("Model {} Performance:".format(model_path))
        print("--------------------------------------------")
        print("F1 Score: {}".format(f1_total))
        print("Accuracy: {}".format(acc_total))
        print("Precission: {}".format(precision_total))
        print("Recall: {}".format(recall_total))
        print("--------------------------------------------")

        with open(os.path.join('result',filename), 'w') as outfile:
            json.dump(json_data, outfile)


if __name__ == '__main__':
    args = get_args()
    # look for gpu
    gpu_found = torch.cuda.is_available()
    # data
    files = glob.glob(args.test_folder+"/*.csv")
    test_dataset = ImuPoseDataset(files = files, include_null = args.include_null_class, return_old_data = True)
    if args.include_null_class:
        NUM_CLASSES = 7
        LABEL = ['No Activity','Standing still', 'Sitting and relaxing', 'Lying down', 'Walking', 'Climbing', 'Running']

    kwargs = {'num_workers': 1, 'pin_memory': True} if gpu_found else {}
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = args.batch_size, shuffle=True, **kwargs)

    test(test_loader, vars(args), gpu_found)
