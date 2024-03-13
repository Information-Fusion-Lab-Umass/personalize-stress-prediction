import pickle
import numpy as np
from copy import deepcopy
import torch
from sklearn import metrics

import src.utils.tensorify as tensorify
from src.utils.train_val_utils import *

def check_BCN():
    # load overal scores
    with open('data/cross_val_scores/calm_net_with_branching_5fold_3_with_generic.pkl', 'rb') as f:
    # with open('data/cross_val_scores/calm_net_with_branching_loocv_3_7.pkl', 'rb') as f:
        var = pickle.load(f)
    f1s = np.array([np.max(i['val_f1']['micro']) for i in var if len(i) != 0])
    print('f1', f1s.mean())
    
    # load data to fetch the splitting info
    data_file_path = 'data/training_data/shuffled_splits/training_date_normalized_shuffled_splits_select_features_no_prev_stress_all_students.pkl'
    data = read_data(data_file_path)
    data = tensorify.tensorify_data_gru_d(deepcopy(data), torch.cuda.is_available())

    # load groups
    clusters_name = 'one_for_each' # 'one_for_each', 'all_in_one'
    print('The groups: ' + clusters_name)
    groups_file_path = 'src/experiments/clustering/student_groups/' + clusters_name + '.pkl'
    student_groups = read_data(groups_file_path) # student groups
    
    splits = get_splits('5fold', data, student_groups, days_include=0)
    
    # fetching info from which fold
    chosen_fold = 4

    # start fetching
    student_ys = dict() # map: student_id -> [y_pred, y_true, branch_id]

    train_val_record = var[chosen_fold]
    model = train_val_record['model']
    max_epoch = np.argmax(train_val_record['val_f1']['micro'])
    output = train_val_record['outputs'][max_epoch]

    y_pred = np.argmax(output, axis=1)
    # y_true = val_data['labels'].cpu().detach().numpy()

    # fetch the validation ids
    i = 0
    for split_no, split in enumerate(splits):
        if i == chosen_fold:
            val_ids = split['val_ids']
            break
        i += 1
    
    # fetch performance for each student
    i = 0
    for key in val_ids:
        actual_data, covariate_data, histogram_data, label = data['data'][key]
        student_id = key.split('_')[0]
        if student_ys.get(student_id) == None:
            student_ys[student_id] = [[[y_pred[i]], [label.squeeze().cpu().detach().item()]]]
        else:
            student_ys[student_id][0][0].append(y_pred[i])
            student_ys[student_id][0][1].append(label.squeeze().cpu().detach().item())
        i += 1
    
    # fetch branch assigned
    for student_id in student_ys:
        group_id = student_groups['student_{}'.format(student_id)]
        branch = model.branching.probabilities[group_id].argmax().cpu().detach().item()
        student_ys[student_id].append(branch)
    
        # calculate performances
        # print(student_ys[student_id][0][1])
        # print(student_ys[student_id][0][0])
        student_ys[student_id][0] = metrics.f1_score(
            student_ys[student_id][0][1], 
            student_ys[student_id][0][0], 
            average='micro'
        )
    
    # check
    for student_id in student_ys:
        print(student_id, student_ys[student_id])

def check_loocv():
    # load overal scores
    # with open('data/cross_val_scores/calm_net_with_branching_5fold_3.pkl', 'rb') as f:
    with open('data/cross_val_scores/semi_online_learning/calm_net_with_branching_generic_head_loocv_3_7.pkl', 'rb') as f:
        var = pickle.load(f)
    f1s = np.array([np.max(i['val_f1']['micro']) for i in var if len(i) != 0])
    print('f1', f1s.mean())
    
    # fetching info from which fold
    chosen_fold = 8

    # start fetching
    student_ys = dict() # map: student_id -> [y_pred, y_true, branch_id]

    train_val_record = var[chosen_fold]
    model = train_val_record['model']
    # for i in model.branching.probabilities:
    #     print(i)
    # exit()
    max_epoch = np.argmax(train_val_record['val_f1']['micro'])
    output = train_val_record['outputs'][max_epoch]

    # load data to fetch the splitting info
    data_file_path = 'data/training_data/shuffled_splits/training_date_normalized_shuffled_splits_select_features_no_prev_stress_all_students.pkl'
    data = read_data(data_file_path)
    data = tensorify.tensorify_data_gru_d(deepcopy(data), torch.cuda.is_available())

    # load groups
    clusters_name = 'one_for_each' # 'one_for_each', 'all_in_one'
    print('The groups: ' + clusters_name)
    groups_file_path = 'src/experiments/clustering/student_groups/' + clusters_name + '.pkl'
    student_groups = read_data(groups_file_path) # student groups
    
    splits = get_splits('loocv', data, student_groups, days_include=0)

    # fetch the validation ids
    i = 0
    for split_no, split in enumerate(splits):
        if i == chosen_fold:
            val_ids = split['train_ids']
            break
        i += 1
    
    # fetch performance for each student
    i = 0
    for key in val_ids:
        actual_data, covariate_data, histogram_data, label = data['data'][key]
        student_id = key.split('_')[0]
        if student_ys.get(student_id) == None:
            student_ys[student_id] = [0]
        i += 1
    
    # fetch branch assigned
    for student_id in student_ys:
        group_id = student_groups['student_{}'.format(student_id)]
        branch = model.branching.probabilities[group_id].argmax().cpu().detach().item()
        student_ys[student_id].append(branch)
    
    # check
    for student_id in student_ys:
        print(student_id, student_ys[student_id])

if __name__ == '__main__':
    # check_loocv()
    check_BCN()