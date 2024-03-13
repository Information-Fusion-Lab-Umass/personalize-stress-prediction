import os
import sys
import torch
import tqdm
import random

import src.bin.tensorify as tensorify
import src.utils.data_conversion_utils as conversions
import src.data_manager.student_life_var_binned_data_manager as data_manager
import src.bin.trainer as trainer
from statistics import mean as list_mean

from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer

from torch import nn
from copy import deepcopy
from src import definitions
from src.bin import statistics
from src.bin import checkpointing
from src.data_manager import cross_val
from src.models.multitask_learning import multitask_autoencoder
from src.utils.read_utils import read_pickle
from src.utils import write_utils

feature_list = data_manager.FEATURE_LIST

# ##### Pickle #####
data_file_path = 'data/training_data/shuffled_splits/training_date_normalized_shuffled_splits_select_features_no_prev_stress_all_students.pkl'
data = read_pickle(data_file_path)

################ get clusters from command line arg ###############
is_avg_stress = True if "avg_stress" in sys.argv[1] else False
print("running with average stress", is_avg_stress)

clusters = None
try:
    clusters = sys.argv[1] # file name in folder /Data/student_groups
except:
    clusters = 'one_for_each'
    exit()
print('The groups: ' + clusters)
groups_file_path = 'src/experiments/clustering/student_groups/' + clusters + '.pkl'
student_groups = read_pickle(groups_file_path) # student groups

# check how students are distributed
if not is_avg_stress:
    print("student distribution: ")
    group_ids = list()
    student_ids = list()
    rev_groups = dict()
    for student in student_groups:
        if rev_groups.get(student_groups[student]) != None:
            rev_groups[student_groups[student]].append(student)
        else:
            rev_groups[student_groups[student]] = [student]
            group_ids.append(student_groups[student])
        student_ids.append(student.split('_')[1])
    group_ids = set(group_ids)
    for group in rev_groups:
        print(group + ': ' + str(rev_groups[group]))
#####################################################################

######## Split data #################################################
# k-fold cross validation
stratification_type = None
try:
    stratification_type = sys.argv[2]
except:
    stratification_type = "students"
#splits = cross_val.get_k_fod_cross_val_splits_stratified_by_students(data=data, groups = student_groups, n_splits=5, stratification_type=stratification_type)

# leave one subject out
# num_subject = 5
# ids = random.sample(student_ids, num_subject)
batch = dict()
batch['0'] = ["4", "7", "8"]
batch['1'] = ["10", "14", "16", "17", "19"]
batch['2'] = ["22", "23", "24", "32", "33"]
batch['3'] = ["35", "36", "43", "44", "49"]
batch['4'] = ["51", "52", "53", "57", "58"]
# missing = [8, 10, 14, 22, 23, 36, 43, 49, 51, 57, 58]

#ids = student_ids
try:
    ids = batch[sys.argv[3]]
except:
    print("invalid batch num! ")
    exit()
print('Choosen student: ', ids)
splits = cross_val.leave_one_subject_out_split(data=data, groups=student_groups, ids=ids, subject='students')

print("Splits: ", len(splits))
#####################################################################

################################## Init #############################
use_historgram = True
autoencoder_bottle_neck_feature_size = 128
autoencoder_num_layers = 1
alpha , beta = 0.0001, 1
decay = 0.0001
first_key = next(iter(data['data'].keys()))
if use_historgram:
    num_features = len(data['data'][first_key][4][0])
else:
    num_features = len(data['data'][first_key][0][0])
num_covariates = len(data['data'][first_key][definitions.COVARIATE_DATA_IDX])
shared_hidden_layer_size = 256
user_dense_layer_hidden_size = 64
num_classes = 3
learning_rate = 0.000001
n_epochs = 1 # 500
shared_layer_dropout_prob=0.00
user_head_dropout_prob=0.00

device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')

print("Num Features:", num_features)
print("Device: ", device)
print("Num_covariates:", num_covariates)

cuda_enabled = torch.cuda.is_available()
tensorified_data = tensorify.tensorify_data_gru_d(deepcopy(data), cuda_enabled)
student_list = conversions.extract_distinct_student_idsfrom_keys(data['data'].keys())

#####################################################################

# ################## For inspecting score of each group ###################################
# # store the score of each groups
# # to see the generalization ability of each group
# # only when split by groups
# group_val_score = dict() # map: group -> list(val_scores)
# for group in group_ids:
#     group_val_score[group] = list()
# group_label_pred = dict() # map: group -> map: labels -> list, map: preds -> list
# for group in group_val_score:
#     group_label_pred[group] = dict()
#     group_label_pred[group]['labels'] = list()
#     group_label_pred[group]['preds'] = list()

# key_group = dict() # map: key -> group
# for key in data['data']:
#     key_group[key] = student_groups['student_' + key.split('_')[0]]
# ########################################################################################

split_val_scores = list()
best_score_epoch_log = list()
best_models = list()

split_roc_macro = list()
split_roc_micro = list()
split_roc_weighted = list()

confusion_matrices = dict() # map: str(id) -> nparray(confusion matrix)

for split_no, split in enumerate(splits):
    print("Split No: ", split_no)

    best_split_score = -1
    epoch_at_best_score = 0
    best_model = None

    best_val_roc_macro = -1
    best_val_roc_micro = -1
    best_val_roc_weighted = -1

    tensorified_data['train_ids'] = split['train_ids']
    data['train_ids'] = split['train_ids']

    tensorified_data['val_ids'] = split['val_ids']
    data['val_ids'] = split['val_ids']

    tensorified_data['test_ids'] = list()
    
    leaved_student = data['val_ids'][0].split('_')[0]
    print('############# val student!!! {} ################'.format(leaved_student))

    validation_user_statistics_over_epochs = list()

    class_weights = torch.tensor(statistics.get_class_weights_in_inverse_proportion(data))
    class_weights = torch.tensor([0.6456, 0.5635, 1.0000])
    print("Class Weights: ", class_weights)

    #### fetch groups ####
    groups = student_groups
    if is_avg_stress:
        groups = student_groups[leaved_student]

    model = multitask_autoencoder.MultiTaskAutoEncoderLearner(
        conversions.prepend_ids_with_string(student_list, "student_"),
        groups,
        num_features,
        autoencoder_bottle_neck_feature_size,
        autoencoder_num_layers,
        shared_hidden_layer_size,
        user_dense_layer_hidden_size,
        num_classes,
        num_covariates,
        shared_layer_dropout_prob,
        user_head_dropout_prob)
    if cuda_enabled:
        model.cuda()
        class_weights = class_weights.cuda()

    reconstruction_criterion = torch.nn.L1Loss(reduction="sum")
    classification_criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)

    for epoch in tqdm.tqdm(range(n_epochs)):

        (train_total_loss, train_total_reconstruction_loss, train_total_classification_loss,
         train_labels, train_preds, train_users) = trainer.evaluate_multitask_learner(tensorified_data,
                                                                                      'train_ids',
                                                                                      num_classes,
                                                                                      model,
                                                                                      reconstruction_criterion,
                                                                                      classification_criterion,
                                                                                      device,
                                                                                      optimizer=optimizer,
                                                                                      alpha=alpha,
                                                                                      beta=beta,
                                                                                      use_histogram=use_historgram)
        
        (val_total_loss, val_total_reconstruction_loss, val_total_classification_loss,
         val_labels, val_preds, val_users) = trainer.evaluate_multitask_learner(tensorified_data,
                                                                                'val_ids',
                                                                                num_classes,
                                                                                model,
                                                                                reconstruction_criterion,
                                                                                classification_criterion,
                                                                                device,
                                                                                alpha=alpha,
                                                                                beta=beta,
                                                                                use_histogram=use_historgram)
        ############ For inspecting score of each group ###################################
        # group_label_pred_this_split = deepcopy(group_label_pred)
        # (val_total_loss, val_total_reconstruction_loss, val_total_classification_loss,
        #  val_labels, val_preds, val_users) = trainer.evaluate_multitask_learner(tensorified_data,
        #                                                                         'val_ids',
        #                                                                         num_classes,
        #                                                                         model,
        #                                                                         reconstruction_criterion,
        #                                                                         classification_criterion,
        #                                                                         device,
        #                                                                         alpha=alpha,
        #                                                                         beta=beta,
        #                                                                         use_histogram=use_historgram,
        #                                                                         key_group=key_group,
        #                                                                         group_label_pred=group_label_pred_this_split)
        ################################################################################

        ######## Appending Metrics ########
        train_label_list = conversions.tensor_list_to_int_list(train_labels)
        train_pred_list = conversions.tensor_list_to_int_list(train_preds)
        val_label_list = conversions.tensor_list_to_int_list(val_labels)
        val_pred_list = conversions.tensor_list_to_int_list(val_preds)

        train_scores = metrics.precision_recall_fscore_support(train_label_list, train_pred_list, average='weighted')
        val_scores = metrics.precision_recall_fscore_support(val_label_list, val_pred_list, average='weighted')

        # Compute AUC sorces
        mlb = MultiLabelBinarizer()
        mlb.fit([val_label_list])
        print('val classes', [i for i in set(val_label_list)])
        print('pred classes', [i for i in set(val_pred_list)])
        y_true = mlb.transform([[i] for i in val_label_list])
        y_pred = mlb.transform([[i] for i in val_pred_list])
        print('y_true shape', y_true.shape)
        print('y_pred shape', y_pred.shape)
        print("confusion matrix: ")
        con_matrix = metrics.confusion_matrix(val_label_list, val_pred_list, labels=[0, 1, 2])
        print(con_matrix)

        val_roc_macro = metrics.roc_auc_score(y_true, y_pred, average='macro')
        val_roc_micro = metrics.roc_auc_score(y_true, y_pred, average='micro')
        val_roc_weighted = metrics.roc_auc_score(y_true, y_pred, average='weighted')

        if val_roc_macro > best_val_roc_macro:
            best_val_roc_macro = val_roc_macro
        if val_roc_micro > best_val_roc_micro:
            best_val_roc_micro = val_roc_micro
        if val_roc_weighted > best_val_roc_weighted:
            best_val_roc_weighted = val_roc_weighted
        print("AUC (macro) this Epoch: {} Best Score: {}".format(val_roc_macro, best_val_roc_macro))
        print("AUC (micro) this Epoch: {} Best Score: {}".format(val_roc_micro, best_val_roc_micro))
        print("AUC (weighted) this Epoch: {} Best Score: {}".format(val_roc_weighted, best_val_roc_weighted))

        # validation_user_statistics_over_epochs.append(statistics.generate_training_statistics_for_user(val_labels,
        #                                                                                                val_preds,
        #                                                                                                val_users))

        if val_scores[2] > best_split_score:
            best_split_score = val_scores[2]
            epoch_at_best_score = epoch
            best_model = deepcopy(model)

            confusion_matrices[data['val_ids'][0].split('_')[0]] = con_matrix

        print("Split: {} Score This Epoch: {} Best Score: {}".format(split_no, val_scores[2], best_split_score))

        ############ For inspecting score of each group ###################################
        # for group in group_val_score:
        #     val_score = metrics.precision_recall_fscore_support(group_label_pred_this_split[group]['labels'], group_label_pred_this_split[group]['preds'], average='weighted')
        #     try:
        #         group_val_score[group][int(split_no)] = max(group_val_score[group][int(split_no)], val_score[2])
        #     except:
        #         group_val_score[group].append(val_score[2])
            #print(group + ': ' + str(group_val_score[group]))
        ################################################################################

    split_val_scores.append(best_split_score)
    best_score_epoch_log.append(epoch_at_best_score)
    best_models.append(deepcopy(best_model))

    split_roc_macro.append(best_val_roc_macro)
    split_roc_micro.append(best_val_roc_micro)
    split_roc_weighted.append(best_val_roc_weighted)

print('F1 split scores: ' + str(split_val_scores))
#print("alpha: {} Beta: {}".format(alpha, beta))
print("Avg F1 Cross Val Score: {}".format(list_mean(split_val_scores)))
max_idx = split_val_scores.index(max(split_val_scores))

print('AUC (macro) scores: ' + str(split_roc_macro))
print("Avg AUC macro Val Score: {}".format(list_mean(split_roc_macro)))
print('AUC (micro) scores: ' + str(split_roc_micro))
print("Avg AUC micro Val Score: {}".format(list_mean(split_roc_micro)))
print('AUC (weighted) scores: ' + str(split_roc_weighted))
print("Avg AUC weighted Val Score: {}".format(list_mean(split_roc_weighted)))

############## For inspecting score of each group #################################
# for group in group_val_score:
#     group_val_score[group] = sum(group_val_score[group]) / len(group_val_score[group])
#     print('Avg Cross Val Score of ' + group + ': ' + str(group_val_score[group]))
################################################################################

scores_and_epochs = (split_val_scores, epoch_at_best_score)
scores_and_epochs_file_name = os.path.join('data', "cross_val_scores/multitask_autoencoder_" + clusters + '_' + sys.argv[3]  + ".pkl")
write_utils.data_structure_to_pickle(scores_and_epochs, scores_and_epochs_file_name)

AUC_scores = (split_roc_macro, split_roc_micro, split_roc_weighted)
AUC_scores_file_name = os.path.join('data', "cross_val_scores/auc_multitask_autoencoder_" + clusters + '_' + sys.argv[3]  + ".pkl")
write_utils.data_structure_to_pickle(AUC_scores, AUC_scores_file_name)

model_file_name = "saved_models/multitask_lstm-ae_{}_{}.model".format(clusters, sys.argv[3])
model_file_name = os.path.join('data', model_file_name)
checkpointing.save_checkpoint(best_models[max_idx].state_dict(), model_file_name)

confusion_matrices_file_name = os.path.join('data', "check/confusion_matrices_" + clusters + '_' + sys.argv[3]  + ".pkl")
write_utils.data_structure_to_pickle(confusion_matrices, confusion_matrices_file_name)