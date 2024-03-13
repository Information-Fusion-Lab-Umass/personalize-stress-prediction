import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import metrics

from src.bin import validations
from src.utils import data_conversion_utils as conversions
from src.utils import object_generator_utils as object_generator
from src.models.user_dense_heads import softmax_select

HISTOGRAM_IDX_AFTER_TENSORIFY = 2

def branching_on_leaved_out_with_exist_heads(
    data,
    key_set: str,
    epochs,
    model,
    classification_criterion,
    device,
    use_covariates=True,
    use_histogram=False
):
    # extract keys from first week (can be optimized by number of days, at begin_interval)
    month_days = {0: 0, 1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31} # map: month -> # days

    begin_interval = 7 # how many data used for the leaved out student
    start_day = -1

    min_data_need = list()
    val_keys = list()
    for key in data[key_set]:
        month = int(key.split('_')[1])
        day = int(key.split('_')[2])
        curr_day = sum([month_days[i] for i in range(month + 1)]) + day
        if start_day < 0:
            start_day = curr_day
        else:
            if curr_day - start_day >= begin_interval:
                val_keys.append(key)
            else:
                min_data_need.append(key)
    
    heads_ind = dict()
    ind = 0
    for head in model.user_heads.user_layer:
        heads_ind[ind] = head
        ind += 1

    # class weighted_out(nn.Module):
    #     def __init__(self, num_branches):
    #         super().__init__()
    #         self.p = nn.Parameter(torch.exp(torch.ones(num_branches, device=torch.device("cuda"))), requires_grad=True)
        
    #     def forward(self):
    #         # weight sum to [0, 1]
    #         return self.p
    
    # leaved_out_head = weighted_out(len((heads_ind)))
    leaved_out_head = softmax_select(len(heads_ind))

    optimizer = torch.optim.Adam(
        [
            {'params': leaved_out_head.parameters()},
        ],
        lr=1e-5,
        weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.996)
    train_loss, val_loss = list(), list()
    f1_train_scores, f1_val_scores = list(), list()
    val_roc_macros, val_roc_micros, val_roc_weighteds = list(), list(), list()
    best_split_score = -1
    best_model = None
    best_confmat = None
    best_out = None
    for epoch in range(epochs):
        train_labels = list()
        train_preds = list()
        val_labels = list()
        val_preds = list()
        total_train_loss = 0
        leaved_out_head.train()
        for key in min_data_need:
            student_id = conversions.extract_student_id_from_key(key)
            student_key = 'student_' + str(student_id)
            actual_data, covariate_data, histogram_data, train_label = data['data'][key]
            actual_data = actual_data[0].unsqueeze(0)
            if use_histogram:
                actual_data = histogram_data.unsqueeze(0)

            # forward
            # select branch
            prob_out, branch_ind = leaved_out_head()
            chosen_key = heads_ind[int(branch_ind)]

            # prior out
            decoded_output, y_pred = model(
                chosen_key,
                '-1',
                actual_data,
                covariate_data if use_covariates else None
            )

            # with probability factor
            y_pred *= prob_out

            # # weighted, not drop
            # prob = leaved_out_head()
            # final_out = 0
            # for ind in heads_ind:
            #     chosen_key = heads_ind[ind]

            #     decoded_output, y_pred = model(
            #         chosen_key,
            #         '-1',
            #         actual_data,
            #         covariate_data if use_covariates else None
            #     )

            #     final_out += y_pred * prob[ind]
            
            # y_pred = final_out

            # compute loss
            classification_loss = classification_criterion(y_pred, train_label)
            total_train_loss += classification_loss.item()

            # optimize
            model.zero_grad()
            leaved_out_head.zero_grad()
            classification_loss.backward()
            optimizer.step()

            train_labels.append(train_label)
            predicted_class = get_predicted_class(y_pred)
            train_preds.append(predicted_class)
        train_loss.append(total_train_loss)

        # validate
        total_val_loss = 0
        leaved_out_head.eval()
        outs = list()
        for key in val_keys:
            student_id = conversions.extract_student_id_from_key(key)
            student_key = 'student_' + str(student_id)
            actual_data, covariate_data, histogram_data, train_label = data['data'][key]
            actual_data = actual_data[0].unsqueeze(0)
            if use_histogram:
                actual_data = histogram_data.unsqueeze(0)

            # forward
            # select branch
            prob_out, branch_ind = leaved_out_head()
            chosen_key = heads_ind[int(branch_ind)]

            # prior out
            decoded_output, y_pred = model(
                chosen_key,
                '-1',
                actual_data,
                covariate_data if use_covariates else None
            )

            outs.append(y_pred.cpu().detach().numpy())

            # # weighted, not drop
            # prob = leaved_out_head()
            # final_out = 0
            # for ind in heads_ind:
            #     chosen_key = heads_ind[ind]

            #     decoded_output, y_pred = model(
            #         chosen_key,
            #         '-1',
            #         actual_data,
            #         covariate_data if use_covariates else None
            #     )

            #     final_out += y_pred * prob[ind]
            
            # y_pred = final_out

            # compute loss
            classification_loss = classification_criterion(y_pred, train_label)
            total_val_loss += classification_loss.item()

            # clear cache
            model.zero_grad()
            leaved_out_head.zero_grad()
            classification_loss.backward()
            model.zero_grad()
            leaved_out_head.zero_grad()

            val_labels.append(train_label)
            predicted_class = get_predicted_class(y_pred)
            val_preds.append(predicted_class)
        val_loss.append(total_val_loss)

        # learning rate decay
        scheduler.step()

        # update score
        train_label_list = conversions.tensor_list_to_int_list(train_labels)
        train_pred_list = conversions.tensor_list_to_int_list(train_preds)
        val_label_list = conversions.tensor_list_to_int_list(val_labels)
        val_pred_list = conversions.tensor_list_to_int_list(val_preds)

        train_scores = metrics.precision_recall_fscore_support(train_label_list, train_pred_list, average='weighted')
        val_scores = metrics.precision_recall_fscore_support(val_label_list, val_pred_list, average='weighted')

        f1_train_scores.append(train_scores[2])
        f1_val_scores.append(val_scores[2])

        # compute val AUC scores
        mlb = MultiLabelBinarizer()
        mlb.fit([[0],[1], [2]])
        y_true = mlb.transform([[i] for i in val_label_list])
        y_pred = mlb.transform([[i] for i in val_pred_list])
        print("confusion matrix: ")
        con_matrix = metrics.confusion_matrix(val_label_list, val_pred_list, labels=[0, 1, 2])
        print(con_matrix)

        val_roc_macro = None
        try:
            val_roc_macro = metrics.roc_auc_score(y_true, y_pred, average='macro')
        except:
            val_roc_macro = 0.0

        val_roc_micro = None
        try:
            val_roc_micro = metrics.roc_auc_score(y_true, y_pred, average='micro')
        except:
            val_roc_micro = 0.0
        val_roc_weighted = None
        try:
            val_roc_weighted = metrics.roc_auc_score(y_true, y_pred, average='weighted')
        except:
            val_roc_weighted = 0.0
        val_roc_macros.append(val_roc_macro)    
        val_roc_micros.append(val_roc_micro)    
        val_roc_weighteds.append(val_roc_weighted)

        if train_scores[2] > best_split_score:
            best_split_score = train_scores[2]
            best_model = deepcopy(model)
            # best_model = model
            # best_branching_scores = copy.deepcopy(branching_scores)

            best_confmat = con_matrix
            best_out = outs

        print('epoch {}, val_score {}'.format(epoch, val_scores))

    return train_loss, val_loss, leaved_out_head, f1_train_scores, f1_val_scores, val_roc_macros, val_roc_micros, val_roc_weighteds, best_confmat, best_out

def loocv_multitask_learner_with_branching_validation(
    data,
    ids, 
    key_set: str,
    num_classes,
    num_branches,
    multitask_lerner_model,
    reconstruction_criterion,
    classification_criterion,
    device,
    optimizer=None,
    alpha=1,
    beta=1,
    use_histogram=False,
    histogram_seq_len=None,
    ordinal_regression=False,
    use_covariates=True):

    # validations.validate_data_dict_keys(data)
    # validate_key_set_str(key_set)

    total_reconstruction_loss = 0
    total_classification_loss = 0
    total_joint_loss = 0

    labels = list()
    predictions = list()
    users = list()

    if not optimizer:
        multitask_lerner_model.eval()
        multitask_lerner_model.user_heads.branching_probs.eval()
    else:
        multitask_lerner_model.train()
        multitask_lerner_model.user_heads.branching_probs.train()

    # extract keys from first week (can be optimized by number of days, at begin_interval)
    month_days = {0: 0, 1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31} # map: month -> # days

    begin_interval = 7 # how many data used for the leaved out student
    start_day = -1

    min_data_need = list()
    val_keys = list()
    for key in data[key_set]:
        month = int(key.split('_')[1])
        day = int(key.split('_')[2])
        curr_day = sum([month_days[i] for i in range(month + 1)]) + day
        if start_day < 0:
            start_day = curr_day
        else:
            if curr_day - start_day >= begin_interval:
                val_keys.append(key)
            else:
                min_data_need.append(key)
    
    # find best existing head
    heads_score = dict()
    for head in multitask_lerner_model.user_heads.user_layer:
        heads_score[head] = 0

    # inner loop, loop over existing heads
    for key in min_data_need:
        student_id = conversions.extract_student_id_from_key(key)
        actual_data, covariate_data, histogram_data, train_label = data['data'][key]

        if ordinal_regression:
            train_label_vector = get_target_vector_for_ordinal_regression(train_label, num_classes, device)

        actual_data = actual_data[0].unsqueeze(0)
        if use_histogram:
            if histogram_seq_len:
                histogram_data = histogram_data[:max(histogram_seq_len, len(histogram_data))]
            actual_data = histogram_data.unsqueeze(0)
        
        # forward
        min_loss = np.inf
        min_key = None
        for id_ in ids:
            if str(id_) == str(student_id):
                continue
            student_key = 'student_' + str(id_)
            decoded_output, y_pred = multitask_lerner_model(
                student_key,
                '-1',
                actual_data,
                covariate_data if use_covariates else None
            )
            # decoded output is `None` if training on only co-variates.
            reconstruction_loss = reconstruction_criterion(actual_data, decoded_output) if decoded_output is not None else object_generator.get_tensor_on_correct_device([0])

            # compute loss
            classification_loss = None
            if ordinal_regression:
                classification_loss = classification_criterion(y_pred, train_label_vector)
            else:
                classification_loss = classification_criterion(y_pred, train_label)

            joint_loss = alpha * reconstruction_loss + beta * classification_loss
            curr_loss = joint_loss.item()

            if min_key == None:
                min_key = student_key
                min_loss = curr_loss
            elif curr_loss < min_loss:
                min_key = student_key
                min_loss = curr_loss
        heads_score[min_key] += 1
    
    chosen_key = None
    chosen_score = -1
    for skey in heads_score:
        if heads_score[skey] > chosen_score:
            chosen_score = heads_score[skey]
            chosen_key = skey
    
    for key in val_keys:
        student_id = conversions.extract_student_id_from_key(key)
        student_key = 'student_' + str(student_id)
        actual_data, covariate_data, histogram_data, train_label = data['data'][key]

        if ordinal_regression:
            train_label_vector = get_target_vector_for_ordinal_regression(train_label, num_classes, device)

        actual_data = actual_data[0].unsqueeze(0)
        if use_histogram:
            if histogram_seq_len:
                histogram_data = histogram_data[:max(histogram_seq_len, len(histogram_data))]
            actual_data = histogram_data.unsqueeze(0)
        
        # forward
        decoded_output, y_pred = multitask_lerner_model(
            chosen_key,
            '-1',
            actual_data,
            covariate_data if use_covariates else None
        )

        # decoded output is `None` if training on only co-variates.
        reconstruction_loss = reconstruction_criterion(actual_data, decoded_output) if decoded_output is not None else object_generator.get_tensor_on_correct_device([0])
        total_reconstruction_loss += reconstruction_loss.item()

        # compute loss
        classification_loss = None
        if ordinal_regression:
            classification_loss = classification_criterion(y_pred, train_label_vector)
        else:
            classification_loss = classification_criterion(y_pred, train_label)

        total_classification_loss += classification_loss.item()

        joint_loss = alpha * reconstruction_loss + beta * classification_loss
        total_joint_loss += joint_loss.item()

        labels.append(train_label)
        predicted_class = get_predicted_class(y_pred, ordinal_regression=ordinal_regression)
        predictions.append(predicted_class)
        users.append(student_id)
    
    return total_joint_loss, total_reconstruction_loss, total_classification_loss, labels, predictions, users

def evaluate_multitask_learner_with_branching(data,
                                            key_set: str,
                                            num_classes,
                                            num_branches,
                                            branching_scores,
                                            multitask_lerner_model,
                                            reconstruction_criterion,
                                            classification_criterion,
                                            device,
                                            optimizer=None,
                                            alpha=1,
                                            beta=1,
                                            use_histogram=False,
                                            histogram_seq_len=None,
                                            ordinal_regression=False,
                                            use_covariates=True):
    validations.validate_data_dict_keys(data)
    validate_key_set_str(key_set)

    total_reconstruction_loss = 0
    total_classification_loss = 0
    total_joint_loss = 0

    labels = list()
    predictions = list()
    users = list()

    if not optimizer:
        multitask_lerner_model.eval()
        multitask_lerner_model.user_heads.branching_probs.eval()
    else:
        multitask_lerner_model.train()
        multitask_lerner_model.user_heads.branching_probs.train()

    for key in data[key_set]:
        student_id = conversions.extract_student_id_from_key(key)
        student_key = 'student_' + str(student_id)
        actual_data, covariate_data, histogram_data, train_label = data['data'][key]

        if ordinal_regression:
            train_label_vector = get_target_vector_for_ordinal_regression(train_label, num_classes, device)

        actual_data = actual_data[0].unsqueeze(0)
        if use_histogram:
            if histogram_seq_len:
                histogram_data = histogram_data[:max(histogram_seq_len, len(histogram_data))]
            actual_data = histogram_data.unsqueeze(0)

        # forward
        decoded_output, y_pred = multitask_lerner_model(
            student_key,
            '-1',
            actual_data,
            covariate_data if use_covariates else None
        )

        # decoded output is `None` if training on only co-variates.
        reconstruction_loss = reconstruction_criterion(actual_data, decoded_output) if decoded_output is not None else object_generator.get_tensor_on_correct_device([0])
        total_reconstruction_loss += reconstruction_loss.item()

        # compute loss
        classification_loss = None
        if ordinal_regression:
            classification_loss = classification_criterion(y_pred, train_label_vector)
        else:
            classification_loss = classification_criterion(y_pred, train_label)

        total_classification_loss += classification_loss.item()

        joint_loss = alpha * reconstruction_loss + beta * classification_loss
        total_joint_loss += joint_loss.item()

        # if training, optimize
        if optimizer:
            multitask_lerner_model.zero_grad()
            joint_loss.backward()
            optimizer.step()

        labels.append(train_label)
        predicted_class = get_predicted_class(y_pred, ordinal_regression=ordinal_regression)
        predictions.append(predicted_class)
        users.append(student_id)

    return total_joint_loss, total_reconstruction_loss, total_classification_loss, labels, predictions, users

# # Try by yunfeiluo in Fall 2020 #########################################################################################
# def evaluate_multitask_learner_with_branching(data,
#                                             key_set: str,
#                                             num_classes,
#                                             num_branches,
#                                             branching_scores,
#                                             multitask_lerner_model,
#                                             reconstruction_criterion,
#                                             classification_criterion,
#                                             device,
#                                             optimizer=None,
#                                             alpha=1,
#                                             beta=1,
#                                             use_histogram=False,
#                                             histogram_seq_len=None,
#                                             ordinal_regression=False,
#                                             use_covariates=True):
#     validations.validate_data_dict_keys(data)
#     validate_key_set_str(key_set)

#     total_reconstruction_loss = 0
#     total_classification_loss = 0
#     total_joint_loss = 0

#     labels = list()
#     predictions = list()
#     users = list()

#     if not optimizer:
#         multitask_lerner_model.eval()
#     else:
#         multitask_lerner_model.train()

#     for key in data[key_set]:
#         student_id = conversions.extract_student_id_from_key(key)
#         student_key = 'student_' + str(student_id)
#         actual_data, covariate_data, histogram_data, train_label = data['data'][key]

#         if ordinal_regression:
#             train_label_vector = get_target_vector_for_ordinal_regression(train_label, num_classes, device)

#         actual_data = actual_data[0].unsqueeze(0)
#         if use_histogram:
#             if histogram_seq_len:
#                 histogram_data = histogram_data[:max(histogram_seq_len, len(histogram_data))]
#             actual_data = histogram_data.unsqueeze(0)

#         # if not in training process
#         if not optimizer:
#             decoded_output, y_pred, shared_out = multitask_lerner_model(student_key,
#                                                                         str(np.argmax(branching_scores[student_key])),
#                                                                         actual_data,
#                                                                         covariate_data if use_covariates else None)
            
#             # decoded output is `None` if training on only co-variates.
#             reconstruction_loss = reconstruction_criterion(actual_data, decoded_output) if decoded_output is not None else object_generator.get_tensor_on_correct_device([0])
#             total_reconstruction_loss += reconstruction_loss.item()
            
#             classification_loss = None
#             if ordinal_regression:
#                 classification_loss = classification_criterion(y_pred, train_label_vector)
#             else:
#                 classification_loss = classification_criterion(y_pred, train_label)
            
#             total_classification_loss += classification_loss.item()

#             joint_loss = alpha * reconstruction_loss + beta * classification_loss
#             total_joint_loss += joint_loss.item()

#             labels.append(train_label)
#             predicted_class = get_predicted_class(y_pred, ordinal_regression=ordinal_regression)
#             predictions.append(predicted_class)
#             users.append(student_id)

#             continue

#         # compute out for first branching block, with the autoencoder out
#         decoded_output, y_pred, shared_out = multitask_lerner_model(student_key,
#                                                                     '0',
#                                                                     actual_data,
#                                                                     covariate_data if use_covariates else None)

#         # decoded output is `None` if training on only co-variates.
#         reconstruction_loss = reconstruction_criterion(actual_data, decoded_output) if decoded_output is not None else object_generator.get_tensor_on_correct_device([0])
#         total_reconstruction_loss += reconstruction_loss.item()
        
#         classification_loss = None
#         if ordinal_regression:
#             classification_loss = classification_criterion(y_pred, train_label_vector)
#         else:
#             classification_loss = classification_criterion(y_pred, train_label)

#         # Iterate over the rest branching blocks, find the best block
#         best_branch_ind = 0
#         best_branch_loss = classification_loss
#         for branch_id in range(1, num_branches):
#             y_pred = multitask_lerner_model(
#                 student_key,
#                 str(branch_id),
#                 actual_data,
#                 covariate_data if use_covariates else None,
#                 shared_out=shared_out,
#                 has_shared=True
#             )

#             # compute loss
#             classification_loss = None
#             if ordinal_regression:
#                 classification_loss = classification_criterion(y_pred, train_label_vector)
#             else:
#                 classification_loss = classification_criterion(y_pred, train_label)
            
#             if classification_loss.item() > best_branch_loss.item():
#                 best_branch_loss = classification_loss
#                 best_branch_ind = branch_id

#         branching_scores[student_key][best_branch_ind] += 1

#         total_classification_loss += best_branch_loss.item()

#         joint_loss = alpha * reconstruction_loss + beta * best_branch_loss
#         total_joint_loss += joint_loss.item()

#         # Optimize
#         multitask_lerner_model.zero_grad()
#         joint_loss.backward()
#         optimizer.step()

#         labels.append(train_label)
#         predicted_class = get_predicted_class(y_pred, ordinal_regression=ordinal_regression)
#         predictions.append(predicted_class)
#         users.append(student_id)

#     return total_joint_loss, total_reconstruction_loss, total_classification_loss, labels, predictions, users

########################################################################################################################

def validate_key_set_str(key_set: str):
    assert key_set in ['test_ids', 'val_ids', 'train_ids'], "Invalid Key Set. Must be either test or val!"


def evaluate_set(data, key_set: str, model, criterion, optimizer=None, train_covariates=False):
    validations.validate_data_dict_keys(data)
    validate_key_set_str(key_set)
    total_loss = 0
    labels = []
    predictions = []

    if not optimizer:
        model.eval()
    else:
        model.train()

    for key in data[key_set]:
        actual_data, covariate_data, train_label = data['data'][key]
        y_pred = model(actual_data, covariate_data) if train_covariates else model(actual_data)
        y_pred_unqueezed = y_pred.unsqueeze(0)
        loss = criterion(y_pred_unqueezed, train_label)
        total_loss += loss.item()

        # Check if training
        if criterion and optimizer:
            model.zero_grad()
            loss.backward()
            optimizer.step()

        labels.append(train_label)
        _, max_idx = y_pred.max(0)
        predictions.append(max_idx)

    return total_loss, labels, predictions


def evaluate_autoencoder_set(data, key_set: str, autoencoder, criterion, optimizer, use_histogram=False):
    validate_key_set_str(key_set)

    total_loss = 0
    decoded_outputs = {}

    for key in data[key_set]:
        if use_histogram:
            input_seq = data['data'][key][HISTOGRAM_IDX_AFTER_TENSORIFY]
        else:
            input_seq = data['data'][key][0][0].unsqueeze(0)

        decoded_output = autoencoder(input_seq)
        decoded_outputs[key] = decoded_output

        loss = criterion(input_seq, decoded_output)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    return total_loss, decoded_outputs


def evaluate_multitask_learner(data,
                               key_set: str,
                               num_classes,
                               multitask_lerner_model,
                               reconstruction_criterion,
                               classification_criterion,
                               device,
                               optimizer=None,
                               alpha=1,
                               beta=1,
                               use_histogram=False,
                               histogram_seq_len=None,
                               ordinal_regression=False,
                               use_covariates=True):
    validations.validate_data_dict_keys(data)
    validate_key_set_str(key_set)

    total_reconstruction_loss = 0
    total_classification_loss = 0
    total_joint_loss = 0

    labels = []
    predictions = []
    users = []

    if not optimizer:
        multitask_lerner_model.eval()
    else:
        multitask_lerner_model.train()

    outs = list()
    for key in data[key_set]:
        student_id = conversions.extract_student_id_from_key(key)
        student_key = 'student_' + str(student_id)
        actual_data, covariate_data, histogram_data, train_label = data['data'][key]

        if ordinal_regression:
            train_label_vector = get_target_vector_for_ordinal_regression(train_label, num_classes, device)

        actual_data = actual_data[0].unsqueeze(0)
        if use_histogram:
            if histogram_seq_len:
                histogram_data = histogram_data[:max(histogram_seq_len, len(histogram_data))]
            actual_data = histogram_data.unsqueeze(0)

        decoded_output, y_pred = multitask_lerner_model(student_key,
                                                        '-1',
                                                        actual_data,
                                                        covariate_data if use_covariates else None)
        outs.append(y_pred.cpu().detach().numpy())

        # decoded output is `None` if training on only co-variates.
        reconstruction_loss = reconstruction_criterion(actual_data, decoded_output) if decoded_output is not None else object_generator.get_tensor_on_correct_device([0])
        total_reconstruction_loss += reconstruction_loss.item()

        if ordinal_regression:
            classification_loss = classification_criterion(y_pred, train_label_vector)
        else:
            classification_loss = classification_criterion(y_pred, train_label)

        total_classification_loss += classification_loss.item()

        joint_loss = alpha * reconstruction_loss + beta * classification_loss
        total_joint_loss += joint_loss.item()

        # Check if training
        if optimizer:
            multitask_lerner_model.zero_grad()
            joint_loss.backward()
            optimizer.step()

        labels.append(train_label)
        predicted_class = get_predicted_class(y_pred, ordinal_regression=ordinal_regression)
        predictions.append(predicted_class)
        users.append(student_id)

    return total_joint_loss, total_reconstruction_loss, total_classification_loss, labels, predictions, users, outs


def evaluate_multitask_learner_per_user(data,
                               key_set: str,
                               num_classes,
                               multitask_lerner_model_dict,
                               reconstruction_criterion,
                               classification_criterion,
                               device,
                               optimize=False,
                               alpha=1,
                               beta=1,
                               use_histogram=False,
                               histogram_seq_len=None,
                               ordinal_regression=False,
                               use_covariates=True):
    validations.validate_data_dict_keys(data)
    validate_key_set_str(key_set)

    total_reconstruction_loss = 0
    total_classification_loss = 0
    total_joint_loss = 0

    labels = []
    predictions = []
    users = []

    for key in data[key_set]:
        student_id = conversions.extract_student_id_from_key(key)

        multitask_lerner_model, optimizer = multitask_lerner_model_dict[student_id]

        if not optimize:
            multitask_lerner_model.eval()
        else:
            multitask_lerner_model.train()

        student_key = 'student_' + str(student_id)
        actual_data, covariate_data, histogram_data, train_label = data['data'][key]

        if ordinal_regression:
            train_label_vector = get_target_vector_for_ordinal_regression(train_label, num_classes, device)

        actual_data = actual_data[0].unsqueeze(0)
        if use_histogram:
            if histogram_seq_len:
                histogram_data = histogram_data[:max(histogram_seq_len, len(histogram_data))]
            actual_data = histogram_data.unsqueeze(0)

        decoded_output, y_pred = multitask_lerner_model(student_key,
                                                        actual_data,
                                                        covariate_data if use_covariates else None)

        # decoded output is `None` if training on only co-variates.
        reconstruction_loss = reconstruction_criterion(actual_data, decoded_output) if decoded_output is not None else object_generator.get_tensor_on_correct_device([0])
        total_reconstruction_loss += reconstruction_loss.item()

        if ordinal_regression:
            classification_loss = classification_criterion(y_pred, train_label_vector)
        else:
            classification_loss = classification_criterion(y_pred, train_label)

        total_classification_loss += classification_loss.item()

        joint_loss = alpha * reconstruction_loss + beta * classification_loss
        total_joint_loss += joint_loss.item()

        # Check if training
        if optimize:
            multitask_lerner_model.zero_grad()
            joint_loss.backward()
            optimizer.step()

        labels.append(train_label)
        predicted_class = get_predicted_class(y_pred, ordinal_regression=ordinal_regression)
        predictions.append(predicted_class)
        users.append(student_id)

    return total_joint_loss, total_reconstruction_loss, total_classification_loss, labels, predictions, users


def evaluate_multitask_lstm_learner(data,
                               key_set: str,
                               multitask_lerner_model,
                               classification_criterion,
                               optimizer=None,
                               use_histogram=False):
    validations.validate_data_dict_keys(data)
    validate_key_set_str(key_set)

    total_classification_loss = 0

    labels = []
    predictions = []
    users = []

    if not optimizer:
        multitask_lerner_model.eval()
    else:
        multitask_lerner_model.train()

    for key in data[key_set]:
        student_id = conversions.extract_student_id_from_key(key)
        student_key = 'student_' + str(student_id)
        actual_data, covariate_data, histogram_data, train_label = data['data'][key]
        actual_data = actual_data[0].unsqueeze(0)
        if use_histogram:
            actual_data = histogram_data.unsqueeze(0)
        y_pred = multitask_lerner_model(student_key, actual_data, covariate_data)

        classification_loss = classification_criterion(y_pred, train_label)
        total_classification_loss += classification_loss.item()

        # Check if training
        if optimizer:
            multitask_lerner_model.zero_grad()
            classification_loss.backward()
            optimizer.step()

        labels.append(train_label)
        y_pred_squeezed = y_pred.squeeze(0)
        _, max_idx = y_pred_squeezed.max(0)
        predictions.append(max_idx)
        users.append(student_id)

    return total_classification_loss, labels, predictions, users


def is_reconstruction_loss_available(y_pred):
    if isinstance(y_pred, tuple) and len(y_pred) == 2:
        return True
    return False


def get_target_vector_for_ordinal_regression(train_label, num_classes, device):
    label_val = train_label.item() + 1
    new_target_vector = torch.ones(label_val, dtype=torch.float, device=device)

    if new_target_vector.shape[-1] < num_classes:
        zeroes = torch.zeros(num_classes - label_val, dtype=torch.float, device=device)
        new_target_vector = torch.cat([new_target_vector, zeroes], 0)

    return new_target_vector.unsqueeze(0)


def get_predicted_class(y_pred, ordinal_regression=False, or_threshold=0.5):
    y_pred_squeezed = y_pred.squeeze(0)

    if ordinal_regression:
        predicted_class = y_pred_squeezed.ge(or_threshold).sum().int()

    else:
        _, predicted_class = y_pred_squeezed.max(0)

    return predicted_class

