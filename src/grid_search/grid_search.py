import itertools
import torch
import copy
import os
import tqdm

from src import definitions
from sklearn import metrics
from src.bin import tensorify
from src.bin import plotting
from src.data_manager import cross_val
from src.grid_search import helper
from src.utils import data_conversion_utils as conversions
from src.utils import read_utils
from src.utils import write_utils
from statistics import mean as list_mean
from src.bin import statistics

F_SCORE_INDEX = 2

TRAINING_DATA_FILE_NAME = read_utils.read_yaml(definitions.GRID_SEARCH_CONFIG_FILE_PATH)['data_file_name']


def get_hyper_parameter_list_for_grid_search(experiment="multitask_learner_auto_encoder"):
    experiment_config = read_utils.read_yaml(definitions.GRID_SEARCH_CONFIG_FILE_PATH)[experiment]
    hyper_parameter_list = []
    params = experiment_config.keys()

    for param in params:
        hyper_parameter_list.append(experiment_config[param])

    hyper_parameters_list = list(itertools.product(*hyper_parameter_list))
    final_hyper_prameters_list = []

    for hyper_parameters in hyper_parameters_list:
        hyper_parameters_dict = {}
        for idx, param in enumerate(params):
            hyper_parameters_dict[param] = hyper_parameters[idx]

        final_hyper_prameters_list.append(hyper_parameters_dict)

    return final_hyper_prameters_list


def search_best_params_for_experiment(experiment, data: dict):
    if experiment == "multitask_learner_auto_encoder":
        search_multitask_auto_encoder(get_hyper_parameter_list_for_grid_search(experiment), data)


def search_multitask_auto_encoder(hyper_parameters_list, data: dict):
    splits = cross_val.get_k_fod_cross_val_splits_stratified_by_students(data)
    student_list = conversions.extract_distinct_student_idsfrom_keys(data['data'].keys())
    tensorified_data = tensorify.tensorify_data_gru_d(copy.deepcopy(data), torch.cuda.is_available())

    final_scores_for_each_config = []

    print("Label Distribution")
    print(statistics.get_train_test_val_label_counts_from_raw_data(data))

    for model_params_no, model_params in enumerate(hyper_parameters_list):
        print("###################### Param Config No: {} ########################".format(model_params_no))
        print("Params: ", model_params)

        (use_histogram,
         autoencoder_bottle_neck_feature_size,
         autoencoder_num_layers,
         alpha, beta,
         decay,
         num_features,
         num_covariates,
         shared_hidden_layer_size,
         user_dense_layer_hidden_size,
         num_classes,
         learning_rate,
         n_epochs,
         shared_layer_dropout_prob,
         user_head_dropout_prob,
         class_weights,
         device) = helper.get_params_from_model(model_params, data)

        best_val_scores = []

        for split_no, split in enumerate(splits):

            print("Split {}".format(split_no))

            best_split_score = -1

            tensorified_data['train_ids'] = split["train_ids"]
            tensorified_data['val_ids'] = split["val_ids"]
            tensorified_data['test_ids'] = []

            model, reconstruction_criterion, classification_criterion, optimizer = helper.init_multitask_autoencoder_learner(
                num_features,
                autoencoder_bottle_neck_feature_size,
                autoencoder_num_layers,
                shared_hidden_layer_size,
                user_dense_layer_hidden_size,
                num_classes,
                num_covariates,
                shared_layer_dropout_prob,
                user_head_dropout_prob,
                learning_rate,
                decay,
                class_weights,
                student_list)

            total_loss_over_epochs, scores_over_epochs = plotting.get_empty_stat_over_n_epoch_dictionaries()
            reconstruction_loss_over_epochs = copy.deepcopy(total_loss_over_epochs)
            classification_loss_over_epochs = copy.deepcopy(total_loss_over_epochs)

            for epoch in tqdm.tqdm(range(n_epochs)):

                (train_total_loss,
                 train_total_reconstruction_loss,
                 train_total_classification_loss,
                 train_labels,
                 train_preds,
                 train_users), (val_total_loss,
                                val_total_reconstruction_loss,
                                val_total_classification_loss,
                                val_labels,
                                val_preds,
                                val_users) = helper.train_for_one_epoch(tensorified_data,
                                                                        num_classes,
                                                                        model,
                                                                        reconstruction_criterion,
                                                                        classification_criterion,
                                                                        device,
                                                                        optimizer,
                                                                        alpha,
                                                                        beta,
                                                                        use_histogram)

                ######## Appending losses ########
                total_loss_over_epochs['train_loss'].append(train_total_loss)
                total_loss_over_epochs['val_loss'].append(val_total_loss)

                reconstruction_loss_over_epochs['train_loss'].append(train_total_reconstruction_loss)
                reconstruction_loss_over_epochs['val_loss'].append(val_total_reconstruction_loss)

                classification_loss_over_epochs['train_loss'].append(train_total_classification_loss)
                classification_loss_over_epochs['val_loss'].append(val_total_classification_loss)

                ######## Appending Metrics ########
                train_label_list = conversions.tensor_list_to_int_list(train_labels)
                train_pred_list = conversions.tensor_list_to_int_list(train_preds)
                val_label_list = conversions.tensor_list_to_int_list(val_labels)
                val_pred_list = conversions.tensor_list_to_int_list(val_preds)

                train_scores = metrics.precision_recall_fscore_support(train_label_list,
                                                                       train_pred_list,
                                                                       average='weighted')[F_SCORE_INDEX]
                val_scores = metrics.precision_recall_fscore_support(val_label_list,
                                                                     val_pred_list,
                                                                     average='weighted')[F_SCORE_INDEX]

                scores_over_epochs['train_scores'].append(train_scores)
                scores_over_epochs['val_scores'].append(val_scores)

                if val_scores > best_split_score:
                    best_split_score = val_scores

            best_val_scores.append(best_split_score)

        avg_val_score = list_mean(best_val_scores)
        final_scores_for_each_config.append((avg_val_score, model_params))

        print("Average score for current configuration: {}".format(avg_val_score))

    grid_search_details_file_path = os.path.join(definitions.DATA_DIR, "grid_search_details.pkl")
    write_utils.data_structure_to_pickle(final_scores_for_each_config, grid_search_details_file_path)


def run_grid_search():
    data_file_path = os.path.join(definitions.DATA_DIR, 'training_data/shuffled_splits', TRAINING_DATA_FILE_NAME)
    data = read_utils.read_pickle(data_file_path)
    search_best_params_for_experiment("multitask_learner_auto_encoder", data)


run_grid_search()
