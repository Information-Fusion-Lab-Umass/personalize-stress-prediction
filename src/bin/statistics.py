import pandas as pd
import numpy as np

from collections import Counter
from tabulate import tabulate
from src import definitions
from src.bin import validations
from src.bin import user_statistics
from src.utils import data_conversion_utils as conversions
from src.data_manager import student_life_var_binned_data_manager as data_manager

LABEL_COUNT_HEADERS = ['Train', 'Val', 'Test']
USER_TRAIN_STATISTICS_MAP = {
    'confusion_matrix': user_statistics.user_confusion_matrix,
    'f1_score': user_statistics.user_f1_score,
    'accuracy': user_statistics.user_accuracy,
    'label_count': user_statistics.label_count
}


def get_statistics_on_data_dict(data: dict, feature_list: list):
    """
    @attention  Statistics returned are ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'].
    @param data: Data in classic dictionary format.
    @param feature_list: Feature list for the data.
    @return: Statistics on whole data and raw appended data.
    """
    validations.validate_data_dict_keys(data)
    validations.validate_all_data_present_in_data_dict(data)
    df_for_statistics = pd.DataFrame()

    for key in data['data']:
        unit_sequence = data['data'][key][0]
        df_for_statistics = df_for_statistics.append(pd.DataFrame(unit_sequence),
                                                     ignore_index=True)

    if not data_manager.FLATTEN_SEQUENCE_TO_COLS:
        df_for_statistics.columns = feature_list
    df_for_statistics.replace(to_replace=-1, value=np.nan, inplace=True)
    return df_for_statistics.describe(percentiles=[0.25, 0.5, 0.75]), df_for_statistics


def get_train_test_val_label_counts_from_raw_data(data: dict):
    """

    @param data: Data in dictionary format.
    @return: Counts.
    """
    train_counter = get_label_count_in_split(data, 'train')
    val_counter = get_label_count_in_split(data, 'val')
    test_counter = get_label_count_in_split(data, 'test')
    overall_counts = convert_label_counters_to_list(train_counter, val_counter, test_counter)

    return tabulate(overall_counts, headers=LABEL_COUNT_HEADERS)


def get_label_count_in_split(data: dict, split: str = None):
    """

    @param data: Data in dictionary format.
    @param split: Split for which label counts are required.
                  Accepts 'test', 'train' and 'val'.
    @return: Label count for the given split.
    """
    assert split in ['train', 'test', 'val', None]

    labels = []

    if split is None:
        for split_id in data['data']:
            label = data['data'][split_id][definitions.LABELS_IDX]
            labels.append(label)
    else:
        for split_id in data[split + "_ids"]:
            label = data['data'][split_id][definitions.LABELS_IDX]
            labels.append(label)

    counters = Counter(labels)
    return counters


def convert_label_counters_to_list(*counters):
    counters = list(counters)
    overall_counts = []

    def append_count_for_label(label):
        counts_per_element = [label]
        for count in counters:
            counts_per_element.append(count[label])
        overall_counts.append(counts_per_element)

    for label in definitions.LABELS:
        append_count_for_label(label)

    return overall_counts


def tensor_preds_to_scalar_counter(tensor_preds):
    preds_list = []
    for pred in tensor_preds:
        preds_list.append(pred.item())

    return Counter(preds_list)


def get_train_test_val_label_counts_from_predictions(*predictions):
    predictions = list(predictions)
    counters = []
    for preds in predictions:
        counters.append(tensor_preds_to_scalar_counter(preds))
    overall_counts = convert_label_counters_to_list(*counters)

    return tabulate(overall_counts, LABEL_COUNT_HEADERS)


def get_class_weights_in_inverse_proportion(data: dict):
    train_label_counts = get_label_count_in_split(data)
    train_label_counts = [train_label_counts[label] for label in definitions.LABELS]

    # Weight All classes equally if any one class label missing.
    if any([True if x == 0 else False for x in train_label_counts]):
        return [1.0] * len(definitions.LABELS)

    class_weights = [x / max(train_label_counts) for x in train_label_counts]
    class_weights = [1 / x for x in class_weights]
    class_weights = [x / max(class_weights) for x in class_weights]

    return class_weights


def generate_training_statistics_for_user(labels, predictions, users, print_confusion=False):
    """
    Prints the confusion matrix for each student.

    @param labels: The target labels.
    @param predictions: Predictions from the model.
    @param users: The user for the respective label.
    @param print_confusion: If true, it prints the result.
    @return Returns the confusion matrix of each student in a dictionary.
    """

    data_frame_dict = {"user": users,
                       "label": conversions.tensor_list_to_int_list(labels),
                       "prediction": conversions.tensor_list_to_int_list(predictions)}

    distinct_users = set(users)
    statistics_per_user = {}
    labels_predictions_users = pd.DataFrame(data_frame_dict)

    for distinct_user in distinct_users:
        filter_mask = labels_predictions_users['user'] == distinct_user
        user_data = labels_predictions_users[filter_mask]
        statistics = {}

        for statistic in USER_TRAIN_STATISTICS_MAP:
            statistics[statistic] = USER_TRAIN_STATISTICS_MAP[statistic](user_data)

        statistics_per_user[distinct_user] = statistics

    if print_confusion:
        for user in statistics_per_user:
            print(tabulate(statistics_per_user[user]['confusion_matrix']))
            print("---")

    return statistics_per_user


def get_sequence_length_and_num_features_from_data(data: dict, print_output=True):
    first_key = next(iter(data['data'].keys()))
    actual_data = data['data'][first_key][definitions.ACTUAL_DATA_IDX]
    histogram = data['data'][first_key][definitions.HISTOGRAM_IDX]
    covariates = data['data'][first_key][definitions.COVARIATE_DATA_IDX]

    sequence_len, num_features, histogram_seq_len, histogram_num_features = len(actual_data), len(actual_data[0]), len(
        histogram), len(histogram[0])
    covariate_len = len(covariates)

    if print_output:
        print("sequence_len: {} num_features: {} histogram_seq_len: {} histogram_num_feats: {} covariate_len: {}".format(
                sequence_len,
                num_features,
                histogram_seq_len,
                histogram_num_features,
                covariate_len))

    return sequence_len, num_features, histogram_seq_len, histogram_num_features, covariate_len
