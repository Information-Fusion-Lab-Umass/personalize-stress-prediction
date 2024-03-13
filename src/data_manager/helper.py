import pandas as pd

from src import definitions
from src.bin import validations
from src.utils import read_utils
from src.data_processing import helper as processing_helper

VAR_BINNED_DATA_CONFIG = read_utils.read_yaml(definitions.DATA_MANAGER_CONFIG_FILE_PATH)[
    definitions.VAR_BINNED_DATA_MANAGER_ROOT]

TIME_DELTA_BEHIND_FROM_LABEL_H = VAR_BINNED_DATA_CONFIG['time_deltas']['time_delta_behind_from_label_h']
TIME_DELTA_AHEAD_FROM_LABEL_H = VAR_BINNED_DATA_CONFIG['time_deltas']['time_delta_ahead_from_label_h']
TIME_DELTA_BEHIND_FROM_LABEL_H = pd.Timedelta(str(TIME_DELTA_BEHIND_FROM_LABEL_H) + ' hours')
TIME_DELTA_AHEAD_FROM_LABEL_H = pd.Timedelta(str(TIME_DELTA_AHEAD_FROM_LABEL_H) + ' hours')
USE_HISTOGRAM = VAR_BINNED_DATA_CONFIG['use_histogram']
HISTOGRAM_CONFIGS = VAR_BINNED_DATA_CONFIG['histogram']


def get_data_for_single_day(training_values, covariate_values, missing_values,
                            time_delta, y_labels, label_idx):
    """

    @return: Return split for a single day. i.e. One label corresponds to several data points,
             takes in raw data frame and the label for which the split has to be calculated.
    """
    day_string_format = '%Y-%m-%d'
    day_string = label_idx.to_pydatetime().strftime(day_string_format)
    # todo(@abhinavshaw): For covariates adjust wth delta as you can carry the original label here.
    return (training_values.loc[day_string, :].values.tolist(),
            missing_values.loc[day_string, :].values.tolist(),
            time_delta.loc[day_string, :].values.tolist(),
            covariate_values.loc[label_idx, :].values.tolist(),
            y_labels.loc[label_idx].values.tolist()[0])


def get_data_for_single_label_based_on_time_delta(training_values, covariate_values, missing_values,
                                                  time_delta, y_labels, label_idx):
    time_indices_to_keep = pd.date_range(label_idx - TIME_DELTA_BEHIND_FROM_LABEL_H,
                                         label_idx + TIME_DELTA_AHEAD_FROM_LABEL_H,
                                         freq=definitions.DEFAULT_BASE_FREQ,
                                         closed="left")

    # No-op if enough data is not available.
    if not validations.check_if_enough_indices_in_data_frame(training_values,
                                                             time_indices_to_keep):
        return

    training_values = training_values.reindex(time_indices_to_keep)
    missing_values = missing_values.reindex(time_indices_to_keep)
    time_delta = time_delta.reindex(time_indices_to_keep)
    delta = training_values.index[1] - training_values.index[0]
    histogram_values = get_histogram(training_values)

    return (training_values.values.tolist(),
            missing_values.values.tolist(),
            time_delta.values.tolist(),
            covariate_values.loc[label_idx - delta, :].values.tolist(),
            histogram_values.values.tolist(),
            y_labels.loc[label_idx].values.tolist()[0])


def get_histogram(training_values: pd.DataFrame) -> pd.DataFrame:
    resampler = training_values.resample(rule="60T")
    rule = {}

    for feature in training_values.columns:
        # Skip if not in config. A simple way to controll what feature to process.
        if feature not in HISTOGRAM_CONFIGS.keys():
            continue

        feature_rule = processing_helper.get_aggregation_rule_for_histogram(feature,
                                                                            HISTOGRAM_CONFIGS[feature])

        for key in feature_rule:
            rule[key] = feature_rule[key]

    resampled_data = resampler.agg(rule)
    resampled_data.columns = ['_'.join(col).strip() if 'student_id' not in col else 'student_id'
                              for col in resampled_data.columns.values]

    return resampled_data
