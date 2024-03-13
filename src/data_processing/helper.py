"""
Helper function for data processing.
"""

import pandas as pd
import numpy as np

from src import definitions
from src.utils import read_utils
from src.bin import validations as validations
from src.data_processing import aggregates
from src.data_processing import covariates as covariate_processor
from src.data_processing import imputation

FEATURE_IMPUTATION_STRATEGY = read_utils.read_yaml(definitions.FEATURE_CONFIG_FILE_PATH)[
    'feature_imputation_strategy']

COVARIATE_FUNC_MAPPING = {
    'day_of_week': covariate_processor.day_of_week,
    'epoch_of_day': covariate_processor.epoch_of_day,
    'time_since_last_label': covariate_processor.time_since_last_label_min,
    'time_to_next_label': covariate_processor.time_to_next_label_min,
    'gender': covariate_processor.evaluate_gender,
    'previous_stress_label': covariate_processor.previous_stress_label,
    'time_to_next_deadline': covariate_processor.time_to_next_deadline
}

AGGREGATE_FUNC_MAPPING = {
    'mode': aggregates.mode,
    'inferred_feature': aggregates.inferred_feature,
    'robust_sum': aggregates.robust_sum,
    'time': aggregates.time_group,
    "0": aggregates.count_0,
    "1": aggregates.count_1,
    "2": aggregates.count_2,
    "3": aggregates.count_3

}

INTERPOLATION_FUNC_MAPPING = {
    'linear': imputation.linear_interpolation,
    'forward_fill': imputation.forward_fill,
    'mean_fill': imputation.mean_fill,
    'mode_fill': imputation.mode_fill,
    'none': None
}


def get_aggregation_rule(feature_inference_cols, feature_config, student_id):
    """

    @param feature_inference_cols:
    @param feature_config:
    @return: Return Aggregation Rule for the feature based on the configuration.
    """
    def value(array_like):
        return student_id

    # List of custom aggregate function.
    custom_aggregates = []
    simple_aggregates = feature_config['simple_aggregates']

    for custom_aggregate in feature_config['custom_aggregates']:
        custom_aggregates.append(AGGREGATE_FUNC_MAPPING[custom_aggregate])

    rule = {"student_id": value}

    for col in feature_inference_cols:
        rule[col] = simple_aggregates + custom_aggregates

    return rule


def get_aggregation_rule_for_histogram(feature_name, feature_config):
    simple_aggregate = feature_config['simple_aggregates']
    custom_aggregate = []

    for agg in feature_config['custom_aggregates']:
        custom_aggregate.append(AGGREGATE_FUNC_MAPPING[agg])

    rule = {feature_name: simple_aggregate + custom_aggregate}

    return rule


def get_resampled_aggregated_data(feature_data: pd.DataFrame, feature_config, student_id) -> pd.DataFrame:
    """

    @param feature_data: Un-resampled data for the feature.
    @param feature_config: Configs for the specific feature.
    @return: Aggregated data on the resampled frequency.
    """
    validations.validate_config_key('resample_freq_min', config=feature_config)

    # Extracting columns other than student id (These are the feature inference columns)
    feature_inference_cols = list(feature_data.columns.values)
    feature_inference_cols.remove("student_id")
    # Resampling and applying aggregate rule.
    resample_freq_min = feature_config[definitions.RESAMPLE_FREQ_CONFIG_KEY]
    resampled_feature_data = feature_data.resample(rule=str(resample_freq_min) + "T")
    aggregation_rule = get_aggregation_rule(feature_inference_cols, feature_config, student_id)
    aggregated_data = resampled_feature_data.agg(aggregation_rule)

    # Flattening all the columns.
    aggregated_data.columns = ['_'.join(col).strip() if 'student_id' not in col else 'student_id'
                               for col in aggregated_data.columns.values]

    return aggregated_data


def get_flattened_student_data_from_list(student_data: pd.DataFrame, student_id) -> pd.DataFrame:
    """

    @param student_data: A list of data frame with various features from the student_life data-set.
    @param student_id: Student id of the student.
    @return: flattened data-set after applying a left join.
    """
    validations.validate_student_id_in_data(*student_data)

    # Pre-processing
    feature_data_first = student_data[0]
    start_date = feature_data_first.index[0].floor("D")
    # todo(abhinavshaw): add one more day to end date to give some room while data processing. Verify end to end.
    end_date = feature_data_first.index[-1].floor("D")
    flattened_df_index = pd.date_range(start_date, end_date, freq=definitions.DEFAULT_BASE_FREQ)
    flattened_df = pd.DataFrame(np.full(len(flattened_df_index), student_id),
                                index=flattened_df_index,
                                columns=["student_id"])

    for idx, feature_df in enumerate(student_data):
        feature_df_dropped_student_id = feature_df.drop("student_id", axis=1, inplace=False)
        flattened_df = flattened_df.join(feature_df_dropped_student_id, how='left', sort=True)

    return flattened_df


def impute_missing_feature(flattened_student_data: pd.DataFrame) -> pd.DataFrame:
    # TODO(abhinavshaw): allow multiple sequential imputation for features and clean up this code.
    if FEATURE_IMPUTATION_STRATEGY['impute_features']:
        for feature_col in flattened_student_data.columns:
            propagation_types = FEATURE_IMPUTATION_STRATEGY[feature_col]
            if len(propagation_types) > 0:
                for propagation_type in propagation_types:
                    flattened_student_data[feature_col] = INTERPOLATION_FUNC_MAPPING[propagation_type](
                        flattened_student_data[feature_col])
                    flattened_student_data[feature_col] = flattened_student_data[feature_col].round(decimals=0)

    return flattened_student_data


def replace_neg_one_with_nan(df):
    """

    @param df: DataFrame to be processed.
    @return: Replaces -1(int) or -1.0(double) all rows to np.nan.
    """
    # Converting any missing values to NaN.
    return df.replace(to_replace={-1: np.nan, -1.0: np.nan}, value=None, inplace=False)


def remove_days_with_no_stress_label(flattened_student_data: pd.DataFrame) -> pd.DataFrame:
    """

    @param flattened_student_data: Flattened data of student. Must contain stress_level_mode as
                                   one of the columns.
    @return: processed data frame where sequences belonging to the same day are removed where there
            are no stress label.
    """

    validations.validate_student_id_in_data(flattened_student_data)

    stress_not_null_df = flattened_student_data[flattened_student_data['stress_level_mode'].notnull()]
    stress_not_null_indices = stress_not_null_df.index
    td = pd.Timedelta('1 days')

    for idx, time_index in enumerate(stress_not_null_indices):
        floored_time_index = time_index.floor("D")
        if idx == 0:
            time_indices_to_keep = pd.date_range(floored_time_index,
                                                 floored_time_index + td,
                                                 freq=definitions.DEFAULT_BASE_FREQ,
                                                 closed="left")
        else:
            time_indices_to_keep = time_indices_to_keep.union(
                pd.date_range(floored_time_index,
                              floored_time_index + td,
                              freq=definitions.DEFAULT_BASE_FREQ,
                              closed="left"))

    indices_to_be_dropped = flattened_student_data.index.difference(time_indices_to_keep)
    flattened_student_data_dropped = flattened_student_data.drop(indices_to_be_dropped)

    return flattened_student_data_dropped


def get_time_deltas_min(flattened_student_data: pd.DataFrame) -> pd.DataFrame:
    """
    @attention: Doesnt calculates time deltas for the student_id column as that is an identifier.
    @param flattened_student_data:
    @return: Returns time deltas of the last observed data in a DataFrame.
    """
    validations.validate_student_id_in_data(flattened_student_data)

    time_deltas = pd.DataFrame(index=flattened_student_data.index,
                               columns=flattened_student_data.columns,
                               dtype=float)
    last_observed_time = {}
    for col in flattened_student_data.columns:
        last_observed_time[col] = flattened_student_data.index[0]

    cols = flattened_student_data.columns
    rows = len(flattened_student_data)

    for i in range(0, rows):
        for col_idx, col in enumerate(cols):

            is_col_nan = np.isnan(flattened_student_data.iloc[i][col])
            if not is_col_nan and col != "student_id":
                last_observed_time[col] = flattened_student_data.index[i]

            delta = time_deltas.index[i] - last_observed_time[col]

            # converting to minutes if the col is not student_id.
            time_deltas.iloc[i, col_idx] = \
                flattened_student_data.iat[i, col_idx] \
                    if col == "student_id" else delta.total_seconds() / 60

    return time_deltas


def get_missing_data_mask(flattened_student_data: pd.DataFrame) -> pd.DataFrame:
    """
    @attention: will not calculate missing flag for the student id column. Since that is an identifier
                and not a feature or a label.
    @param flattened_student_data:
    @return: Return and integer data frame with value = 0 where data is missing else value = 1.
    """
    validations.validate_student_id_in_data_as_first_col(flattened_student_data)

    # Calculate masks on all but the "student_id" col.
    missing_value_mask = flattened_student_data.copy()
    missing_value_mask.iloc[:, 1:] = flattened_student_data.iloc[:, 1:].isnull().astype(int)

    return missing_value_mask


def process_covariates(flattened_student_data: pd.DataFrame, covariates: dict) -> pd.DataFrame:
    """

    @param flattened_student_data:
    @param covariates: Dictionary of covariates and their boolean flags.
    @return: Data frame after processing covariates.
    """

    for covariate, bool_flag in covariates.items():
        if bool_flag:
            processed_flattened_student_data = COVARIATE_FUNC_MAPPING[covariate](flattened_student_data)
            flattened_student_data = processed_flattened_student_data if \
                processed_flattened_student_data is not None else flattened_student_data

    return flattened_student_data


def get_feature_cols_from_data(feature_data: pd.DataFrame):
    """

    @return: The columns that contain actual feature. This ignores student_id as a column,
             as that does not contain any feature values.
    """

    feature_cols = list(feature_data.columns.values)
    feature_cols.remove("student_id")

    return feature_cols
