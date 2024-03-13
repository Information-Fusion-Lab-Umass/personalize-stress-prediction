"""
Script to generate binned aggregates based on the configuration per feature.
"""
import os
import pandas as pd
import numpy as np

from src import definitions
from src.data_processing import helper
from src.utils import student_utils
from src.utils import read_utils
from src.utils.write_utils import df_to_csv
from src.utils import data_conversion_utils as conversions

# Reading Configs.
FEATURE_CONFIG = read_utils.read_yaml(definitions.FEATURE_CONFIG_FILE_PATH)['explode_duration_based_features']
AVAILABLE_FEATURE = FEATURE_CONFIG.keys()
STUDENT_CONFIG = read_utils.read_yaml(definitions.FEATURE_CONFIG_FILE_PATH)['students']
AVAILABLE_STUDENTS = student_utils.get_available_students(definitions.MINIMAL_PROCESSED_DATA_PATH)
DEFAULT_RESAMPLING_AGGREGATE_CONFIG = {
    "resample_freq_min": 1,
    "simple_aggregates": [],
    "custom_aggregates": ['robust_sum']
}
DEFAULT_INFERENCE_VALUE_WHEN_INFERRED = 1


def explode_values(feature_values: pd.Series, feature_name):
    """

    @return: These Values are going to explode.
    """

    final_exploded_df = pd.DataFrame()
    for index in feature_values.index:
        start_date = index
        value = feature_values.loc[index].item()
        end_date = start_date + pd.Timedelta(str(value) + " min")
        exploded_df_index = pd.date_range(start_date,
                                          end_date,
                                          freq=definitions.DEFAULT_EXPLODING_BASE_FREQ,
                                          closed="left")
        exploded_df = pd.DataFrame(np.full(len(exploded_df_index),
                                           DEFAULT_INFERENCE_VALUE_WHEN_INFERRED),
                                   index=exploded_df_index,
                                   columns=[feature_name])
        final_exploded_df = final_exploded_df.append(exploded_df)
        final_exploded_df = conversions.drop_duplicate_indices_from_df(final_exploded_df)

    return final_exploded_df


def explode_feature_data(feature_data: pd.DataFrame):
    resampled_feature_data = helper.get_resampled_aggregated_data(feature_data,
                                                                  DEFAULT_RESAMPLING_AGGREGATE_CONFIG,
                                                                  student_id)
    feature_cols = helper.get_feature_cols_from_data(resampled_feature_data)

    for col in feature_cols:
        not_null_mask = resampled_feature_data[col].notnull()
        values_to_explode = resampled_feature_data[col][not_null_mask]
        exploded_feature_values = explode_values(values_to_explode, col)
        overlapping_indices = resampled_feature_data[col].index.intersection(exploded_feature_values.index)
        resampled_feature_data[col].loc[overlapping_indices] = exploded_feature_values[col].loc[overlapping_indices]
        resampled_feature_data[col].fillna(value=0, inplace=True)
        # Todo (abhinavshaw): see how to handle appending the last duration sequence.
    return resampled_feature_data


def rename_postfix_from_cols(data_frame: pd.DataFrame):
    def remove_suffix(col):
        col = col.replace("_robust_sum", "_inferred")
        return col

    return data_frame.rename(remove_suffix, axis="columns")


############## Main Loop To Process Data ##################
for student_id in AVAILABLE_STUDENTS:

    student_data = []

    for idx, feature in enumerate(AVAILABLE_FEATURE):
        feature_data_path = os.path.join(definitions.MINIMAL_PROCESSED_DATA_PATH,
                                         definitions.STUDENT_FOLDER_NAME_PREFIX + str(student_id),
                                         feature + ".csv")
        feature_data = pd.read_csv(feature_data_path, index_col=[0])
        feature_data.index = pd.to_datetime(feature_data.index)
        exploded_feature_data = explode_feature_data(feature_data)
        exploded_feature_data = rename_postfix_from_cols(exploded_feature_data)
        exploded_feature_path_to_folder = os.path.join(definitions.MINIMAL_PROCESSED_DATA_PATH,
                                                       definitions.STUDENT_FOLDER_NAME_PREFIX + str(student_id))
        exploded_feature_filename = feature + "_inferred.csv"
        df_to_csv(exploded_feature_data, exploded_feature_filename, exploded_feature_path_to_folder)

    print("Feature exploded for student: {}".format(student_id))
