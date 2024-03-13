"""
Stores all validations required by the Lib.
"""
import pandas as pd

from src import definitions

DATA_DICT_KEYS = ['data', 'train_ids', 'test_ids', 'val_ids']


def check_if_enough_indices_in_data_frame(training_vales: pd.DataFrame, time_indices_to_keep):
    """

    @brief: Checks if the data frame has the indices required.
            This is done by intersection operation of the indices.
    @return: True, if enough data available.
    """
    required_len = len(time_indices_to_keep)
    intersection_len = len(training_vales.index.intersection(time_indices_to_keep))

    return required_len == intersection_len


def check_if_all_columns_present_in_df(df: pd.DataFrame, columns: list):
    return all(col in df.columns for col in columns)


def check_if_element_in_list(element, src_list: list):
    assert element in src_list, "Element: {} not present is the given list".format(element)


def check_if_key_present_in_dict(key, src_dict: dict):
    check_if_element_in_list(key, list(src_dict.keys()))


def validate_student_id_in_data(*data: pd.DataFrame):
    for df in data:
        assert "student_id" in df.columns, "Invalid data. missing column 'student_id'."


def validate_config_key(*keys: str, config):
    for key in keys:
        assert key in config, "Invalid config!. Key: {} not present in config.".format(key)


def validate_student_id_in_data_as_first_col(*data: pd.DataFrame):
    validate_student_id_in_data(*data)

    for df in data:
        assert "student_id" == df.columns[0], "First Column in DataFrame is not 'student_id'."


def validate_data_integrity_for_len(*data_frame: pd.DataFrame):
    data_frames = list(data_frame)

    for df in data_frames[1:]:
        assert len(df) == len(data_frame[0]), "Lengths of the DataFrame do not match."


def validate_data_dict_keys(data_dict):
    assert all([k in DATA_DICT_KEYS for k in data_dict.keys()])


def validate_all_data_present_in_data_dict_for_key(data_dict: dict, key):
    validate_data_dict_keys(data_dict)
    first_key = next(iter(data_dict['data'].keys()))
    assert len(data_dict['data'][first_key]) == definitions.DATA_TUPLE_LEN, \
        "Data Tuple len mismatch. Expected: {} Found: {}. If found less than expected, one of these could be missing -'Actual Data', 'Covariate','Missing Flags', 'Time Deltas', 'Label'".format(
            definitions.DATA_TUPLE_LEN,
            len(data_dict['data'][first_key]))


def validate_all_data_present_in_data_dict(data_dict: dict):
    for key in data_dict['data']:
        validate_all_data_present_in_data_dict_for_key(data_dict, key)


def validate_no_nans_in_tensor(tensor):
    assert not (tensor != tensor).any(), "null exists in input!"


def validate_all_columns_present_in_data_frame(*data_frames: pd.DataFrame, columns: list):
    for df in list(data_frames):
        assert len(df.columns) >= len(columns), "More columns requested than available in data frame."
        assert all([column in df.columns for column in columns]
                   ), "These columns missing in data frame: {}".format(
            [col if col not in df.columns else None for col in columns])


def validate_integrity_of_covariates(covariates, covariate_data):
    assert (covariates == 0 and covariate_data is None
            ) or (covariates > 0 and covariate_data is not None
                  ), "Mismatch in covariate initialization and covariate data."


def validate_single_values_column_in_df(df: pd.DataFrame, column):
    """
    Validate if the passed column of the data frame just has one column.
    """

    assert len(df[column].value_counts()) == 1, "Column: {} has multiple value. This must have only one value.".format(column)


def validate_user_data(user_data):
    assert check_if_all_columns_present_in_df(user_data, ['label', 'user', 'prediction'])


def validate_sequential_model_size_parameters(*size_params):
    size_params = list(size_params)
    for i in range(2, len(size_params), 2):
        previous_size = size_params[i - 1]
        cur_size = size_params[i]

        assert previous_size == cur_size, "Miss-match in size parameters of the network."
