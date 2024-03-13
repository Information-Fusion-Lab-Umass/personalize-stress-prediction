import numpy as np
import pandas as pd

from src import definitions

def transpose_data(data: list):
    np_data_array = np.array(data, dtype=np.float32)
    return np.transpose(np_data_array)

def get_mean_for_series(series, mask):
    assert len(series) == len(mask), "Length mismatch of series: {} and mask: {}".format(
        len(series),
        len(mask))
    return np.mean(series[mask.astype(bool)])


def get_mean_for_series(series, mask):
    return np.mean(series[mask.astype(bool)])

def adjust_classes_wrt_median(label):
    if label < 2:
        return 0
    elif label > 2:
        return 2
    else:
        return 1


def flatten_matrix(matrix):
    """

    @param matrix: Accepts numpy matrix of list to be flattened.
    @return: Flattened list or Matrix.
    """
    assert isinstance(matrix, np.ndarray) or isinstance(matrix,
                                                        list), "Invalid data type, please give either np.ndarray or a lists."

    if isinstance(matrix, np.ndarray):
        return matrix.flatten()
    else:
        return np.array(matrix).flatten().tolist()


def extract_keys_and_labels_from_dict(data: dict):
    keys = []
    labels = []

    for key in data['data']:
        keys.append(key)
        labels.append(data['data'][key][definitions.LABELS_IDX])

    return keys, labels


def extract_student_ids_from_keys(keys):
    student_ids = []
    for key in keys:
        student_ids.append(extract_student_id_from_key(key))

    return student_ids


def extract_distinct_student_idsfrom_keys(keys):
    return set(extract_student_ids_from_keys(keys))


def extract_student_id_from_key(key):
    return key.split("_")[0]


def extract_actual_missing_and_time_delta_from_raw_data_for_student(raw_data, student_id):
    assert len(raw_data) == 3, \
        "Invalid raw data, it missing one of the following: Actual data, Missing flags or Time Deltas"

    (student_data, missing_data, time_delta) = raw_data
    student_data = student_data[student_data['student_id'] == student_id]
    missing_data = missing_data[missing_data['student_id'] == student_id]
    time_delta = time_delta[time_delta['student_id'] == student_id]

    return student_data, missing_data, time_delta


def extract_keys_of_student_from_data(data: dict, student_id):
    keys = []

    for key in data['data']:
        if str(student_id) == extract_student_id_from_key(key):
            keys.append(key)

    return keys


def extract_labels_for_student_id_form_data(data: dict, student_id):
    student_keys = extract_keys_of_student_from_data(data, student_id)
    labels = []

    for key in student_keys:
        labels.append(data['data'][key][definitions.LABELS_IDX])

    return labels


def get_filtered_keys_for_these_students(*student_id, keys):
    filtered_keys = []
    student_ids = list(student_id)

    for key in keys:
        curr_student = key.split("_")[0]
        if curr_student in student_ids:
            filtered_keys.append(key)

    return filtered_keys


def flatten_data(data: list):
    """

    @param data: Data to be flattened, i.e. the rows will be appended as columns.
    @return: Convert sequences to columns by flattening all rows into a single row.
    """
    assert len(data) == 4, "Missing either of the one in data - Actual data, missing flags, time deltas or label"
    flattened_data_list = []
    # Cannot flatten the labels.
    for i in range(len(data) - 1):
        flattened_data_list.append(flatten_matrix(data[i]))
    # Append the label as well.
    flattened_data_list.append(data[-1])

    return flattened_data_list


def convert_df_to_tuple_of_list_values(*data_frames):
    data_frames_as_list = []
    for df in data_frames:
        data_frames_as_list.append(df.values.tolist())

    return tuple(data_frames_as_list)


def get_indices_list_in_another_list(a, b):
    """

    @param a: List of elements who's indices need to be found.
    @param b: Base list containing superset of a.
    @return: indices of elements of list a in list b.
    """
    indices = []
    for element in a:
        indices.append(b.index(element))

    return indices


def drop_duplicate_indices_from_df(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df.index.duplicated(keep="first")]


def convert_to_string_if_int(value):
    return str(value) if isinstance(value, int) else value


def convert_to_int_if_str(value):
    if value.isdigit():
        return int(value)


def convert_list_of_strings_to_list_of_ints(string_list):
    return [convert_to_int_if_str(x) for x in string_list]


def prepend_ids_with_string(ids, string):
    return [string + str(x) for x in ids]


def tensor_list_to_int_list(tensor_list):
    int_list = []
    for t in tensor_list:
        int_list.append(t.item())

    return int_list


def get_model_parameters(model):
    total_model_parameters = sum(p.numel() for p in
                                 model.parameters() if p.requires_grad)
    return total_model_parameters
