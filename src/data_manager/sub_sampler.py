import numpy as np

from src import definitions
from src.bin import validations
from src.utils import read_utils
from src.utils import object_generator_utils

HOURS_TO_MINUTES = 60
DATA_MANAGER_CONFIG = read_utils.read_yaml(definitions.DATA_MANAGER_CONFIG_FILE_PATH)[
    definitions.VAR_BINNED_DATA_MANAGER_ROOT]
SUB_SAMPLING_CONFIG = DATA_MANAGER_CONFIG['sub_sampling']
SUB_SAMPLE_COUNT = SUB_SAMPLING_CONFIG['sub_sample_count']
BASE_FREQ = int(definitions.DEFAULT_BASE_FREQ.split(" ")[0])
SEQUENCE_LEN = HOURS_TO_MINUTES * DATA_MANAGER_CONFIG['time_deltas']['time_delta_behind_from_label_h'] / BASE_FREQ
OUT_SEQUENCE_LEN = SUB_SAMPLING_CONFIG['output_sequence_len']


def find_set_for_key(data: dict, key):
    """

    @param data:
    @return:  set where the key exists.
    """

    if key in data['train_ids']:
        return 'train_ids'
    elif key in data['val_ids']:
        return 'val_ids'
    elif key in data['test_ids']:
        return 'test_ids'
    else:
        return None


def get_sub_sampled_sequences(data: dict):
    validations.validate_all_data_present_in_data_dict(data)
    validations.validate_data_dict_keys(data)
    new_data = object_generator_utils.get_empty_data_dict()

    for key in data['data']:
        key_set = find_set_for_key(data, key)
        if key_set:
            sub_sample_sequences(data['data'][key], key, key_set, new_data)

    return new_data


def sub_sample_sequences(data_tuple, key, key_set, new_data):
    """

    @param data_tuple:
    @param new_data: Modifies this dictionary in place.
    """
    end_range = SEQUENCE_LEN - OUT_SEQUENCE_LEN - 1
    sampled_seq_start_indices = np.random.randint(0, end_range, size=SUB_SAMPLE_COUNT)

    actual = data_tuple[definitions.ACTUAL_DATA_IDX]
    missing = data_tuple[definitions.MISSING_FLAGS_IDX]
    time_delta = data_tuple[definitions.TIME_DELTA_IDX]
    covariates = data_tuple[definitions.COVARIATE_DATA_IDX]
    y_label = data_tuple[definitions.LABELS_IDX]

    for idx, start_idx in enumerate(sampled_seq_start_indices):
        sub_actual, sub_missing, sub_time_delta = slice_sequence(actual,
                                                                 missing,
                                                                 time_delta,
                                                                 start_idx=start_idx,
                                                                 output_seq_length=OUT_SEQUENCE_LEN)
        new_key = key + "_" + str(idx)
        new_data['data'][new_key] = sub_actual, sub_missing, sub_time_delta, covariates, y_label
        new_data[key_set].append(new_key)


def slice_sequence(*data: list, start_idx, output_seq_length):
    validate_if_data_slice_out_of_bound(*data,
                                        start_idx=start_idx,
                                        output_seq_length=output_seq_length)
    out_data = []
    for datum in list(data):
        out_data.append(datum[start_idx:start_idx + output_seq_length])

    return tuple(out_data)


def validate_if_data_slice_out_of_bound(*data: list, start_idx, output_seq_length):
    assert start_idx >= 0, "Start idx cannot be negative!"
    for datum in list(data):

        assert len(datum) >= start_idx + output_seq_length, "Sequence length out of bound in list."
