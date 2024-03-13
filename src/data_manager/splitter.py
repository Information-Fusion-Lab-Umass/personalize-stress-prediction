from datetime import datetime

DEFAULT_STUDENT_LIFE_YEAR = 2013


def split_data_by_percentage(data_list, start_index: int = 0, percent: float = -1):
    """

    @param data_list: The data for which slice is required.
    @param start_index: all indices before this are not considered for slicing.
    @param percent: Percentage of data that contributes to the slice. If percent = -1,
           then everything from start_index to len(data) is returned.
    @return:
    """
    data_len = len(data_list)
    slice_length = round(data_len * percent / 100)

    assert 0 < percent <= 100 or percent == -1, "Percent value must be between 1 and 100 but got {}".format(percent)
    assert 0 <= start_index < data_len
    assert start_index + slice_length < data_len, "Over flow of data list. " \
                                                  "Enter smaller percent value or reduce the start_index."

    if percent == -1:
        data_slice = data_list[start_index:]
        data_slice_keys = [month_day_hour_key for month_day_hour_key, data in data_slice]
        end_index = data_len - 1
    else:
        data_slice = data_list[start_index: start_index + slice_length]
        data_slice_keys = [month_day_hour_key for month_day_hour_key, data in data_slice]
        end_index = start_index + slice_length

    return data_slice_keys, end_index


def split_data_by_date_range(data_list, start_date: str=None, end_date: str=None):
    """
    @attention end_date is not included in the slice.
    @param data_list: Data list for which a slice is required.
    @param start_date: Start date of the slice.
    @param end_date: End date of the slice.
    @return: sliced data_list.
    """

    sliced_data_key_list = []

    if start_date is None:
        date_key, data = data_list[0]
        start_date = datetime_key_to_date(date_key)
    else:
        start_date = datetime_key_to_date(start_date)

    if end_date is None:
        date_key, data = data_list[-1]
        end_date = datetime_key_to_date(date_key)
    else:
        end_date = datetime_key_to_date(end_date)

    for date_key, data in data_list:

        cur_date = datetime_key_to_date(date_key)

        if start_date <= cur_date < end_date:
            sliced_data_key_list.append(date_key)

    return sliced_data_key_list


def datetime_key_to_date(date_key):
    month, day, hour = tuple(map(int, date_key.split("_")))
    return datetime(year=2013, month=month, day=day, hour=hour)


def get_data_split_by_percentage(data_list):
    # Splitting data into Train, Val  and Test Split.
    train_set, end_idx = split_data_by_percentage(data_list, start_index=0, percent=25)
    val_set, end_idx = split_data_by_percentage(data_list, start_index=end_idx, percent=15)
    test_set, end_idx = split_data_by_percentage(data_list, start_index=end_idx, percent=1)

    train_set_2, end_idx = split_data_by_percentage(data_list, start_index=end_idx, percent=25)
    val_set_2, end_idx = split_data_by_percentage(data_list, start_index=end_idx, percent=15)
    test_set_2, end_idx = split_data_by_percentage(data_list, start_index=end_idx, percent=1)

    train_set_3, end_idx = split_data_by_percentage(data_list, start_index=end_idx, percent=10)
    val_set_3, end_idx = split_data_by_percentage(data_list, start_index=end_idx, percent=1)
    test_set_3, end_idx = split_data_by_percentage(data_list, start_index=end_idx, percent=-1)

    train_set = train_set + train_set_2 + train_set_3
    val_set = val_set + val_set_2 + val_set_3
    test_set = test_set + test_set_2 + test_set_3

    return train_set, val_set, test_set


def get_data_split_by_date(data_list):
    # Before midterm
    train_set = split_data_by_date_range(data_list, start_date=None, end_date='04_10_0')
    val_set = split_data_by_date_range(data_list, start_date='04_10_0', end_date='04_16_0')
    test_set = split_data_by_date_range(data_list, start_date='04_16_0', end_date='04_17_0')

    # During midterm.
    train_set_2 = split_data_by_date_range(data_list, start_date='04_17_0', end_date='04_27_0')
    val_set_2 = split_data_by_date_range(data_list, start_date='04_27_0', end_date='04_30_0')
    test_set_2 = split_data_by_date_range(data_list, start_date='04_30_0', end_date='05_2_0')

    # after midterm
    train_set_3 = split_data_by_date_range(data_list, start_date='05_2_0', end_date='05_11_0')
    val_set_3 = split_data_by_date_range(data_list, start_date='05_11_0', end_date='05_13_0')
    test_set_3 = split_data_by_date_range(data_list, start_date='05_16_0', end_date=None)

    train_set = train_set + train_set_2 + train_set_3
    val_set = val_set + val_set_2 + val_set_3
    test_set = test_set + test_set_2 + test_set_3

    return train_set, val_set, test_set
