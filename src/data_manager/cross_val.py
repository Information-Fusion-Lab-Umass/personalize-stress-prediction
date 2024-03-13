import numpy as np

from random import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from src.utils import data_conversion_utils as conversions


SPLITTER_RANDOM_STATE = 100


def random_stratified_splits(data: dict, stratify_by="labels"):
    """

    @param data: Data in classic dict format.
    @param stratify_by: By what the splits need to be stratified.
           Accepts - `labels` and `students`.
    @return: Return splits which are randomize and stratified by labels.
    """
    keys, labels = conversions.extract_keys_and_labels_from_dict(data)
    keys, labels = np.array(keys), np.array(labels)

    if stratify_by == "students":
        student_ids = conversions.extract_student_ids_from_keys(keys)
        stratification_array = np.array(student_ids)
    else:
        stratification_array = labels

    (X_train, X_test,
     y_train, y_test,
     stratification_array, new_stratification_array) = train_test_split(keys,
                                                                        labels,
                                                                        stratification_array,
                                                                        test_size=0.40,
                                                                        shuffle=True,
                                                                        stratify=stratification_array)
    X_val, X_test = train_test_split(X_test,
                                     test_size=0.40,
                                     shuffle=True,
                                     stratify=new_stratification_array)

    return X_train.tolist(), X_val.tolist(), X_test.tolist()

# helper function, extract first n weeks/days data:
def get_first_n_data(keys, n):
    month_days = {0: 0, 1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31} # map: month -> # days

    begin_interval = n # how many data used for the leaved out student
    start_day = -1

    key_list = list()
    val_keys = list()
    for key in keys:
        month = int(key.split('_')[1])
        day = int(key.split('_')[2])
        curr_day = sum([month_days[i] for i in range(month + 1)]) + day
        if start_day < 0:
            start_day = curr_day
        else:
            if curr_day - start_day >= begin_interval:
                val_keys.append(key)
            else:
                key_list.append(key)
    return key_list, val_keys

def leave_one_subject_out_split(data: dict, days_include=0):
    """
    @param data: data for which the splits are needed to be generated.
    @param days_include: the number of days of 
                            leaved out data included in the taining
    @return: Return list of dictionary (map: train_ids, val_ids -> data_keys)
    """
    print('########## leave one out split, subject: ' + subject + '##########')
    splits = list()

    data_keys = data['data'].keys()

    student_key = dict() # map: id -> keys
    for key in data_keys:
        try:
            student_key[key.split('_')[0]].append(key)
        except:
            student_key[key.split('_')[0]] = [key]
        
    #for student in student_key:
    for student in student_key:
        splitting_dict = dict()
        splitting_dict['train_ids'] = list()
        for rest_student in student_key:
            if rest_student != student:
                splitting_dict['train_ids'] += student_key[rest_student]
            else:
                loo_train_keys, loo_val_keys = get_first_n_data(student_key[rest_student], days_include)
                splitting_dict['train_ids'] += loo_train_keys
                splitting_dict['val_ids'] = loo_val_keys
        splits.append(splitting_dict)

    return splits


def get_k_fod_cross_val_splits_stratified_by_students(data: dict, groups:dict, n_splits=5,
                                                      stratification_type="students"):
    """
    @param data: data for which the splits are needed to be generated.
    @param groups: map: student_ids -> groups ids
    @param n_splits: number of split
    @param stratification_type: deterimine the criteria for stratified split
    @return: Return list of dictionary (map: train_ids, val_ids -> data_keys)
    """
    
    print('########## k_fold stratification split, stratified by: ' + stratification_type + '############')
    print('split n: ' + str(n_splits))
    splits = list()

    data_keys = data['data'].keys()

    # determine values in stratified column
    stratification_column = list()
    pos = 0 if stratification_type == "students" else -1 if stratification_type == 'labels' else None
    if pos != None:
        for key in data_keys:
            stratification_column.append(int(key.split('_')[pos]))
    elif stratification_type == 'student_label':
        keys, labels = conversions.extract_keys_and_labels_from_dict(data)
        student_ids = conversions.extract_student_ids_from_keys(keys)
        for i in range(len(student_ids)):
            stratification_column.append(str(student_ids[i]) + "_" + str(labels[i]))
    else:
        print('No such kind of criteria for splitting!!!')
        exit()

    # splitting
    data_keys = np.array(list(data_keys))
    stratification_column = np.array(list(stratification_column))
    splitter = StratifiedKFold(n_splits=n_splits, random_state=SPLITTER_RANDOM_STATE)
    for train_index, val_index in splitter.split(X=data_keys, y=stratification_column):

        splitting_dict = dict()
        splitting_dict['train_ids'] = data_keys[train_index].tolist()
        splitting_dict['val_ids'] = data_keys[val_index].tolist()
        splits.append(splitting_dict)

    return splits


def filter_student_id(filter_by_student_ids, student_id_list, labels_list, data_keys):
    filtered_student_id_list, filtered_labels_list, filtered_data_keys = [], [], []

    for idx, student_id in enumerate(student_id_list):
        if conversions.convert_to_int_if_str(student_id) in filter_by_student_ids:
            filtered_student_id_list.append(student_id)
            filtered_labels_list.append(labels_list[idx])
            filtered_data_keys.append(data_keys[idx])

    return filtered_student_id_list, filtered_labels_list, filtered_data_keys
