import sys
from copy import deepcopy

from src.experiments.config import *
from src.utils.train_val_utils import *
import src.utils.tensorify as tensorify
import src.utils.data_conversion_utils as conversions

if __name__ == '__main__':
    # read command line arguments
    job_name = sys.argv[1] # choice: 'test', 'calm_net', 'calm_net_with_branching'
    split_name = sys.argv[2] # choice: ['5fold', 'loocv']
    num_branches = int(sys.argv[3]) # any interger
    print('Job name:', job_name)
    print('Type:', split_name)

    # use gpu if available
    device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
    print("Device: ", device)

    # load data
    print('Loading Data...')
    data_file_path = 'data/training_data/shuffled_splits/training_date_normalized_shuffled_splits_select_features_no_prev_stress_all_students.pkl'
    data = read_data(data_file_path)
    tensorified_data = tensorify.tensorify_data_gru_d(deepcopy(data), torch.cuda.is_available())
    student_list = conversions.extract_distinct_student_idsfrom_keys(data['data'].keys())

    # load groups
    clusters_name = 'all_in_one' # 'one_for_each', 'all_in_one'
    print('The groups: ' + clusters_name)
    groups_file_path = 'src/experiments/clustering/student_groups/' + clusters_name + '.pkl'
    student_groups = read_data(groups_file_path) # student groups

    # check how students are distributed
    print("student distribution: ")
    rev_groups = dict() # map: group_id -> student_ids
    for student in student_groups:
        if rev_groups.get(student_groups[student]) != None:
            rev_groups[student_groups[student]].append(student)
        else:
            rev_groups[student_groups[student]] = [student]
    for group in rev_groups:
        print(group + ': ' + str(rev_groups[group]))
    
    ############ SETTINGS ############
    use_historgram = True
    first_key = next(iter(data['data'].keys()))
    if use_historgram:
        num_features = len(data['data'][first_key][4][0])
    else:
        num_features = len(data['data'][first_key][0][0])
    num_covariates = len(data['data'][first_key][3]) # 3 is the covariate index

    # get split 
    days = [7, 14, 21, 28]
    stats = dict() # map: day -> dictionary(map: student -> [train_num, val_num])
    for day in days:
        stats[day] = dict() # map: student -> [train_num, val_num]

        print('Validation Type:', split_name)
        splits = get_splits(split_name, data, student_groups, days_include=day)

        # get configurations for model and training
        config = get_config(job_name, device, num_features, student_groups, num_branches)
        model_params = config['model_params']
        training_params = config['training_params']

        if training_params['use_covariates']:
            model_params['shared_in_size'] += num_covariates
        ####################################

        for split_no, split in enumerate(splits):
            # if the days include exceed the max range of date of keys, skip
            # this three lines only for loocv. A empty dict will be appended
            if len(split['val_ids']) == 0:
                continue
            
            for key in split['val_ids']:
                student = int(key.split('_')[0])
                break

            train_num = 0
            for key in split['train_ids']:
                curr_s = int(key.split('_')[0])
                if curr_s == student:
                    train_num += 1
            val_num = len(split['val_ids'])
            
            stats[day][student] = [train_num, val_num]
    
    # save results
    with open('data/check/train_val_num_stats.pkl', 'wb') as f:
        pickle.dump(stats, f)