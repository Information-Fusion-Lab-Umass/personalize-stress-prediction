import sys
from copy import deepcopy

from src.experiments.config import *
from src.utils.train_val_utils import *
import src.utils.tensorify as tensorify
import src.utils.data_conversion_utils as conversions

def run_exp(job_name, split_name, num_branches, days_include, clusters_name, remark):
    print('Job name:', job_name)
    print('Type:', split_name)

    # use gpu if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    print("Device: ", device)

    # ====== LOAD STUDENTLIFE DATA ===================================

    # load data
    print('Loading Data...')
    data_file_path = 'data/training_data/shuffled_splits/training_date_normalized_shuffled_splits_select_features_no_prev_stress_all_students.pkl'
    data = read_data(data_file_path)
    tensorified_data = tensorify.tensorify_data_gru_d(deepcopy(data), torch.cuda.is_available())
    student_list = conversions.extract_distinct_student_idsfrom_keys(data['data'].keys())

    # load groups
    clusters_name = clusters_name # 'one_for_each', 'all_in_one', 'pre_survey_scores_7
    print('The groups: ' + clusters_name)
    groups_file_path = 'src/experiments/clustering/student_groups/' + clusters_name + '.pkl'
    student_groups = read_data(groups_file_path) # student groups

    # ================================================================
    
    # ====== LOAD SAMPLE DATA ========================================

    # with open("data/training_data/sample_input.pkl", "rb") as f:
    #     data = pickle.load(f)
    #     tensorified_data = data
    #     student_groups = data["one_for_each"]

    # ================================================================

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
        num_features = len(tensorified_data['data'][first_key][2][0])
    else:
        num_features = len(tensorified_data['data'][first_key][0][0])
    num_covariates = len(tensorified_data['data'][first_key][1]) # 1 is the covariate index

    # get split 
    print('Validation Type:', split_name)
    splits = get_splits(split_name, data, student_groups, days_include=days_include)
    # print(splits)
    
    # get configurations for model and training
    config = get_config(job_name, device, num_features, student_groups, num_branches)
    model_params = config['model_params']
    training_params = config['training_params']

    if training_params['use_covariates']:
        model_params['shared_in_size'] += num_covariates
    ####################################

    # ======== FOR UPWEIGHT ======================================
    # # start training
    # student_id_list = [4, 7, 8, 10, 14, 16, 17, 19, 22, 23, 24, 32, 33, 35, 36, 43, 44, 49, 51, 52, 53, 57, 58]
    # # chosen_student = [4, 7, 8, 16, 22, 52]
    # chosen_student = student_id_list

    # # for determine up-weight coefficient
    # if split_name == 'loocv' and days_include > 0:
    #     with open('data/check/train_val_num_stats.pkl', 'rb') as f:
    #         train_val_stats = pickle.load(f)[days_include]

    # ============================================================

    saved_records = list() # list of results from each split
    for split_no, split in enumerate(splits):
        # if the days include exceed the max range of date of keys, skip
        # this three lines only for loocv. A empty dict will be appended
        if len(split['val_ids']) == 0:
            saved_records.append(dict())
            continue

        print("Split No: ", split_no)

        # fetch train and val data
        tensorified_data['train_ids'] = split['train_ids']
        tensorified_data['val_ids'] = split['val_ids']

        # fetch leaved out student
        if split_name == 'loocv' and days_include > 0:
            leaved_student = split['val_ids'][0].split('_')[0]

            # # for upweight
            # if int(leaved_student) not in chosen_student:
            #     continue

            # # determine up-weight coefficient
            # leaved_train = train_val_stats[int(leaved_student)][0] # #of train from leaved_student
            # rest_train = len(split['train_ids']) - leaved_train
            # up_weight_k = int(rest_train / leaved_train)

            up_weight_k = 1.0
        else: # 5-fold
            leaved_student = -1
            up_weight_k = 1.0

        # Training and Validation
        curr_record = train_and_val(
            data=tensorified_data,
            model_params=model_params,
            training_params=training_params,
            leaved_student=leaved_student,
            up_weight_k=up_weight_k,
        )
        curr_record['model'] = None # tentative, since EmbConvBlock contain lambda. can keep model once the model write/load function is optimize using torch.load
        saved_records.append(curr_record)
    
    #  # new added
    # saved_filename = 'val_labels_{}_{}_{}.pkl'.format(job_name, split_name, num_branches)
    # with open(saved_filename, 'wb') as f:
    #     pickle.dump(saved_records, f)

    # save results
    saved_filename = 'data/cross_val_scores/{}_{}_{}_{}_{}.pkl'.format(job_name, split_name, num_branches, days_include, remark)
    with open(saved_filename, 'wb') as f:
        pickle.dump(saved_records, f)

if __name__ == '__main__':
    # read command line arguments
    job_name = sys.argv[1] # choice: 'test', 'calm_net', 'calm_net_with_branching'
    split_name = sys.argv[2] # choice: ['5fold', 'loocv']
    num_branches = int(sys.argv[3]) # any interger, 1 for calm_net, >=2 for branched_calm_net
    days_include = int(sys.argv[4]) # any interger, 0 for 5fold, {7, 14, 21, 28} for loocv
    clusters_name = sys.argv[5] # 'one_for_each' for calm_net and branched_calm_net
    remark = sys.argv[6] # any additional info (for differentiate saved filename)
    
    run_exp(job_name, split_name, num_branches, days_include, clusters_name, remark)