import sys
from copy import deepcopy

from src.experiments.config import *
from src.utils.train_val_utils import *
import src.utils.tensorify as tensorify
import src.utils.data_conversion_utils as conversions
from src.experiments.layers import *

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
    clusters_name = 'one_for_each'
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
    print('Validation Type:', split_name)
    splits = get_splits(split_name, data, student_groups)

    # get configurations for model and training
    config = get_config(job_name, device, num_features, student_groups, num_branches)
    model_params = config['model_params']
    training_params = config['training_params']

    if training_params['use_covariates']:
        model_params['shared_in_size'] += num_covariates
    ####################################

    # start training
    saved_records = list() # list of results from each split
    for split_no, split in enumerate(splits):
        print("Split No: ", split_no)

        val_student = split['val_ids'][0].split('_')[0]
        print('leaved student:', val_student)

        # delete the leaved key
        key = 'student_{}'.format(val_student)
        val = model_params['groups']['student_{}'.format(val_student)]
        del model_params['groups']['student_{}'.format(val_student)]

        # pre-Training
        tensorified_data['train_ids'] = [i for i in split['train_ids'] if i.split('_')[0] != val_student]
        tensorified_data['val_ids'] = tensorified_data['train_ids'][:1]

        print('Pre-Training...')
        pre_record = train_and_val(
            data=tensorified_data,
            model_params=model_params,
            training_params=training_params,
        )

        # assemble downstream task
        downstream_layers = personal_head(
            num_heads=22, 
            in_size=model_params['heads_hidden_size'],
            hidden_size=model_params['heads_hidden_size'],
            out_size=model_params['num_classes'],
            device=model_params['device'],
        )

        pre_record['model'].set_transfer_learn(downstream_layers)
        # pre_record['model'].transfer_learn = True
        # pre_record['model'].out_heads.transfer_learn = True
        # pre_record['model'].downstream_layers = downstream_layers

        # fine-tuning
        tensorified_data['train_ids'] = [i for i in split['train_ids'] if i.split('_')[0] == val_student]
        tensorified_data['val_ids'] = split['val_ids']
        
        print('Fine-Tuning...')
        saved_records.append(train_and_val(
            data=tensorified_data,
            model_params=model_params,
            training_params=training_params,
            pre_record=pre_record,
        ))

        # retrieve the deleted key
        model_params['groups'][key] = val
    
    #  # new added
    # saved_filename = 'val_labels_{}_{}_{}.pkl'.format(job_name, split_name, num_branches)
    # with open(saved_filename, 'wb') as f:
    #     pickle.dump(saved_records, f)
    
    # save results
    saved_filename = 'data/cross_val_scores/transfer_learn_{}_{}_{}.pkl'.format(job_name, split_name, num_branches)
    with open(saved_filename, 'wb') as f:
        pickle.dump(saved_records, f)
        