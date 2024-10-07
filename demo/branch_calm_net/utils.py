import os
import pickle
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def get_default_config(device):
    # group placeholder
    subject_group = { # map: ID_# -> group_id
        "ID_0": '0',
        "ID_1": '1',
        "ID_2": '2'
    }
    
    return {
        'model_params': {
            'device': device,
            'in_size': 14,
            'AE': 'lstm', 
            'AE_num_layers': 1,
            'AE_hidden_size': 128,
            'covariate_size': 4, # number of covariates
            'shared_in_size': 128, # init bottleneck size
            'shared_hidden_size': 256,
            'num_branches': 5, # when equal to 1, it is equivalent to CALM_Net
            'groups': subject_group,
            'heads_hidden_size': 64,
            'num_classes': 3
        },
        'training_params': {
            'device': device, 
            'loss_weight': {
                'alpha': 1e-4,
                'beta': 1,
                'theta': 1 / 22, # 1 over number of students
            },
            'class_weights': [0.6456, 0.5635, 1.0000],
            # 'class_weights': [0.6456, 0.5635+1.0000],
            'global_lr': 1e-5,
            'branching_lr': 1e-5,
            'weight_decay': 1e-4,
            'epochs': 50, 
            'batch_size': 4,
            'use_histogram': True,
            'use_covariates': True, 
            'use_decoder': True,
        }
    }

def get_samples(ids, example_idx=[0]):
    all_fns = os.listdir('samples')

    sample_x, sample_y, sample_ids, sample_cov = list(), list(), list(), list()
    for id_ in tqdm(ids):
        for l in ['0', '1', '2']:
            curr_fns = np.array([fn for fn in all_fns if '{}_{}_'.format(id_, l) in fn])[example_idx]
            for curr_fn in curr_fns:
                # load data
                with open('samples/{}'.format(curr_fn), 'rb') as f:
                    sample = pickle.load(f)['signal']
                
                # load covariates
                with open('portraits/{}'.format(id_), 'rb') as f:
                    cov = pickle.load(f)
                
                sample_x.append(sample)
                sample_y.append(int(l))
                sample_ids.append(id_)
                sample_cov.append(cov)
    return torch.tensor(sample_x).float(), torch.tensor(sample_y).long(), sample_ids, torch.tensor(sample_cov).float()