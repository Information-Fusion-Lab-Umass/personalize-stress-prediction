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

def gumbel_softmax(mat, tau=0.1, decay=True):
    if not decay:
        return np.array(mat), [0 for _ in range(len(mat))]
    ts = list()
    res = list()
    for i in range(len(mat)):
        ts.append(tau)
        v = np.exp(np.log(mat[i]) / tau)
        res.append(v / np.sum(v))
        tau = max(0.01, tau*0.98)
    res = np.array(res)
    print("Max:", np.max(res), "Min:", np.min(res))
    return res, ts

# process
def check_branch_evolve(model, sub_ind="0", tau=0.1, decay=True, cmap='jet', step_size=3):
    # get branching evolve parameters
    branches = model.record_branch(return_only=True)[sub_ind]

    # transform
    mat, ts = gumbel_softmax(branches, tau=tau, decay=decay)
    inds = [i for i in range(len(mat)) if i % step_size == 0]
    mat = mat[inds]

    # heat map
    plt.figure(figsize = (15,5))
    plt.imshow(mat.T, cmap=cmap)
    tick_list = np.linspace(0, 1.0, 5)
    cb = plt.colorbar(orientation='horizontal', aspect=50, location="bottom", pad=0.4)
    cb.ax.tick_params(labelsize=22) 
    cb.ax.set_title('Colorbar Reference')
    cb.ax.set_xlabel('Branching Probability from 0 to 1')
#     cb.ax.legend()
#     cb.ax.set_yticklabels(tick_list)
    plt.xlabel("Epochs")
    plt.ylabel("Branching ID")
    plt.show()