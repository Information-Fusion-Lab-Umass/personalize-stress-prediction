import tqdm
import pickle
import sys

import numpy as np
import torch
from torch import nn
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

from src.utils.train_val_utils import *

SPLITTER_RANDOM_STATE = 100

def cv(student_data, n_splits=5):
    data = list()
    samples = list()
    labels = list()
    stratification_column = list()
    for s in student_data:
        for i in range(len(student_data[s]['features'])):
            samples.append(student_data[s]['features'][i])
            labels.append(student_data[s]['labels'][i])
            stratification_column.append('{}_{}'.format(s, student_data[s]['labels'][i]))
        
    samples = np.array(samples)
    labels = np.array(labels)
    stratification_column = np.array(stratification_column)
    splitter = StratifiedKFold(n_splits=n_splits, random_state=SPLITTER_RANDOM_STATE)
    for train_index, val_index in splitter.split(X=samples, y=stratification_column):
        curr_fold = {
            'train_data': samples[train_index],
            'train_labels': labels[train_index],
            'val_data': samples[val_index],
            'val_labels': labels[val_index],
        }

        for i in curr_fold:
            print(curr_fold[i].shape)
        print('++++++++++++++++++++++++++++')

        data.append(curr_fold)
    return data
    
def loocv(student_data, key_data, days_include=0):
    data = list()
    splits = cross_val.leave_one_subject_out_split(key_data, days_include=days_include)
    for split in splits:
        curr_fold = {
            'train_data': np.array([key_data['data'][split['train_ids'][0]]['sample']]),
            'train_labels': np.array([key_data['data'][split['train_ids'][0]]['label']]),
            'val_data': np.array([key_data['data'][split['val_ids'][0]]['sample']]),
            'val_labels': np.array([key_data['data'][split['val_ids'][0]]['label']])
        }
        for i in range(1, len(split['train_ids'])):
            curr_fold['train_data'] = np.concatenate((curr_fold['train_data'], np.array([key_data['data'][split['train_ids'][i]]['sample']])), axis=0)
            # print(curr_fold['train_labels'])
            # print(key_data['data'][split['train_ids'][i]]['label'])
            curr_fold['train_labels'] = np.concatenate((curr_fold['train_labels'], np.array([key_data['data'][split['train_ids'][i]]['label']])), axis=0)
        for i in range(1, len(split['val_ids'])):
            curr_fold['val_data'] = np.concatenate((curr_fold['val_data'], np.array([key_data['data'][split['val_ids'][i]]['sample']])), axis=0)
            curr_fold['val_labels'] = np.concatenate((curr_fold['val_labels'], np.array([key_data['data'][split['val_ids'][i]]['label']])), axis=0)
    # for s in student_data:
    #     curr_fold = {
    #         'train_data': None,
    #         'train_labels': None,
    #         'val_data': None,
    #         'val_labels': None
    #     }

    #     for other_s in student_data:
    #         if other_s == s:
    #             loo_train_keys, loo_val_keys = get_first_n_data(key_data[s], days_include)
    #             if curr_fold['train_data'] is None:
    #                 curr_fold['train_data'] = student_data[other_s]['features']
    #                 curr_fold['train_labels'] = student_data[other_s]['labels']
    #         if curr_fold['train_data'] is None:
    #             curr_fold['train_data'] = student_data[other_s]['features']
    #             curr_fold['train_labels'] = student_data[other_s]['labels']
    #         else:
    #             curr_fold['train_data'] = np.concatenate((curr_fold['train_data'], student_data[other_s]['features']), axis=0)
    #             curr_fold['train_labels'] = np.concatenate((curr_fold['train_labels'], student_data[other_s]['labels']), axis=0)
        for i in curr_fold:
            print(curr_fold[i].shape)
        print('++++++++++++++++++++++++++++')
        print(' ')
        data.append(curr_fold)
    # exit()
    return data

def load_data(days_include=0):
    # student_data = {
    #     'student_1': {
    #         'features': np.ones((5, 12)),
    #         'labels': np.ones(5)
    #     },
    #     'student_2': {
    #         'features': np.ones((5, 12)),
    #         'labels': np.ones(5)
    #     },
    #     'student_3': {
    #         'features': np.ones((5, 12)),
    #         'labels': np.ones(5)
    #     }
    # }
    with open("data/location_data/gatis-new-23.pkl", 'rb') as f:
        all_data = pickle.load(f)
    
    student_data = dict()
    key_data = {'data': dict()}
    students = set([s for s in all_data["student_id"]])
    for s in students:
        student_data[s] = dict()
        curr_data = all_data.loc[all_data['student_id'] == s].sort_values(by="time").to_numpy()
        student_data[s]["features"] = curr_data[:, 1:13].astype('float')
        student_data[s]["labels"] = curr_data[:, 13].astype('int')
        student_data[s]["times"] = curr_data[:, 0].astype('str')
        for i in range(len(student_data[s]["features"])):
            times = student_data[s]["times"][i].split()[0].split('-')[1:]
            key_data['data']["{}_{}_{}".format(s, int(times[0]), int(times[1]))] = {
                'sample': student_data[s]["features"][i],
                'label': student_data[s]["labels"][i]
            }
    
    # return loocv(student_data, key_data, days_include=days_include)
    return cv(student_data)

class LocationMLP(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.fc_liner = nn.Sequential(
            nn.Linear(12, 57),
            nn.Tanh(),
            nn.BatchNorm1d(57),
            nn.Dropout(p=0.35),
            nn.Linear(57, 35),
            nn.Tanh(),
            nn.BatchNorm1d(35),
            nn.Dropout(p=0.25),
            nn.Linear(35, 35),
            nn.Tanh(),
            nn.BatchNorm1d(35),
            nn.Dropout(p=0.15),
            nn.Linear(35, 3),
            nn.Softmax(dim=1),
            nn.BatchNorm1d(3),
        )

        self.loss_func = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.6456, 0.5635, 1.0000], device=device)
    )
    
    def forward(self, x):
        return self.fc_liner(x)
    
    def loss(self, samples, labels):
        out = self.forward(samples)
        return self.loss_func(out, labels)
    
    def predict(self, samples):
        return self.forward(samples).argmax(dim=1)

# ================= SPLIT LINE ====================================================

def train_val(
    model, 
    optimizer, 
    train_data, 
    train_labels,
    val_data, 
    val_labels,
    epochs, 
    batch_size
    ):

    saved_records = {
        'val_f1': {
            'micro': list(),
            'macro': list(),
            'weighted': list()
        },
        'val_auc': {
            'micro': list(),
            'macro': list(),
            'weighted': list()
        },
        'outputs': list(),
        'confmats': list(),
    }

    train_inds = [i for i in range(len(train_data))]
    val_inds = [i for i in range(len(val_data))]

    for e in tqdm.tqdm(range(epochs)):
        # train
        model.train()
        batch_inds = get_mini_batchs(batch_size, train_inds)
        for batch_ind in batch_inds:
            if len(batch_ind) == 1:
                continue
            # forward
            loss = model.loss(train_data[batch_ind], train_labels[batch_ind])

            # backpropagation
            model.zero_grad()
            loss.backward()
            optimizer.step()
        
        # validation
        model.eval()
        y_pred = list()
        batch_inds = get_mini_batchs(batch_size, val_inds)
        for batch_ind in batch_inds:
            y_pred += model.predict(val_data[batch_ind]).cpu().detach().numpy().tolist()
        saved_records['outputs'].append(model(val_data).cpu().detach().numpy())
        saved_records['confmats'].append(metrics.confusion_matrix(val_labels, y_pred, labels=[0, 1, 2]))
        
        # evaluate
        for avg_type in ['micro', 'macro', 'weighted']:
            saved_records['val_f1'][avg_type].append(eval_f1_score(y_pred, val_labels, avg_type))
            saved_records['val_auc'][avg_type].append(eval_auc_score(saved_records['outputs'][-1], val_labels, [[0], [1], [2]], avg_type))
    print("best weighted val f1", np.max(saved_records['val_f1']['weighted']))
    print("best micro val f1", np.max(saved_records['val_f1']['micro']))
    return saved_records

def run(data, device, remark):
    records = list()

    # config hyper-param
    epochs = 300 # 300
    batch_size = 32
    for fold in data:
        train_data = torch.Tensor(fold['train_data']).to(device).float()
        train_labels = torch.Tensor(fold['train_labels']).to(device).long()
        val_data = torch.Tensor(fold['val_data']).to(device).float()
        val_labels = fold['val_labels']
        # val_labels = torch.Tensor(fold['val_labels']).long()

        print('train shape', train_data.shape, train_labels.shape)
        print('val shape', val_data.shape, val_labels.shape)
    
        model = LocationMLP(device).to(device)

        # optimizer = torch.optim.Adam(
        #     model.parameters(),
        #     lr=1e-3,
        #     weight_decay=1e-4
        # )

        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.98)

        records.append(train_val(
            model, 
            optimizer, 
            train_data, 
            train_labels,
            val_data, 
            val_labels,
            epochs, 
            batch_size
        ))
    
    # save results
    with open('data/cross_val_scores/location_mlp_5fold_0_0_{}.pkl'.format(remark), 'wb') as f:
        pickle.dump(records, f)
    
    # print("avg weighted f1", np.mean([np.max(r['val_f1']['weighted']) for r in records]))
    # print("avg micro f1", np.mean([np.max(r['val_f1']['micro']) for r in records]))

    
if __name__ == '__main__':
    # use gpu if available
    device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
    # device = torch.device('cpu')
    print("Device: ", device)

    # start train-val process
    days_include = int(sys.argv[1]) # 7, 14, 21, 28
    data = load_data(days_include=days_include)
    # run(data, device, "ol_{}".format(days_include))

    for i in range(10):
        run(data, device, str(i))
    