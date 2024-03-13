import numpy as np
import pickle
import copy
import tqdm
from sklearn import metrics
from sklearn.preprocessing import MultiLabelBinarizer

from src.experiments.models import *
from src.utils import cross_val

######## helper methods ########
def read_data(filename):
    data = None
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

def get_splits(split_name, data, student_groups, days_include=0):
    splits = None
    if split_name == '5fold':
        # k-fold cross validation
        stratification_type = "student_label"
        n_splits = 5
        splits = cross_val.get_k_fod_cross_val_splits_stratified_by_students(
            data=data,
            groups=student_groups,
            n_splits=n_splits,
            stratification_type=stratification_type
        )
    elif split_name == 'loocv':
        splits = cross_val.leave_one_subject_out_split(data=data, days_include=days_include)
        print("Num Splits: ", len(splits))
    elif split_name == '5fold_c':# chronological order
        splits = cross_val.get_k_fod_chronological(
            data=data,
            n_splits=5
        )
    return splits

def get_mini_batchs(batch_size, inds, shuffle=True):
    batch_inds = list()
    if shuffle:
        np.random.shuffle(inds)
    i = 0
    while i < len(inds):
        batch_inds.append(inds[i:i+batch_size])
        i += batch_size
    return batch_inds

def formatting_train_val_data(data, training_params):
    train_data = {
        'samples': list(),
        'covariate_data': torch.tensor([]).to(training_params['device']),
        'labels': torch.tensor([]).type(torch.LongTensor).to(training_params['device']),
        'ids': list(),
    }
    val_data = {
        'samples': list(),
        'covariate_data': torch.tensor([]).to(training_params['device']),
        'labels': torch.tensor([]).type(torch.LongTensor).to(training_params['device']),
        'ids': list(),
    }
    for key in data['train_ids']:
        actual_data, covariate_data, histogram_data, train_label = data['data'][key]
        if training_params['use_histogram']:
            actual_data = histogram_data.unsqueeze(0)
        
        # for train on two labels
        # train_label = torch.minimum(torch.tensor([1]), train_label)

        # update global variable
        train_data['samples'].append(actual_data.to(training_params['device']))
        train_data['labels'] = torch.cat((train_data['labels'], train_label.to(training_params['device'])), dim=0)
        train_data['covariate_data'] = torch.cat((train_data['covariate_data'], covariate_data.unsqueeze(0).to(training_params['device'])), dim=0)
        train_data['ids'].append(key.split('_')[0])
    for key in data['val_ids']:
        actual_data, covariate_data, histogram_data, train_label = data['data'][key]
        if training_params['use_histogram']:
            actual_data = histogram_data.unsqueeze(0)

        # for train on two labels
        # train_label = torch.minimum(torch.tensor([1]), train_label)

        val_data['samples'].append(actual_data.to(training_params['device']))
        val_data['labels'] = torch.cat((val_data['labels'], train_label.to(training_params['device'])), dim=0)
        val_data['covariate_data'] = torch.cat((val_data['covariate_data'], covariate_data.unsqueeze(0).to(training_params['device'])), dim=0)
        val_data['ids'].append(key.split('_')[0])
    train_data['inds'] = list(range(len(train_data['samples'])))
    train_data['ids'] = np.array(train_data['ids'])
    val_data['inds'] = list(range(len(val_data['samples'])))
    val_data['ids'] = np.array(val_data['ids'])
    return train_data, val_data
    
#### evaluation metrics ####
def eval_accuracy(y_pred, y_true):
    return (y_pred == y_true).mean()

def eval_f1_score(y_pred, y_true, avg_type):
    return metrics.f1_score(y_true, y_pred, average=avg_type)

def eval_auc_score(y_pred, y_true, labels, avg_type, b=False):
    if not b:
        mlb = MultiLabelBinarizer()
        mlb.fit(labels)
        y_true = mlb.transform([[i] for i in y_true])
        # y_pred = mlb.transform([[i] for i in y_pred])
    
    roc_weighted = None
    try:
        roc_weighted = metrics.roc_auc_score(y_true, y_pred, average=avg_type)
    except:
        roc_weighted = 0.0
    return roc_weighted
################################################################

def train_ae_then_freeze(model, optimizer, reconstruction_criterion, train_data, training_params):
    # Training Autoencoder (AE)
    for e in range(200):
         # training
        model.train()
        batchs = get_mini_batchs(training_params['batch_size'], train_data['inds'])        
        train_loss = 0
        for batch in batchs:
            # forward
            final_out, AE_out = model(
                x=train_data['samples'],
                inds=batch,
                ids=train_data['ids'][batch],
                covariate_data=train_data['covariate_data'][batch]
            )

            reconstruction_loss = 0
            for i in range(len(AE_out)):
                reconstruction_loss += reconstruction_criterion(train_data['samples'][batch[i]], AE_out[i])
            reconstruction_loss *= training_params['loss_weight']['alpha']
            total_loss = reconstruction_loss

            # backpropagation
            model.zero_grad()
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.cpu().detach().item()
    
    # freeze AE
    for p in model.autoencoder.parameters():
        p.requires_grad = False

#### main trian function ####
def train_and_val(
    data,
    model_params,
    training_params,
    pre_record=None,
    leaved_student=None,
    up_weight_k=None,
):  
    # prepare data
    train_data, val_data = formatting_train_val_data(data, training_params)
    # return val_data['labels'].cpu() # new added

    # declare results for save
    saved_records = {
        'model': None,
        'train_losses': list(),
        'val_losses': list(),  
        'outputs': list(),
        'generic_outputs': list(),
        'confmats': list(),
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
        'labels': list(),
        'generic_records': {
            'outputs': list(),
            'confmats': list(),
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
        }
    }

    # construct model
    print('Initializing...')
    model = None
    if pre_record == None:
        model = MultitaskAutoencoder(model_params, training_params['use_covariates']).to(training_params['device'])
    else:
        model = pre_record['model'].to(training_params['device'])
    reconstruction_criterion = torch.nn.L1Loss(reduction="sum")
    classification_criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(training_params['class_weights'], device=training_params['device'])
    )

    # construct optimizer
    optimizer = torch.optim.Adam(
        [
            {'params': model.autoencoder.parameters()},
            {'params': model.out_heads.parameters()},
            {'params': model.branching.parameters(), 'lr': training_params['branching_lr']},
            {'params': model.branch_layer.parameters(), 'lr': training_params['branching_lr']},
        ],
        lr=training_params['global_lr'],
        weight_decay=training_params['weight_decay'],
    )

    # # train ae prior than train entire
    # train_ae_then_freeze(model, optimizer, reconstruction_criterion, train_data, training_params)

    # start training
    print('Training...')
    for epoch in tqdm.tqdm(range(training_params['epochs'])):
        # training
        model.train()
        batchs = get_mini_batchs(training_params['batch_size'], train_data['inds'])        
        train_loss = 0
        for batch in batchs:
            # forward
            if not model.with_generic_head:
                final_out, AE_out = model(
                    x=train_data['samples'],
                    inds=batch,
                    ids=train_data['ids'][batch],
                    covariate_data=train_data['covariate_data'][batch]
                )
            else:
                final_out, AE_out, generic_out = model(
                    x=train_data['samples'],
                    inds=batch,
                    ids=train_data['ids'][batch],
                    covariate_data=train_data['covariate_data'][batch]
                )

            # up-weighting loss
            # fetch weight vector
            weight_vec = list()
            ids = train_data['ids'][batch]
            for i in range(len(batch)):
                if ids[i] == leaved_student:
                    weight_vec.append([up_weight_k])
                else:
                    weight_vec.append([1])
            weight_vec = torch.Tensor(weight_vec).to(training_params['device'])

            # up-weight
            final_out *= weight_vec
            # AE_out *= weight_vec

            # compute loss
            classification_loss = classification_criterion(final_out, train_data['labels'][batch]) * training_params['loss_weight']['beta'] 
            total_loss = classification_loss

            if training_params['use_decoder']:
                reconstruction_loss = 0
                for i in range(len(AE_out)):
                    reconstruction_loss += reconstruction_criterion(train_data['samples'][batch[i]], AE_out[i])
                reconstruction_loss *= training_params['loss_weight']['alpha']
                total_loss = reconstruction_loss + classification_loss

            if model.with_generic_head:
                total_loss += classification_criterion(generic_out, train_data['labels'][batch]) * training_params['loss_weight']['theta']

            # backpropagation
            model.zero_grad()
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.cpu().detach().item()

        # validation
        model.eval()
        # forward
        if not model.with_generic_head:
            final_out, AE_out = model(
                x=val_data['samples'],
                inds=list(range(len(val_data['samples']))), 
                ids=val_data['ids'],
                covariate_data=val_data['covariate_data']
            )
        else:
            final_out, AE_out, generic_out = model(
                x=val_data['samples'],
                inds=list(range(len(val_data['samples']))), 
                ids=val_data['ids'],
                covariate_data=val_data['covariate_data']
            )

        # compute loss
        classification_loss = classification_criterion(final_out, val_data['labels']) * training_params['loss_weight']['beta'] 
        val_loss = classification_loss

        if training_params['use_decoder']:
            reconstruction_loss = 0
            for i in range(len(AE_out)):
                reconstruction_loss += reconstruction_criterion(val_data['samples'][i], AE_out[i])
            reconstruction_loss *= training_params['loss_weight']['alpha']
            val_loss = reconstruction_loss + classification_loss

        # evaluate and update global variables
        saved_records['train_losses'].append(train_loss)
        saved_records['val_losses'].append(val_loss.cpu().detach().item())
        
        # save validation information
        saved_records['outputs'].append(final_out.cpu().detach().numpy())

        y_pred = np.argmax(saved_records['outputs'][-1], axis=1)
        y_true = val_data['labels'].cpu().detach().numpy()
        labels = [[0], [1], [2]]
        saved_records['confmats'].append(metrics.confusion_matrix(y_true, y_pred, labels=[i[0] for i in labels]))

        for avg_type in ['micro', 'macro', 'weighted']:
            saved_records['val_auc'][avg_type].append(eval_auc_score(saved_records['outputs'][-1], y_true, labels, avg_type))
            saved_records['val_f1'][avg_type].append(eval_f1_score(y_pred, y_true, avg_type))
        
        if saved_records['val_f1']['weighted'][-1] == max(saved_records['val_f1']['weighted']):
            saved_records['model'] = copy.deepcopy(model).cpu()

        # if the model has the generic head
        if model.with_generic_head:
            val_loss += classification_criterion(generic_out, val_data['labels']) * training_params['loss_weight']['theta'] 
            saved_records['generic_records']['outputs'].append(generic_out.cpu().detach().numpy())

            y_pred = np.argmax(saved_records['generic_records']['outputs'][-1], axis=1)
            y_true = val_data['labels'].cpu().detach().numpy()
            labels = [[0], [1], [2]]
            saved_records['generic_records']['confmats'].append(metrics.confusion_matrix(y_true, y_pred, labels=[i[0] for i in labels]))

            for avg_type in ['micro', 'macro', 'weighted']:
                saved_records['generic_records']['val_auc'][avg_type].append(eval_auc_score(saved_records['generic_records']['outputs'][-1], y_true, labels, avg_type))
                saved_records['generic_records']['val_f1'][avg_type].append(eval_f1_score(y_pred, y_true, avg_type))

        print("F1 Score This Epoch: {} Best Score: {}".format(saved_records['val_f1']['weighted'][-1], max(saved_records['val_f1']['weighted'])))

    # final return
    return saved_records
