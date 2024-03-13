import pickle
import os
import torch
from tqdm import tqdm
import numpy as np

from sklearn.model_selection import StratifiedKFold, KFold, TimeSeriesSplit

'''
data/splits_5fold file-structure
{
    "splits": {
        "train_fnames": list(file-names),
        "test_fnames": list(file-names)
    }
    "subjects": list(subjects to construct personalized heads)
}
data/splits_5fold, data/splits_loocv_0, data/splits_5fold_loocv_7, data/splits_5fold_loocv_14, ...

each sample file-structure
{
    "signal": (L, C), multivariate time series,
    "label": R^5, interger stress label,
    "subject": subject id
}
'''

def process_and_store_samples():
    os.mkdir("data/samples")
    subjects = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
    fnames = list()
    for s in tqdm(subjects):
        # fetch samples
        curr_data = None
        for i in range(14):
            with open("preprocess/dl-4-tsc-master/clean_data/WESAD/x_{}_{}.pkl".format(s, i), 'rb') as f:
                d = pickle.load(f)
                d = torch.tensor(d.data).unsqueeze(dim=2) # (N, L)
            if curr_data is None:
                curr_data = d
            else:
                curr_data = torch.cat((curr_data, d), dim=2)

        # fetch labels
        with open("preprocess/dl-4-tsc-master/clean_data/WESAD/y_{}.pkl".format(s), 'rb') as f:
            labels = pickle.load(f)
        
        # save each sample as a file
        for i in range(len(labels)):
            curr_label = int(labels[i] - 1)
            curr_sample = {
                "signal": curr_data[i].numpy().tolist(),
                "label": curr_label,
                "subject": str(s)
            }
            save_fname = "data/samples/{}_{}_{}".format(s, curr_label, i)
            with open(save_fname, 'wb') as f:
                pickle.dump(curr_sample, f)

    # print("data length:", d.data[0].shape) # 240
    # print("num samples:", len(d.data)) # 71, 73, 75, etc.
    # print("num labels:", len(l)) # 71, 73, 75, etc.
    # print("labels:", set(l)) # 1.0, 2.0, 3.0
    return fnames

def split_5fold():
    fnames = np.array([fn for fn in os.listdir("data/samples") if fn[0] != "."])
    ys = np.array(["_".join(fn.split('_')[:2]) for fn in fnames])
    subjects = [str(s) for s in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]]
    # return {splits: ({train_fnames, test_fnames}), subjects: list(subject_id)}

    skf = StratifiedKFold(n_splits=3)
    splits = list()
    for i, (train_index, test_index) in enumerate(skf.split(fnames, ys)):
        splits.append(
            {
                "train_fnames": fnames[train_index],
                "test_fnames": fnames[test_index]
            }
        )
    
    split_record = {
        "splits": splits,
        "subjects": subjects
    }
    with open("data/splits_3fold", 'wb') as f:
        pickle.dump(split_record, f)

def split_subjects(n=5, num_include=0.2):
    fnames = np.array([fn for fn in os.listdir("data/samples") if fn[0] != "."])
    subjects = np.array([str(s) for s in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]])
    kf = KFold(n_splits=n)

    skf = StratifiedKFold(n_splits=2, shuffle=True)
    splits = list()
    for i, (train_index, test_index) in enumerate(kf.split(subjects)):
        split = {
            "train_fnames": [fn for fn in fnames if fn.split("_")[0] in subjects[train_index]],
            "test_fnames": list()
        }
        test_fns = sorted([fn for fn in fnames if fn.split("_")[0] in subjects[test_index]])
        for sub in subjects[test_index]:
            for j in range(3):
                curr_key = "{}_{}".format(sub, j)
                curr_fns = [fn for fn in test_fns if curr_key == "_".join(fn.split("_")[:2])]
                include_idx = int(len(curr_fns) * num_include)
                split["train_fnames"] += curr_fns[:include_idx]
                split["test_fnames"] += curr_fns[include_idx:]

        splits.append(split)
    
    split_record = {
        "splits": splits,
        "subjects": subjects
    }
    with open("data/splits_subjects_{}_{}".format(n, num_include), 'wb') as f:
        pickle.dump(split_record, f)

def time_split(n=10):
    subjects = np.array([str(s) for s in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]])
    fnames = np.array([fn for fn in os.listdir("data/samples") if fn[0] != "."])
    ys = np.array([int(fn.split('_')[1]) for fn in fnames])
    all_labels = [_ for _ in set(ys.tolist())]

    splits = list()
    for i in range(n // 2):
        splits.append(
            {
                "train_fnames": list(),
                "test_fnames": list()
            }
        )
    tscv = TimeSeriesSplit(n_splits=n)

    # split for each label
    for y in all_labels:
        # fetch all samples under current label
        curr_fnames = np.array([fnames[i] for i in range(len(fnames)) if ys[i] == y])

        # split fetched samples
        for i, (train_index, test_index) in enumerate(tscv.split(curr_fnames)):
            if i >= len(splits):
                break
            splits[i]["train_fnames"] += curr_fnames[train_index].tolist()
            splits[i]["test_fnames"] += curr_fnames[test_index].tolist()

    split_record = {
        "splits": splits,
        "subjects": subjects
    }
    with open("data/time_split", 'wb') as f:
        pickle.dump(split_record, f)

if __name__ == '__main__':
    # fnames = process_and_store_samples()
    split_5fold()
    # split_subjects(n=5, num_include=0.8)
    # time_split(n=10)