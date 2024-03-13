import numpy as np
import pandas as pd
import pickle

from sklearn.preprocessing import MinMaxScaler

if __name__ == "__main__":
    # read survey scores
    student_list = [4, 7, 8, 10, 14, 16, 17, 19, 22, 23, 24, 32, 33, 35, 36, 43, 44, 49, 51, 52, 53, 57, 58]
    df = pd.read_csv("scores.csv", index_col="student_id")

    # print(np.nan_to_num(np.array([df['pre_PHQ_9'][10]])))

    # form covariate vectors
    scores_matrix = list()
    for id_ in student_list:
        vec = list()
        for key in df:
            vec.append(df[key][id_])
        scores_matrix.append(vec)
    
    # pad nan, min_max normalization
    scores_matrix = np.nan_to_num(np.array(scores_matrix))
    scaler = MinMaxScaler()
    scaler.fit(scores_matrix)
    scores_matrix = scaler.transform(scores_matrix).tolist()
    
    # formatting
    scores_covariates = dict() # id -> scores_vectors
    for i in range(len(student_list)):
        scores_covariates[str(student_list[i])] = scores_matrix[i]
    
    # save the vectors
    with open('survey_scores_covariate', 'wb') as f:
        pickle.dump(scores_covariates, f)
    
    # check saved data
    scores_covariates = None
    with open('survey_scores_covariate', 'rb') as f:
        scores_covariates = pickle.load(f)
    
    for id_ in scores_covariates:
        print(id_, scores_covariates[id_])
