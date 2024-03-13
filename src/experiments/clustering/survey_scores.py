import pandas as pd
import numpy as np
import pickle

# ---------------------------------------------------------------- 
# Independent Study, Student Stree Prediction
#
# File_name: survey_scores
# Functionality: calculate the scores of surveys
# Author: Yunfei Luo
# Start date: EST May.14th.2020
# Last update: EST May.14th.2020
# ----------------------------------------------------------------

class survey_scores:
    def __init__(self, csv_file_path, neg_scale, pos_scale, pos_term, pre_col, post_col, filter_q):
        self.neg_scale = neg_scale
        self.pos_scale = pos_scale
        self.pos_term = pos_term
        self.scores = dict() # map: student_id -> {pre: score, post: score}
        df = pd.read_csv(csv_file_path)

        self.missing_data = list()

        # indexing questions
        self.questions = list()
        for i in range(len(df.columns[2:])):
            if i+1 in filter_q:
                self.questions.append(df.columns[2:][i])
                print(df.columns[2:][i])
        # indexing scores
        for uid in df["uid"]:
            key = int(uid[1:])
            # check if key exist
            if self.scores.get(key) != None:
                continue

            self.scores[key] = {"pre": -1, "post": -1}

            # calc scores
            rows = df.loc[df['uid'] == uid]
            row1 = rows.loc[rows['type'] == "pre"]
            row2 = rows.loc[rows['type'] == "post"]

            # if len(rows[self.questions[0]]) == 1:
            #     self.missing_data.append(key)
            #     continue
            
            pre_score = 0
            post_score = 0
            for i in range(len(self.questions)):
                pos = False
                if i+1 in self.pos_term:
                    pos = True
                
                if pre_score != None:
                    cum = -1
                    try:
                        # calc pre score
                        for j in row1[self.questions[i]]:
                            cum += 1
                            if pos:
                                pre_score += self.pos_scale[j]
                            else:
                                pre_score += self.neg_scale[j]
                    except:
                        pre_score = None
                        self.missing_data.append(key)
                    if cum < 0:
                        pre_score = None
                        self.missing_data.append(key)

                if post_score != None:
                    cum = -1
                    try:
                        # calc post score
                        for j in row2[self.questions[i]]:
                            cum += 1
                            if pos:
                                post_score += self.pos_scale[j]
                            else:
                                post_score += self.neg_scale[j]
                    except:
                        post_score = None
                        self.missing_data.append(key)
                    if cum < 0:
                        post_score = None
                        self.missing_data.append(key)

            # store scores
            self.scores[key]['pre'] = pre_score
            self.scores[key]['post'] = post_score
        
        # # check
        # for key in self.scores:
        #     print("pre score of {} is {}".format(key, self.scores[key]["pre"]))
        #     print("post score of {} is {}".format(key, self.scores[key]["post"]))
        #     print(' ')
        
        # # write to a new file
        # df_out = {pre_col: list(), post_col: list()}
        # ind = list()
        # for key in self.scores:
        #     ind.append(key)
        #     df_out[pre_col].append(self.scores[key]["pre"])
        #     df_out[post_col].append(self.scores[key]["post"])
        # df_out = pd.DataFrame(df_out, index=ind)
        # df_out.index.name = "student_id"
        
        # write to a exist file        
        df_out = pd.read_csv("src/experiments/clustering/survey/scores.csv", index_col="student_id")
        pre_scores = list()
        post_scores = list()
        has = list()
        for i, row in df_out.iterrows():
            try:
                pre_scores.append(self.scores[i]['pre'])
            except:
                pre_scores.append(None)
            try:
                post_scores.append(self.scores[i]['post'])
            except:
                post_scores.append(None)
            has.append(i)
        df_out[pre_col] = pre_scores
        df_out[post_col] = post_scores

        # add a new row
        new = list()
        for key in self.scores:
            if key not in has:
                new.append(key)
        print('new', new)
        for id_ in new:
            new_row = list()
            for i in df_out.columns:
                if i == pre_col:
                    new_row.append(self.scores[key]["pre"])
                elif i == post_col:
                    new_row.append(self.scores[key]["post"])
                else:
                    new_row.append(None)
            df_out.loc[id_] = new_row
        #print(df_out)

        self.missing_data = [i for i in set(self.missing_data)]
        print('missing data', self.missing_data)
        df_out.to_csv('src/experiments/clustering/survey/scores.csv')

def calc_PSS():
    neg_scale = {"Never": 0, "Almost never": 1, "Sometime": 2, "Fairly often": 3, "Very often": 4}
    pos_scale = {"Never": 4, "Almost never": 3, "Sometime": 2, "Fairly often": 1, "Very often": 0}
    pos_term = [4,5,7,8]
    csv_file_path = "src/experiments/clustering/survey/PerceivedStressScale.csv"
    PSS_score = survey_scores(csv_file_path, neg_scale, pos_scale, pos_term, "pre_PSS", "post_PSS")

def calc_PHQ_9():
    neg_scale = {"Extremely difficult": 0, "Very difficult":0, "Somewhat difficult": 0, "Not difficult at all": 0}
    pos_scale = {"Not at all": 0, "Several days": 1, "More than half the days": 2, "Nearly every day": 3}
    pos_term = [i for i in range(10)]
    csv_file_path = "src/experiments/clustering/survey/PHQ-9.csv"
    PSS_score = survey_scores(csv_file_path, neg_scale, pos_scale, pos_term, "pre_PHQ_9", "post_PHQ_9")

def calc_lonliness_scale():
    neg_scale = {}
    pos_scale = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3}
    pos_term = [i for i in range(21)]
    csv_file_path = "src/experiments/clustering/survey/LonelinessScale.csv"
    PSS_score = survey_scores(csv_file_path, neg_scale, pos_scale, pos_term, "pre_lonliness_scale", "post_longliness_scale")

def calc_flourishing_scale():
    neg_scale = {}
    pos_scale = {}
    for i in range(8):
        pos_scale[i+1] = i+1
    pos_term = [i for i in range(9)]
    csv_file_path = "src/experiments/clustering/survey/FlourishingScale.csv"
    PSS_score = survey_scores(csv_file_path, neg_scale, pos_scale, pos_term, "pre_flourishing_scale", "post_flourishing_scale")

def calc_panas():
    neg_scale = {}
    pos_scale = {}
    for i in range(5):
        pos_scale[i+1] = i+1
        neg_scale[i+1] = 0
    # filter_q = [1, 3, 5, 9, 10, 12, 14, 16, 17, 19]
    # pos_term = [1, 3, 5, 9, 10, 12, 14, 16, 17, 19]
    filter_q = [2, 4, 6, 7, 8, 11, 13, 15, 18, 20]
    pos_term = [2, 4, 6, 7, 8, 11, 13, 15, 18, 20]
    csv_file_path = "src/experiments/clustering/survey/panas.csv"
    PSS_score = survey_scores(csv_file_path, neg_scale, pos_scale, pos_term, "pre_panas_negative", "post_panas_negative", filter_q)

def calc_big_five():
    neg_scale = {"Disagree Strongly": 5, "Disagree a little": 4, "Neither agree nor disagree": 3, "Agree a little": 2, "Agree strongly": 1}
    pos_scale = {"Disagree Strongly": 1, "Disagree a little": 2, "Neither agree nor disagree": 3, "Agree a little": 4, "Agree strongly": 5}
    # filter_q = [1, 6, 11, 16, 21, 26, 31, 36]
    # pos_term = [1, 11, 16, 26, 36]
    # filter_q = [2, 7, 12, 17, 22, 27, 32, 37, 42]
    # pos_term = [7, 17, 22, 32, 42]
    # filter_q = [3, 8, 13, 18, 23, 28, 33, 38, 43]
    # pos_term = [3, 13, 28, 33, 38]
    # filter_q = [4, 9, 14, 19, 24, 29, 34, 39]
    # pos_term = [4, 14, 19, 29, 39] 
    filter_q = [5, 10, 15, 20, 25, 30, 35, 40, 41, 44]
    pos_term = [5, 10, 15, 20, 25, 30, 40, 44]
    csv_file_path = "src/experiments/clustering/survey/BigFive.csv"
    PSS_score = survey_scores(csv_file_path, neg_scale, pos_scale, pos_term, "O_pre", "O_post", filter_q)

if __name__ == "__main__":
    # calc_PHQ_9()
    # calc_PSS()
    # calc_lonliness_scale()
    # calc_flourishing_scale()
    # calc_panas()
    # calc_big_five()

    '''
    group form: map: student_id -> group_id
    '''
    from sklearn.cluster import KMeans
    student_list = [4, 7, 8, 10, 14, 16, 17, 19, 22, 23, 24, 32, 33, 35, 36, 43, 44, 49, 51, 52, 53, 57, 58]
    df = pd.read_csv("src/experiments/clustering/survey/scores_pre.csv", index_col="student_id")
    student_survey_scores = dict()
    missing_students = list()
    for i, row in df.iterrows():
        if i not in student_list:
            continue
        curr = list()
        na = 0
        for j in row:
            if "n" in str(j): 
                curr.append(-1)
                na += 1
            else:
                curr.append(j)
        student_survey_scores[i] = curr
        if na > 0:
            missing_students.append(i)

    print("missing students", missing_students)
    ms = np.array([student_survey_scores[i] for i in student_survey_scores if i in missing_students])

    num_groups = 4
    # kmeans on clean dataset
    X = np.array([student_survey_scores[i] for i in student_survey_scores if i not in missing_students])
    kmeans = KMeans(n_clusters=num_groups, random_state=0).fit(X)

    # store group info
    student_group = dict() # map: student_id -> group
    for i in student_list:
        if i in missing_students:
            continue
        student_group[i] = kmeans.predict([np.array(student_survey_scores[i])])[0]
    
    # clustering the missing students, to the closest center
    def clustering_missing_students(centers, missing_students, ms, student_group):
        for i in range(len(ms)):
            student_vec = [num for num in ms[i] if num >= 0]
            close_ind = -1
            close_dist = np.inf
            for j in range(len(centers)):
                center_vec = [centers[j][ind] for ind in range(len(ms[i])) if ms[i][ind] >= 0]
                dist = np.linalg.norm(np.array(center_vec) - np.array(student_vec))
                if dist < close_dist:
                    close_dist = dist
                    close_ind = j

            student_group[missing_students[i]] = close_ind
    
    clustering_missing_students(kmeans.cluster_centers_, missing_students, ms, student_group)
    
    # check
    group_student = dict() # map group -> list of student
    for student in student_group:
        try:
            group_student[student_group[student]].append(student)
        except:
            group_student[student_group[student]] = [student]
    for group in group_student:
        print(group_student[group])
    
    # formalize
    output_group = dict() # copy student_group with correct form
    for i in student_group:
        output_group['student_{}'.format(i)] = "group_{}".format(student_group[i])
    
    # write to file
    with open("src/experiments/clustering/student_groups/pre_survey_scores_{}.pkl".format(num_groups), 'wb') as f:
        pickle.dump(output_group, f)
    
    # check
    with open("src/experiments/clustering/student_groups/pre_survey_scores_{}.pkl".format(num_groups), 'rb') as f:
        readed_file = pickle.load(f)
        for stu in readed_file:
            print("{}, {}".format(stu, readed_file[stu]))

    
    # check missing data
    # missing_ids = list()
    # #df = pd.read_csv("src/experiments/clustering/survey/scores.csv", index_col="student_id")
    # df = pd.read_csv("src/experiments/clustering/survey/BigFive.csv", index_col="uid")
    # df = df.isna()
    # for i, row in df.iterrows():
    #     for j in row:
    #         if j:
    #             missing_ids.append(i)
    # #print(df)
    # missing_ids = [i for i in set(missing_ids)]
    # print("missing ids", missing_ids)
