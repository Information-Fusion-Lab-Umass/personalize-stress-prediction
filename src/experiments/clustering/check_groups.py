# ---------------------------------------------------------------- 
# Independent Study 496, Student Stree Prediction
#
# file_name: clustering_students.py
# Functionality: clustering students, 
#    return dict: map: group_ids(str) -> list_of_student_id(list(str))
# Author: Yunfei Luo
# Start date: EST Feb.22th.2020
# Last update: EST Feb.27th.2020
# ----------------------------------------------------------------

import os
import sys
from src.utils.read_utils import read_pickle

if __name__ == '__main__':
    method = None
    try:
        method = sys.argv[1]
    except:
        method = 'one_for_each'
    #student_groups = clustering(student_list, data['data'], method)

    groups_file_path = 'src/experiments/clustering/student_groups/' + method + '.pkl'
    print('get group file from: ' + groups_file_path)
    student_groups = read_pickle(groups_file_path) 
    
    # check how students are distributed
    print("student distribution: ")
    rev_groups = dict()
    for student in student_groups:
        if rev_groups.get(student_groups[student]) != None:
            rev_groups[student_groups[student]].append(student)
        else:
            rev_groups[student_groups[student]] = [student]
    for group in rev_groups:
        print(group + ': ' + str(rev_groups[group]))
    