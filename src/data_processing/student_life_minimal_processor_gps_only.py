"""
Script to generate minimally processed raw data.
"""
import datetime
import os
import numpy as np
import pandas as pd

from pathlib import Path
# from src.data_processing.query_generator import get_feature_query_for_student, get_stress_query_for_student
# from src.data_processing.query_processor import exec_sql_query

# Collecting distinct students.
# distinct_students = exec_sql_query("select distinct student_id from stress_details")
# distinct_students = distinct_students.values.T.tolist()
# distinct_students = distinct_students[0]
# distinct_students.sort()
distinct_students = [int(p.split('_')[1].split('.')[0][1:]) for p in os.listdir("data/dataset/sensing/gps")]

# getting current working directory for creating directories later.
cwd = Path(os.getcwd())

print("Students: ", distinct_students)

# adj_stress = {
#     "1": "0",
#     "2": "0",
#     "3": "1",
#     "4": "2",
#     "5": "2",
# }

for student in distinct_students:

    newpath = Path(r'data\student_life_minimal_processed_data\student_' + str(student))
    newpath = os.path.join(cwd, newpath)

    if not os.path.exists(newpath):
        os.makedirs(newpath)

    # Getting Stress levels for only student_id = 1. This will be merged with other features.
    # stress_details_raw = exec_sql_query(get_stress_query_for_student(student))
    # stress_details = stress_details_raw.loc[:, ["response_time", "student_id", "stress_level"]]

    stress_details = pd.read_json('data/dataset/EMA/response/Stress/Stress_u{}.json'.format(student))
    stress_details = stress_details.sort_values(by="resp_time")
    stress_details.rename({"resp_time": "time"}, axis='columns', inplace=True)
    stress_details.rename({"level": "stress_level"}, axis='columns', inplace=True)
    # try:
    #     for i in range(len(stress_details["stress_level"])):
    #         try:
    #             stress_details["stress_level"][i] = adj_stress[stress_details["stress_level"][i]]
    #         except:
    #             continue
    # except:
    #     print(student)

    # Extracting first and last index of stress level.
    # We will truncate other features 1 day behind and 1 day ahead.
    first_date = stress_details.loc[0, 'time']
    last_date = stress_details.loc[len(stress_details) - 1, 'time']

    # delta to back and ahead, in days.
    first_date = first_date - datetime.timedelta(days=1)
    last_date = last_date + datetime.timedelta(days=0)

    # for i in range(len(stress_details['time'])):
    #     stress_details['time'][i] = int(round(stress_details['time'][i].timestamp()))
    # first_date = int(round(first_date.timestamp()))
    # last_date = int(round(last_date.timestamp()))

    # feature_map = get_feature_query_for_student(student)
    feature_map = {"gps_details": "select wifi_timestamp as time, student_id, latitude, longitude from gps_details "}

    for key in feature_map.keys():
        feature_query = feature_map[key]
        # Data processing begins..
        # feature_data = exec_sql_query(feature_query)
        feature_data = pd.read_csv("data/dataset/sensing/gps/gps_u{}.csv".format(student), index_col=False)

        # Selecting Time Col and renaming time column from *_time to time.
        train_col_list = []
        # for col in feature_data.columns:
        #     if "time" in col:
        #         time_column = col
        #     else:
        #         train_col_list.append(col)
        feature_data.rename({'time': "time"}, axis='columns', inplace=True)
        time_column = "time"
        feature_data["time"] = [datetime.datetime.fromtimestamp(t) for t in feature_data["time"]]

        # Sorting by values of time.
        feature_data = feature_data.sort_values(by='time')

        # Truncating extra features that do not lie in the time frame.
        feature_data = feature_data[
            np.logical_and(feature_data[time_column] > first_date, feature_data[time_column] < last_date)
        ]

        if feature_data.empty:
            print("Empty DataFrame for Student {} for feature {}".format(student, key))
            continue

        # Writing Feature Data.
        feature_data_file_name = os.path.join(newpath, key+".csv")
        feature_data.to_csv(feature_data_file_name, index=False, header=True)

    # Writing Stress Data.
    stress_data_file_name = os.path.join(newpath, "stress_details.csv")
    stress_details.to_csv(stress_data_file_name, index=False)
