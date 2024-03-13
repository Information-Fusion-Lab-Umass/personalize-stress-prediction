import os
import pandas as pd
import numpy as np

from copy import deepcopy
from src.data_processing import imputation
from src import definitions

processed_deadlines_path = os.path.join(definitions.SURVEYS_AND_COVARIATES_DATA_PATH,
                                        "processed_student_deadlines.csv")
PROCESSED_DEADLINES = pd.read_csv(processed_deadlines_path, index_col=[0])
PROCESSED_DEADLINES.index = pd.to_datetime(PROCESSED_DEADLINES.index)


def day_of_week(flattened_student_data: pd.DataFrame) -> pd.DataFrame:
    """

    @param flattened_student_data: Flattened student data in form of Data Frame.
    @return: Day of week (integer value from 0-6) as a feature in separate column.
    """
    flattened_student_data.insert(loc=1, column='day_of_week', value=flattened_student_data.index.dayofweek)

    return flattened_student_data


def epoch_of_day(flattened_student_data: pd.DataFrame) -> pd.DataFrame:
    """
    @brief: Day has been defined into 4 epoch of 6 hours each.
    @param flattened_student_data: Flattened student data in form of Data Frame.
    @return: Epoch of day (integer value from 0-3) as a feature in separate column.
    """

    flattened_student_data.insert(loc=2, column='epoch_of_day', value=flattened_student_data.index.hour)
    flattened_student_data['epoch_of_day'] = flattened_student_data['epoch_of_day'].map(evaluate_epoch)
    return flattened_student_data


def evaluate_epoch(hour):
    if 0 <= hour < 6:
        return 0
    if 6 <= hour < 12:
        return 1
    if 12 <= hour < 18:
        return 2
    else:
        return 3


def time_since_last_label_min(flattened_student_data: pd.DataFrame) -> pd.DataFrame:
    null_mask = flattened_student_data['stress_level_mode'].isnull()
    flattened_student_data.insert(loc=3, column='time_since_last_label', value=deepcopy(flattened_student_data.index))
    flattened_student_data.loc[null_mask, 'time_since_last_label'] = np.nan
    flattened_student_data['time_since_last_label'].fillna(method='ffill', inplace=True)
    # Filling the sequences which do not have a last label and appear first in the data set.
    flattened_student_data['time_since_last_label'] = (flattened_student_data.index - flattened_student_data[
        'time_since_last_label']).astype('timedelta64[m]')

    return flattened_student_data


def time_to_next_label_min(flattened_student_data: pd.DataFrame) -> pd.DataFrame:
    null_mask = flattened_student_data['stress_level_mode'].isnull()
    flattened_student_data.insert(loc=4, column='time_to_next_label', value=deepcopy(flattened_student_data.index))
    flattened_student_data.loc[null_mask, 'time_to_next_label'] = np.nan
    flattened_student_data['time_to_next_label'].fillna(method='bfill', inplace=True)
    flattened_student_data['time_to_next_label'] = (flattened_student_data[
                                                        'time_to_next_label']
                                                    - flattened_student_data.index).astype('timedelta64[m]')

    return flattened_student_data


def previous_stress_label(flattened_student_data: pd.DataFrame) -> pd.DataFrame:
    flattened_student_data.insert(loc=5,
                                  column='previous_stress_label',
                                  value=flattened_student_data.iloc[:, -1])
    flattened_student_data['previous_stress_label'] = imputation.forward_fill(
        flattened_student_data['previous_stress_label'])

    return flattened_student_data


def time_to_next_deadline(flattened_student_data: pd.DataFrame) -> pd.DataFrame:
    student_id = flattened_student_data['student_id'].values[0]
    deadlines = PROCESSED_DEADLINES.loc[:, str(student_id)]
    deadlines.rename("deadlines", inplace=True)
    flattened_student_data = flattened_student_data.join(deadlines, how='left', sort=True)
    flattened_student_data.insert(loc=6,
                                  column='time_to_next_deadline',
                                  value=deepcopy(flattened_student_data.index))
    null_mask = np.logical_not(flattened_student_data['deadlines'] >= 1)
    flattened_student_data.loc[null_mask, 'time_to_next_deadline'] = np.nan
    flattened_student_data['time_to_next_deadline'].fillna(method='bfill', inplace=True)
    flattened_student_data['time_to_next_deadline'] = (flattened_student_data[
                                                           'time_to_next_deadline']
                                                       - flattened_student_data.index).astype('timedelta64[m]')

    return flattened_student_data.drop(columns='deadlines')


def exam_period(flattened_student_data: pd.DataFrame) -> pd.DataFrame:
    flattened_student_data.insert(loc=0,
                                  column='exam_period',
                                  value=deepcopy(flattened_student_data.index))

    non_exam_mask = (flattened_student_data['exam_period'] < definitions.MIDTERM_START_DATE) | (
            flattened_student_data['exam_period'] > definitions.MIDTERM_END_DATE)

    flattened_student_data['exam_period_inferred'] = 1
    flattened_student_data.loc[non_exam_mask, 'exam_period_inferred'] = 0

    return flattened_student_data.drop(columns='exam_period')


def evaluate_gender(flattened_student_data: pd.DataFrame) -> pd.DataFrame:
    # todo(abhinavshaw): Implement this.
    return
