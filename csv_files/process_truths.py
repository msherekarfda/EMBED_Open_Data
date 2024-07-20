import pandas as pd
import numpy as np
from datetime import timedelta
import os
def process_exam_side(exams, cutoff_days, followup_days):
    """
    :param exams: a DataFrame containing examination data for a single patient.
    :param cutoff_days: the number of days for which
    :param followup_days: the number of days within which follow-up exams should be considered.
    :return: a DataFrame with two columns rad (results of a radiologist in form of a BIRADS score) &
                                          cancerous (if the results are actually cancerous or not)
    """

    # Handling Empty Data
    if exams.empty:
        return pd.DataFrame({'rad': [2], 'cancer': [2]}) # changed cancerous to cancer

    # Initial Dates and DataFrames
    initial_date = exams['study_date_anon_y'].iloc[0]  # Initial study date
    cutoff_date = initial_date + timedelta(days=cutoff_days)  # michelle had 500days
    cutoff_date_followup = initial_date + timedelta(days=followup_days)  # michelle had 180

    # sc -> screening & fu -> followup
    # choosing exam data for screening and follow-up exams
    exams_sc = df[(df['study_date_anon_y'] > initial_date) &
                  (df['study_date_anon_y'] <= cutoff_date) &
                  (df['patient_id'] == exams['patient_id'].iloc[0])]
    exams_fu = df[(df['study_date_anon_y'] >= initial_date) &
                  (df['study_date_anon_y'] <= cutoff_date_followup) &
                  (df['patient_id'] == exams['patient_id'].iloc[0])]

    # BIRADS notations
    # A -> Additional Evaluation, B -> Benign, N -> Negative

    # True Positive: if rad says A and there is a follow-up within given days with confirmed cancerous path-severity, TP
    if exams['asses'].str.contains('A').any() and exams_fu.empty == False and exams_fu['path_severity'].min() <= 1:
        rad = 1
        cancer = 1

    #False Positive: if rad says A and any follow-up exams within given days have non-cancerous path severity OR
    # all screening exams within 500 days have BIRADS N or B, FP
    elif exams['asses'].str.contains('A').any() and (
            (exams_fu.empty == False and exams_fu['path_severity'].min() >= 2) or # min>=2 has also to be varied
            (exams_sc.empty == False and exams_sc['asses'].astype(str).isin(['N', 'B']).all())):
        rad = 1
        cancer = 0

    #if rad says A and there are no more exams within 500 days, say nothing
    elif exams['asses'].str.contains('A').any() and exams_sc.empty == True:
        rad = 2
        cancer = 2

    # True Negative: if rad says N or B and all exams within 500 days have non-canerous path severity or N/B birads,
    elif exams['asses'].astype(str).isin(['N', 'B']).all() and exams_sc.empty == False and (
            exams_sc['path_severity'].min() >= 2 or exams_sc['asses'].astype(str).isin(['N', 'B']).all()):

        rad = 0
        cancer = 0

    # False Negative: if rad says N or B and some follow-up exam has cancerous path severity, FN
    elif exams['asses'].astype(str).isin(['N', 'B']).all() and exams_sc.empty == False and exams_fu[
        'path_severity'].min() <= 1:
        rad = 0
        cancer = 1
    else:
        rad = 2
        cancer = 2

    return pd.DataFrame({'rad': [rad], 'cancer': [cancer]}) # rad means radiologist


def process_patient_exams(patient_exams):
    result = pd.DataFrame(columns=['rad', 'cancer', 'patient_id', 'laterality'])

    for side in ['L', 'R']:
        side_exams = patient_exams[patient_exams['laterality'] == side]
        side_result = process_exam_side(side_exams, cutoff_days=120, followup_days=30) # change cutoff and follow up days here.

        if not side_result.empty:
            side_result['laterality'] = side
            side_result['patient_id'] = patient_exams.index[0]


df = pd.read_csv('data.csv', low_memory=False)
df['study_date_anon_x'] = pd.to_datetime(df['study_date_anon_x'])
df['study_date_anon_y'] = pd.to_datetime(df['study_date_anon_y'])
df_screen = df[df['desc'].str.contains('Screen', case=False)]
result = df_screen.groupby(['patient_id', 'study_date_anon_x']).apply(process_patient_exams)

# remove the 'patient_id' column before resetting the index
result.drop(columns=['patient_id'], inplace=True)
result.reset_index(inplace=True)

df_merged = pd.merge(df, result, on=('patient_id', 'study_date_anon_x', 'laterality'))

df_merged = df_merged[df_merged['rad'] != 2]
df_merged_dropped = df_merged[df_merged['desc'].str.contains('Screen', case=False)]

df_merged_dropped.reset_index(drop=True, inplace=True)

df_merged_dropped.to_csv('120co_30fu')  # co->cutoff & fu->follow up