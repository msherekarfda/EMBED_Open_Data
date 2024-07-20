import pandas as pd
import numpy as np
from datetime import timedelta
import os
# Define paths
base_dir = os.path.dirname(__file__)
csv_dir = os.path.join(base_dir, '..', 'csv_files')

cutoff_days = [500, 365, 180, 120]
followup_days = [180, 120, 90, 60]

for cutoff_day in cutoff_days:
    for followup_day in followup_days:
        def process_exam_side(exams):
            if exams.empty:
                return pd.DataFrame({'rad': [2], 'cancer': [2]})

            initial_date = exams['study_date_anon_y'].iloc[0]
            cutoff_date = initial_date + timedelta(days=cutoff_day)
            cutoff_date_followup = initial_date + timedelta(days=followup_day)
            exams_sc = df[(df['study_date_anon_y'] > initial_date) & (df['study_date_anon_y'] <= cutoff_date) & (df['patient_id'] == exams['patient_id'].iloc[0])]
            exams_fu = df[(df['study_date_anon_y'] >= initial_date) & (df['study_date_anon_y'] <= cutoff_date_followup) & (df['patient_id'] == exams['patient_id'].iloc[0])]

            #if rad says A and there is a follow-up within 180 days with confirmed cancerous path-severity, TP
            if exams['asses'].str.contains('A').any() and  exams_fu.empty == False and exams_fu['path_severity'].min() <= 1:
                rad = 1
                cancer = 1
            #if rad says A and any follow-up exams within 180 days have non-cancerous path severity OR all screening exams within 500 days have BIRADS N or B, FP
            elif exams['asses'].str.contains('A').any() and ((exams_fu.empty == False and exams_fu['path_severity'].min() >=2) or (exams_sc.empty == False and exams_sc['asses'].astype(str).isin(['N', 'B']).all())):
                rad = 1
                cancer = 0
            #if rad says A and there are no more exams within 500 days, say nothing
            elif exams['asses'].str.contains('A').any() and exams_sc.empty == True:
                rad = 2
                cancer = 2
            # if rad says N or B and all exams within 500 days have non-canerous path severity or N/B birads, true negative
            elif exams['asses'].astype(str).isin(['N', 'B']).all() and exams_sc.empty == False and (exams_sc['path_severity'].min() >= 2 or exams_sc['asses'].astype(str).isin(['N', 'B']).all()):
            # Your code here
                rad = 0
                cancer = 0
            #if rad says N or B and some follow-up exam has canceorus path severity, FN
            elif exams['asses'].astype(str).isin(['N', 'B']).all() and exams_sc.empty == False and exams_fu['path_severity'].min() <= 1:
                rad = 0
                cancer = 1
            else:
                rad = 2
                cancer = 2

            return pd.DataFrame({'rad': [rad], 'cancer': [cancer]})

        def process_patient_exams(patient_exams):
            result = pd.DataFrame(columns=['rad', 'cancer', 'patient_id', 'laterality'])

            for side in ['L', 'R']:
                side_exams = patient_exams[patient_exams['laterality'] == side]
                side_result = process_exam_side(side_exams)

                if not side_result.empty:
                    side_result['laterality'] = side
                    side_result['patient_id'] = patient_exams.index[0]
                    result = pd.concat([result, side_result], ignore_index=True)
            #print(result)
            return result


        df = pd.read_csv('master_list_for_split.csv')
        df['study_date_anon_x'] = pd.to_datetime(df['study_date_anon_x'])
        df['study_date_anon_y'] = pd.to_datetime(df['study_date_anon_y'])
        df_screen = df[df['desc'].str.contains('Screen', case=False)]
        result = df_screen.groupby(['patient_id', 'study_date_anon_x']).apply(process_patient_exams)
        # remove the 'patient_id' column before resetting the index
        result.drop(columns=['patient_id'], inplace=True)
        result.reset_index(inplace=True)

        # Perform the merge
        df_merged = pd.merge(df, result, on=('patient_id', 'study_date_anon_x', 'laterality'))

        # Drop rows where 'rad' equals 2
        df_merged = df_merged[df_merged['rad'] != 2]

        # Drop rows that contain 'Screen' in the 'desc' column
        df_merged_dropped = df_merged[df_merged['desc'].str.contains('Screen', case=False)]

        # Reset the index
        df_merged_dropped.reset_index(drop=True, inplace=True)

        # Drop the 'cancer_x' column and rename 'cancer_y' to 'cancer'
        df_merged_dropped = df_merged_dropped.drop(columns=['cancer_x']).rename(columns={'cancer_y': 'cancer'})

        output_file = os.path.join(csv_dir, f'embed_{cutoff_day}co_{followup_day}fu.csv')


        # Save the DataFrame to a CSV file
        #df_merged_dropped.to_csv('365co_180fu.csv', index=False)
        df_merged.to_csv(output_file, index=False)

