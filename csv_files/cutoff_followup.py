import os
import pandas as pd
from datetime import timedelta

def process_exam_side(exams, cutoff_days, followup_days):
    """
    :param exams: a DataFrame containing examination data for a single patient.
    :param cutoff_days: the number of days for which
    :param followup_days: the number of days within which follow-up exams should be considered.
    :return: a DataFrame filtered based on cutoff and follow-up days
    """

    # Handling Empty Data
    if exams.empty:
        return exams

    # Initial Dates and DataFrames
    initial_date = exams['study_date_anon_y'].iloc[0]  # Initial study date
    cutoff_date = initial_date + timedelta(days=cutoff_days)
    cutoff_date_followup = initial_date + timedelta(days=followup_days)

    # choosing exam data for screening and follow-up exams
    exams_filtered = exams[(exams['study_date_anon_y'] >= initial_date) &
                           (exams['study_date_anon_y'] <= cutoff_date_followup)]

    return exams_filtered

def process_patient_exams(patient_exams, cutoff_days, followup_days):
    result = pd.DataFrame()

    for side in ['L', 'R']:
        side_exams = patient_exams[patient_exams['laterality'] == side]
        side_result = process_exam_side(side_exams, cutoff_days=cutoff_days, followup_days=followup_days)

        if not side_result.empty:
            side_result['laterality'] = side
            side_result['patient_id'] = patient_exams.index[0]
            result = pd.concat([result, side_result], ignore_index=True)

    return result

# Define paths
base_dir = os.path.dirname(__file__)
csv_dir = os.path.join(base_dir, '..', 'csv_files')

# Read the master list
df = pd.read_csv(os.path.join(csv_dir, 'train.csv'), low_memory=False)
df['study_date_anon_x'] = pd.to_datetime(df['study_date_anon_x'])
df['study_date_anon_y'] = pd.to_datetime(df['study_date_anon_y'])
df_screen = df[df['desc'].str.contains('Screen', case=False)]

#cutoff_days_list = [500, 365, 180, 120, 90, 60, 30]
#followup_days_list = [180, 150, 120, 90, 60, 30]
#cutoff_days_list = [500]
#followup_days_list = [180]


for cutoff_days in cutoff_days_list:
    for followup_days in followup_days_list:
        result = df_screen.groupby(['patient_id', 'study_date_anon_x']).apply(process_patient_exams, cutoff_days, followup_days)
        result = result.droplevel([0, 1]).reset_index(drop=True)

        df_merged = pd.merge(df, result, on=('patient_id', 'study_date_anon_x', 'laterality'))
        df_merged.reset_index(drop=True, inplace=True)

        # Define the file name
        output_file = f'embed_{cutoff_days}co_{followup_days}fu.csv'

        # Save to CSV
        df_merged.to_csv(os.path.join(csv_dir, output_file), index=False)
