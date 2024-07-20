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
process_exam_side(exams, cutoff_days=500, followup_days=180)