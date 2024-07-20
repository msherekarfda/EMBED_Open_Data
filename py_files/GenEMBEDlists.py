# This script is meant to format data by combining ??
import pandas as pd
import numpy as np
import re
import pydicom
import pylibjpeg
import matplotlib.patches as patches
from matplotlib import pyplot as plt
import os

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 200)

def stats(df): #the function displays dataframe size, countings of unique patients and unique exams
    print('Dataframe size: ' + str(df.shape))
    try:
        print('# patients: ' + str(df.empi_anon.nunique()))
    except:
        print('# patients: ' + str(df.empi_anon_x.nunique()))
    print('# exams: ' + str(df.acc_anon.nunique()))

 # # Parameters
# # -----------------------------------------
basepath = "/projects01/didsr-aiml/common_data/EMBED/" #define the file directory
clinical_csv_file = os.path.join(basepath, "tables/EMBED_OpenData_clinical.csv")
meta_csv_file = os.path.join(basepath, "tables/EMBED_OpenData_metadata.csv")
# # 
default_path_string = "/mnt/NAS2/mammo/anon_dicom/"
replacement_path_string = "/projects01/didsr-aiml/common_data/EMBED/images/"
# #
out_merged_csv_loc = "/projects01/didsr-aiml/michelle.mastrianni/EMBED/"

# # load in clinical and metadata dataframes

#Load clinical and filter for fields needed for the tasks we are showcasing in this notebook
df_clinical = pd.read_csv(clinical_csv_file)
df_clinical = df_clinical[['empi_anon','acc_anon','study_date_anon','asses', 'tissueden',
                          'desc','side','path_severity','procdate_anon','numfind','total_L_find','total_R_find',
                          'massshape','massmargin','massdens','calcfind','calcdistri','calcnumber',
                          'ETHNICITY_DESC','ETHNIC_GROUP_DESC','age_at_study','ETHNIC_GROUP_DESC', 'GENDER_DESC', 'age_at_study', 'MARITAL_STATUS_DESC', 'first_3_zip']]
stats(df_clinical)

#Load metadata and filter for fields needed for the tasks we are showcasing in this notebook
df_metadata = pd.read_csv(meta_csv_file)
df_metadata = df_metadata[['anon_dicom_path','empi_anon','acc_anon','study_date_anon',
                            'StudyDescription','SeriesDescription','FinalImageType',
                            'ImageLateralityFinal','ViewPosition','spot_mag','ROI_coords','num_roi']]

stats(df_metadata)

# # create screening, diagnostic, screening+diagnostic
# #
#filter for screening exams only
#df_clinical_screen = df_clinical.loc[df_clinical.desc.str.contains('screen',case=False)]
#stats(df_clinical_screen)
df_clinical_screen = df_clinical

# # Type of image acquisition
# # 2D, cview, 2D+cview
df_metadata_2D = df_metadata[df_metadata.FinalImageType == "2D"]
stats(df_metadata_2D)


# Merging metadata and clinical data on exam ID (acc_anon). This will link the clinical data to the file list from metadata
df_merge_screen_2D = pd.merge(df_metadata_2D, df_clinical_screen, on=['acc_anon'])

# The 'side' column in the clinical data represents the laterality of the finding in that row, and can be 
#L (left), R (right), B (bilateral), or NaN (when there is no finding). Therefore when merging clinical 
#and metadata, we must first match by exam ID and then match the laterality of the clinical finding (side) 
#to the laterality of the image (ImageLateralityFinal)/ Side "B" and "NaN" can be matched to 
#ImageLateralityFinal both "L" and "R"

df_merge_screen_2D = df_merge_screen_2D.loc[
    (df_merge_screen_2D.side == df_merge_screen_2D.ImageLateralityFinal) 
    | (df_merge_screen_2D.side == 'B') | (pd.isna(df_merge_screen_2D.side))] 

stats(df_merge_screen_2D)


# # decide what to save
# # option 1
df_to_save = df_merge_screen_2D.copy()
out_csv_list =  "/projects01/didsr-aiml/michelle.mastrianni/EMBED/master_list_with_more_negatives.csv"
out_merged_csv_list = "/projects01/didsr-aiml/michelle.mastrianni/EMBED/master_list_for_split.csv"
# # 
df_to_save['anon_dicom_path'][0]

# # Match EMBED to RSNA format for column names
df_to_save['site_id'] = 1
df_to_save['machine_id'] = 1
df_to_save['cancer'] = df_to_save['path_severity'] 
asses_dict = {0: 1,
              1: 1,
              2: 0,
              3: 0,
              4: 0,
              5: 0,
              6: 0}
df_to_save['cancer'].replace(asses_dict, inplace=True)
df_to_save['cohort_num'] = df_to_save['anon_dicom_path'].apply(lambda x: x.split(os.sep)[5][-1])
df_to_save['full_dicom_path'] = df_to_save['anon_dicom_path'].str.replace(default_path_string, replacement_path_string)
df_to_save['anon_dicom_path'] = df_to_save['anon_dicom_path'].str.replace(default_path_string, '')
# df_to_save['anon_dicom_path'] = df_to_save['anon_dicom_path'].apply(lambda x: x.replace(x, os.path.join(*x.split(os.sep)[2:])))
df_to_save['anon_dicom_path'] = df_to_save['anon_dicom_path'].apply(lambda x: x.replace('.dcm', ''))
# #
# # check if the paths in the list exists
df_to_save['file_exists'] = df_to_save['full_dicom_path'].astype(str).map(os.path.exists)
df_to_save_checked = df_to_save.loc[df_to_save['file_exists'] == True]
stats(df_to_save_checked)

# # rename columns
df_to_save_checked = df_to_save_checked.rename(columns={'empi_anon_x': 'patient_id', 
                        'anon_dicom_path': 'image_id',
                        'ImageLateralityFinal': 'laterality'})

#drop procdates associated to screening exam that are before the last procdate 
df_to_save_checked = df_to_save_checked.sort_values(by='procdate_anon',ascending=False) #ADDED LINE
df_to_save_checked['image_id_string'] = df_to_save_checked['image_id'].astype(str)
df_to_save_checked = df_to_save_checked.drop_duplicates(subset='image_id_string', keep='first') #ADDED LINE

# df_to_save['anon_dicom_path'][0]
# df_to_save['cohort_num'][0]
#df_to_save_checked['image_id'][0]


# # save to csv
#df_to_save.to_csv(os.path.join(out_csv_list), index=False)
df_to_save_checked.to_csv(os.path.join(out_merged_csv_list), index=False)


#ADDING MORE NEGATIVES. COMMENT OUT IF WANT TO SPLIT THIS TASK
# Assuming 'patient_id' is the column identifying patients
# Replace it with the actual column name if different
df_sorted = df_to_save_checked.sort_values(by=['patient_id', 'study_date_anon_y'])

# Convert 'study_date_anon' to datetime format
df_sorted['study_date_anon_y'] = pd.to_datetime(df_sorted['study_date_anon_y'], format='%Y-%m-%d')

# Iterate through each row
for idx, row in df_sorted.iterrows():
    patient_id = row['patient_id']
    study_date = row['study_date_anon_y']
    asses = row['asses']
    path_severity = row['path_severity']
    
    # Find the next row for the same patient
    next_rows = df_sorted[(df_sorted['patient_id'] == patient_id) & (df_sorted['study_date_anon_y'] > study_date)]
    
    # If there is no next row
    if next_rows.empty and (asses in ['N','B']):
        df_sorted.at[idx, 'cancer'] = 0
    elif not next_rows.empty:
        next_row = next_rows.head(1)
        next_date = next_row['study_date_anon_y'].values[0]
        days_difference = (next_date - study_date).days
        next_asses = next_row['asses'].values[0]
        # If next exam is within 635 days and asses = 'N' or 'B' and path_severity is blank or low
        if days_difference <= 635 and (next_asses in ['N', 'B']) and (pd.isna(path_severity) or path_severity >= 4):
            df_sorted.at[idx, 'cancer'] = 0

# Drop rows with blank 'cancer' value and remove duplicate rows
df_sorted.dropna(subset=['cancer'], inplace=True)
df_sorted.drop_duplicates(inplace=True)

# Now df_sorted contains only rows with non-blank 'cancer' values and no duplicates

df_sorted.to_csv(os.path.join(out_csv_list), index=False)  # index=False to avoid saving the index as a column