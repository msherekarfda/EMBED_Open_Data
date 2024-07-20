# import pandas as pd
# from sklearn.model_selection import train_test_split
#
#
# def split_patient_data(data, train_size, val_size, test_size, patient_id_col='patient_id', output_prefix=''):
#     # Check that the sizes sum to 1.0
#     assert train_size + val_size + test_size == 1.0, "Train, validation, and test sizes must sum to 1.0"
#
#     # Read CSV if data is a string (file path)
#     if isinstance(data, str):
#         data = pd.read_csv(data)
#
#     # Extract unique patient IDs
#     unique_patients = data[patient_id_col].unique()
#
#     # Split unique patients into training and temp (validation + testing)
#     train_patients, temp_patients = train_test_split(unique_patients, train_size=train_size, random_state=42)
#
#     # Determine the proportion of validation and testing within the temp set
#     val_temp_size = val_size / (val_size + test_size)
#
#     # Split temp patients into validation and testing
#     val_patients, test_patients = train_test_split(temp_patients, train_size=val_temp_size, random_state=42)
#
#     # Split the original data into training, validation, and testing sets based on patient IDs
#     train_data = data[data[patient_id_col].isin(train_patients)]
#     val_data = data[data[patient_id_col].isin(val_patients)]
#     test_data = data[data[patient_id_col].isin(test_patients)]
#
#     # Save to CSV files
#     if output_prefix:
#         train_data.to_csv(f'{output_prefix}_train.csv', index=False)
#         val_data.to_csv(f'{output_prefix}_val.csv', index=False)
#         test_data.to_csv(f'{output_prefix}_test.csv', index=False)
#     else:
#         train_data.to_csv('train.csv', index=False)
#         val_data.to_csv('val.csv', index=False)
#         # test_data.to_csv('test.csv', index=False)
#
#     return train_data, val_data, test_data
#
# # Example usage
#
# data = '500co_180fu.csv'
# train_data, val_data, test_data = split_patient_data(data, train_size=0.8, val_size=0.19, test_size=0.01, output_prefix='500cu_180fu_07172024')
import pandas as pd
from sklearn.model_selection import train_test_split


def split_patient_data(data, train_size, val_size, patient_id_col='patient_id', output_prefix=''):
    # Check that the sizes sum to 1.0
    assert train_size + val_size == 1.0, "Train and validation sizes must sum to 1.0"

    # Read CSV if data is a string (file path)
    if isinstance(data, str):
        data = pd.read_csv(data)

    # Extract unique patient IDs
    unique_patients = data[patient_id_col].unique()

    # Split unique patients into training and validation sets
    train_patients, val_patients = train_test_split(unique_patients, train_size=train_size, random_state=42)

    # Split the original data into training and validation sets based on patient IDs
    train_data = data[data[patient_id_col].isin(train_patients)]
    val_data = data[data[patient_id_col].isin(val_patients)]

    # Save to CSV files
    if output_prefix:
        train_data.to_csv(f'{output_prefix}_train.csv', index=False)
        val_data.to_csv(f'{output_prefix}_val.csv', index=False)
    else:
        train_data.to_csv('train.csv', index=False)
        val_data.to_csv('val.csv', index=False)

    return train_data, val_data

# Example usage
data = 'embed_120co_60fu.csv'
train_data, val_data = split_patient_data(data, train_size=0.8, val_size=0.2, output_prefix='embed_120co_60fu')
