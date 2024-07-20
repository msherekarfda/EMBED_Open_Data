import pandas as pd
import glob

# Find all .csv files that start with "embed"
csv_files = glob.glob('embed*.csv')

for file in csv_files:
    # Read the .csv file into a DataFrame
    df = pd.read_csv(file)

    # Drop the column named 'cancer_x' if it exists
    if 'cancer_x' in df.columns:
        df = df.drop(columns=['cancer_x'])

    # Rename the column 'cancer_y' to 'cancer' if it exists
    if 'cancer_y' in df.columns:
        df = df.rename(columns={'cancer_y': 'cancer'})

    # Save the DataFrame back to the same .csv file
    df.to_csv(file, index=False)
