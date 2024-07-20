import pandas as pd
pd.options.mode.chained_assignment = None

def get_visit_counts(df):
	#get patient number, date of study, type (diagnostic or screening), and score
	working_df = df
	#convert 'asses' score column to 0-6 scale
	working_df.loc[working_df['asses']=='A', 'birads_fix'] = 0
	working_df.loc[working_df['asses']=='N', 'birads_fix'] = 1
	working_df.loc[working_df['asses']=='B', 'birads_fix'] = 2
	working_df.loc[working_df['asses']=='P', 'birads_fix'] = 3
	working_df.loc[working_df['asses']=='S', 'birads_fix'] = 4
	working_df.loc[working_df['asses']=='M', 'birads_fix'] = 5
	working_df.loc[working_df['asses']=='K', 'birads_fix'] = 6
	working_df = df[['empi_anon','study_date_anon','desc','birads_fix']]

	#get series of value counts for unique patient IDs (each row per patient corresponds to a different visit)
	working_df_valcounts = working_df['empi_anon'].value_counts()    
	val_counts = pd.DataFrame({'empi_anon':working_df_valcounts.index,'visit_counts': working_df_valcounts.values})   

	#merge visit_counts column with original dataframe
	working_df_withcounts = pd.merge(working_df, val_counts, on = 'empi_anon') 

	working_df_withcounts.to_csv('/projects01/didsr-aiml/michelle.mastrianni/working_df_withcounts1.csv')

if __name__ == '__main__':
	df = pd.read_csv('/projects01/didsr-aiml/common_data/EMBED/tables/EMBED_OpenData_clinical_reduced.csv')
	get_visit_counts(df)
	print("Done\n")

