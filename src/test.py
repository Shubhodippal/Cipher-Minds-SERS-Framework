from atest import process_audio_dataset

# Paths
input_csv = "banspemo_df.csv"
output_csv = "features.csv"
folder_name = "data"

process_audio_dataset(input_csv, output_csv,output_shape=6,folder_name=folder_name)
