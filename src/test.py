from atest import process_audio_dataset

# Paths
input_csv = "banspemo_df.csv"
output_csv = "features.csv"

# Extract features and save to CSV
process_audio_dataset(input_csv, output_csv)

