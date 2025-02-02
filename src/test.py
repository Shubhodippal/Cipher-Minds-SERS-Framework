from atest import process_audio_dataset

# Paths
input_csv = "Banspemo.csv"
output_csv = "features.csv"

# Extract features and save to CSV
process_audio_dataset(input_csv, output_csv)
