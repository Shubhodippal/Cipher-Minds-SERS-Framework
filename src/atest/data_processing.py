import pandas as pd
from tqdm import tqdm
from .feature_extraction import get_features

def process_audio_dataset(input_csv, output_csv):
    """
    Process a dataset of audio files to extract features and save them to a CSV file.

    Args:
        input_csv (str): Path to the input CSV file. Must contain 'File Path' and 'Emotion' columns.
        output_csv (str): Path to save the output CSV file.

    Returns:
        None
    """
    # Load the dataset
    df = pd.read_csv(input_csv)
    if 'File Path' not in df.columns or 'Emotion' not in df.columns:
        raise ValueError("Input CSV must contain 'File Path' and 'Emotion' columns.")
    
    # Initialize lists for storing features and labels
    extracted_features, emotions = [], []

    # Process each file
    print("Extracting features from audio files...")
    for file_path, emotion_label in tqdm(zip(df['File Path'], df['Emotion']), total=len(df)):
        try:
            features = get_features(file_path)
            for feature_vector in features:
                extracted_features.append(feature_vector)
                emotions.append(emotion_label)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    # Create a DataFrame and save it as CSV
    feature_set = pd.DataFrame(extracted_features)
    feature_set['Emotion_Label'] = emotions
    feature_set.to_csv(output_csv, index=False)
    print(f"Feature extraction complete. File saved as {output_csv}.")

    