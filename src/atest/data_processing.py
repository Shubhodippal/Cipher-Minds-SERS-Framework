import pandas as pd
from tqdm import tqdm
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from .feature_extraction import get_features
from .model import model_architecture

def extract_features(input_csv):
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

    # Create a DataFrame
    feature_set = pd.DataFrame(extracted_features)
    feature_set['Emotion_Label'] = emotions
    return feature_set

def save_features(feature_set, output_csv, folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    feature_set.to_csv(os.path.join(folder_name, output_csv), index=False)
    print(f"Feature extraction complete. File saved as {output_csv}.")

def preprocess_data(feature_set, train_size, random_state, folder_name):
    X = feature_set.iloc[:, :-1].values
    Y = feature_set['Emotion_Label'].values

    encoder = OneHotEncoder()
    Y = encoder.fit_transform(np.array(Y).reshape(-1, 1)).toarray()
    joblib.dump(encoder, os.path.join(folder_name, "onehot_encoder.pkl"))

    # Splitting data
    x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=random_state, train_size=train_size, shuffle=True)

    # Scaling data
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    joblib.dump(scaler, os.path.join(folder_name, "scaler.pkl"))

    # Making data compatible with the model
    x_train = np.expand_dims(x_train, axis=2)
    x_test = np.expand_dims(x_test, axis=2)

    return x_train, x_test, y_train, y_test

def process_audio_dataset(input_csv, output_csv, train_size=0.80, random_state=0, batch_size=64, epochs=100, output_shape=7, optimizer='RMSprop', folder_name="data"):
    feature_set = extract_features(input_csv)
    save_features(feature_set, output_csv, folder_name)
    x_train, x_test, y_train, y_test = preprocess_data(feature_set, train_size, random_state, folder_name)
    print("Data pre-processing complete. Training model...")
    model_architecture(x_train, y_train, x_test, y_test, batch_size, epochs, output_shape, optimizer, folder_name)
    print("Model training complete.")