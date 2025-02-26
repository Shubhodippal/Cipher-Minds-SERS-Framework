import pandas as pd
from tqdm import tqdm
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from .feature_extraction import get_features
from .model import model_architecture

def process_audio_dataset(input_csv, output_csv, train_size=0.80, random_state=0, batch_size=64, epochs=100, output_shape=7, optimizer='RMSprop'):
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
    print("Data pre-processing phase starting")

    X = feature_set.iloc[: ,:-1].values
    Y = feature_set['Emotion_Label'].values

    encoder = OneHotEncoder()
    Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()

    joblib.dump(encoder, "onehot_encoder.pkl")

    # splitting data
    x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=random_state, train_size=train_size, shuffle=True)
    #x_train.shape, y_train.shape, x_test.shape, y_test.shape

    # scaling our data with sklearn's Standard scaler
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    #x_train.shape, y_train.shape, x_test.shape, y_test.shape


    joblib.dump(scaler, "scaler.pkl")

    # making our data compatible to model.
    x_train = np.expand_dims(x_train, axis=2)
    x_test = np.expand_dims(x_test, axis=2)
    #x_train.shape, y_train.shape, x_test.shape, y_test.shape

    model_architecture(x_train, y_train, x_test, y_test, batch_size, epochs, output_shape, optimizer)
    

