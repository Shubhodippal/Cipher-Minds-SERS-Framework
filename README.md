# Audio Emotion Recognition

## Overview
This project is an Audio Emotion Recognition system that processes audio files, extracts features, and trains a deep learning model to classify emotions from speech signals. The system supports data augmentation, feature extraction, model training, and evaluation.

## Features
- **Feature Extraction**: Uses librosa to extract various audio features like Zero Crossing Rate, Chroma Features, MFCC, RMS, and Mel Spectrogram.
- **Data Augmentation**: Adds noise, time stretch, pitch shift, and time shift to improve model robustness.
- **Preprocessing**: Standardizes extracted features and applies One-Hot Encoding to emotion labels.
- **Model Architecture**: A Convolutional Neural Network (CNN) for sequence modeling.
- **Evaluation**: Provides accuracy, confusion matrix, and classification report.

## File Structure
```
Project/
├── [setup.py]
├── README.md
├── LICENSE
├── [requirements.txt]
└── src/
    ├── atest/
    │   ├── __init__.py
    │   ├── data_processing.py
    │   ├── feature_extraction.py
    │   ├── model.py
    └── [test.py]
```

## Requirements
Install the required dependencies using:
```bash
pip install -r requirements.txt
```

## How to Run
1. **Prepare Data**: Provide a CSV file (`banspemo_df.csv`) with columns:
   - `File Path`: Path to the audio file.
   - `Emotion`: Corresponding emotion label.

2. **Execute the Pipeline**:
```bash
python test.py
```
This will:
- Extract features
- Apply augmentation
- Preprocess data
- Train and evaluate the model

3. **Outputs**:
- `features.csv`: Extracted features
- `audio_classification.hdf5`: Trained model
- `Confusion_matrix.csv`: Classification results
- `training_testing_metrics.png`: Performance graphs

## Model Details
- **Conv1D Layers**: 4 layers with batch normalization, dropout, and pooling
- **Dense Layers**: 2 fully connected layers with ReLU activation
- **Optimizer**: RMSprop
- **Loss Function**: Categorical Cross-Entropy

## Acknowledgments
- Uses `librosa` for audio processing
- Built with TensorFlow/Keras
- Data augmentation inspired by various speech processing techniques

## Future Improvements
- Fine-tuning hyperparameters
- Expanding dataset for better generalization
- Implementing real-time emotion detection

