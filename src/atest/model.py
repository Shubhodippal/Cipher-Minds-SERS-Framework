import pandas as pd
import numpy as np
import os
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Dropout, GlobalAveragePooling1D, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2


def build_model(input_shape, output_shape):
    model = Sequential([
        # First Conv1D Layer
        Conv1D(256, kernel_size=9, strides=1, padding='same', activation='relu', kernel_regularizer=l2(0.0001), input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=3, strides=2, padding='same'),
        Dropout(0.2),

        # Second Conv1D Layer
        Conv1D(512, kernel_size=7, strides=1, padding='same', activation='relu', kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        MaxPooling1D(pool_size=3, strides=2, padding='same'),
        Dropout(0.25),

        # Third Conv1D Layer
        Conv1D(1024, kernel_size=5, strides=1, padding='same', activation='relu', kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        MaxPooling1D(pool_size=3, strides=2, padding='same'),
        Dropout(0.3),

        # Fourth Conv1D Layer
        Conv1D(2048, kernel_size=3, strides=1, padding='same', activation='relu', kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        MaxPooling1D(pool_size=3, strides=2, padding='same'),
        Dropout(0.35),

        # Global Average Pooling
        GlobalAveragePooling1D(),

        # Fully connected layers
        Dense(2048, activation='relu', kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Dropout(0.4),

        Dense(1024, activation='relu', kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Dropout(0.4),

        # Output layer
        Dense(output_shape, activation='softmax')
    ])
    return model


def compile_model(model, optimizer):
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(model, x_train, y_train, x_test, y_test, batch_size, epochs, folder_name):
    checkpointer = ModelCheckpoint(filepath=os.path.join(folder_name, 'audio_classification.hdf5'), verbose=1, save_best_only=True)
    rlrp = ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1, patience=5, min_lr=1e-7)

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), callbacks=[rlrp, checkpointer])
    return history


def evaluate_model(model, x_test, y_test):
    accuracy = model.evaluate(x_test, y_test)[1] * 100
    print("Accuracy of our model on test data: ", accuracy, "%")
    return accuracy


def plot_metrics(history, epochs, folder_name):
    epochs_range = range(epochs)
    fig, ax = plt.subplots(1, 2)
    train_acc = history.history['accuracy']
    train_loss = history.history['loss']
    test_acc = history.history['val_accuracy']
    test_loss = history.history['val_loss']

    fig.set_size_inches(20, 6)
    ax[0].plot(epochs_range, train_loss, label='Training Loss')
    ax[0].plot(epochs_range, test_loss, label='Testing Loss')
    ax[0].set_title('Training & Testing Loss')
    ax[0].legend()
    ax[0].set_xlabel("Epochs")

    ax[1].plot(epochs_range, train_acc, label='Training Accuracy')
    ax[1].plot(epochs_range, test_acc, label='Testing Accuracy')
    ax[1].set_title('Training & Testing Accuracy')
    ax[1].legend()
    ax[1].set_xlabel("Epochs")
    plt.savefig(os.path.join(folder_name, 'training_testing_metrics.png'))


def save_confusion_matrix_and_report(model, x_test, y_test, folder_name):
    encoder = joblib.load(os.path.join(folder_name, "onehot_encoder.pkl"))
    pred_test = model.predict(x_test)
    y_pred = encoder.inverse_transform(pred_test)
    y_test = encoder.inverse_transform(y_test)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    cm = pd.DataFrame(cm, index=[i for i in encoder.categories_], columns=[i for i in encoder.categories_])
    sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
    plt.title('Confusion Matrix', size=20)
    plt.xlabel('Predicted Labels', size=14)
    plt.ylabel('Actual Labels', size=14)
    plt.savefig(os.path.join(folder_name, 'confusion_matrix.png'))

    cm.to_csv(os.path.join(folder_name, 'Confusion_matrix.csv'))

    report = classification_report(y_test, y_pred)
    print(report)
    with open(os.path.join(folder_name, 'report.csv'), 'w') as file:
        file.write(report)


def model_architecture(x_train, y_train, x_test, y_test, batch_size=64, epochs=100, output_shape=7, optimizer='RMSprop', folder_name="data"):
    input_shape = (x_train.shape[1], 1)
    model = build_model(input_shape, output_shape)
    model = compile_model(model, optimizer)
    print(model.summary())

    history = train_model(model, x_train, y_train, x_test, y_test, batch_size, epochs, folder_name)
    evaluate_model(model, x_test, y_test)
    plot_metrics(history, epochs, folder_name)
    save_confusion_matrix_and_report(model, x_test, y_test, folder_name)