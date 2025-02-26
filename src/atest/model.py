import keras
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
import joblib

from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Dropout, GlobalAveragePooling1D, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2


def model_architecture(x_train, y_train, x_test, y_test,batch_size=64, epochs=100, output_shape=7, optimizer='RMSprop'):

    
    input_shape = (x_train.shape[1], 1)

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

    # Compile the model
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Callbacks
    checkpointer = ModelCheckpoint(filepath='saved_models/audio_classification.hdf5', verbose=1, save_best_only=True)
    rlrp = ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1, patience=5, min_lr=1e-7)

    print(model.summary())

    # Train the model
    history=model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), callbacks=[rlrp, checkpointer])

    print("Accuracy of our model on test data : " , model.evaluate(x_test,y_test)[1]*100 , "%")

    epochs = [i for i in range(100)]
    fig , ax = plt.subplots(1,2)
    train_acc = history.history['accuracy']
    train_loss = history.history['loss']
    test_acc = history.history['val_accuracy']
    test_loss = history.history['val_loss']

    fig.set_size_inches(20,6)
    ax[0].plot(epochs , train_loss , label = 'Training Loss')
    ax[0].plot(epochs , test_loss , label = 'Testing Loss')
    ax[0].set_title('Training & Testing Loss')
    ax[0].legend()
    ax[0].set_xlabel("Epochs")

    ax[1].plot(epochs , train_acc , label = 'Training Accuracy')
    ax[1].plot(epochs , test_acc , label = 'Testing Accuracy')
    ax[1].set_title('Training & Testing Accuracy')
    ax[1].legend()
    ax[1].set_xlabel("Epochs")
    plt.show()

    # predicting on test data.
    encoder = joblib.load("onehot_encoder.pkl")
    pred_test = model.predict(x_test)
    y_pred = encoder.inverse_transform(pred_test)

    y_test = encoder.inverse_transform(y_test)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize = (12, 10))
    cm = pd.DataFrame(cm , index = [i for i in encoder.categories_] , columns = [i for i in encoder.categories_])
    sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
    plt.title('Confusion Matrix', size=20)
    plt.xlabel('Predicted Labels', size=14)
    plt.ylabel('Actual Labels', size=14)
    plt.show()

    cm.to_csv('Confusion_matrix.csv')

    print(classification_report(y_test, y_pred))
    report = classification_report(y_test, y_pred)

    with open('report.csv', 'w') as file:
        file.write(report)
