import librosa
import numpy as np
from .import augmentation as ag

def extract_features(data, sample_rate):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally

    #Chroma_cqt
    chroma_cqt = np.mean(librosa.feature.chroma_cqt(y=data, sr=sample_rate))
    result = np.hstack((result, chroma_cqt)) # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally
    
    return result

def get_features(path):
    data, sample_rate = librosa.load(path)
    
    # without augmentation
    res1 = extract_features(data=data, sample_rate=sample_rate)
    result = np.array(res1)
    
    # data with noise
    noise_data = ag.noise(data)
    res2 = extract_features(data=noise_data, sample_rate=sample_rate)
    result = np.vstack((result, res2)) # stacking vertically
    
    # data with stretching 
    stretch_data = ag.stretch(data)
    res3 = extract_features(data=stretch_data, sample_rate=sample_rate)
    result = np.vstack((result, res3)) # stacking vertically

    #data with shift
    shift_data =  ag.shift(data)
    res4 = extract_features(data=shift_data, sample_rate=sample_rate)
    result = np.vstack((result, res4)) # stacking vertically

    #data with pitch
    pitch_data = ag.pitch(data, sampling_rate=sample_rate)
    res5 = extract_features(data=pitch_data, sample_rate=sample_rate)
    np.vstack((result, res5)) # stacking vertically
    
    return result