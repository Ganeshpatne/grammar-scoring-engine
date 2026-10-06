"""
Audio Feature Extraction Module for Grammar Scoring Engine
Computes acoustic, spectral, prosodic, and fluency metrics from WAV audio files.
"""

import os
import wave
import numpy as np
import scipy.io.wavfile as wav
import librosa

def extract_acoustic_features(file_path):
    """
    Extracts acoustic & prosodic features from a single WAV audio file.
    
    Parameters:
        file_path (str): Path to .wav audio file.
        
    Returns:
        dict: Extracted numeric feature dictionary.
    """
    features = {}
    
    # 1. Basic Audio Info via wave
    with wave.open(file_path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        duration = n_frames / float(framerate)
        
    features['duration'] = float(duration)
    features['channels'] = int(n_channels)
    features['sample_rate'] = int(framerate)
    
    # 2. Load audio signal with librosa (resampled to 16kHz)
    y, sr = librosa.load(file_path, sr=16000, mono=True)
    
    # 3. Energy & Fluency / Silence Features
    rms = librosa.feature.rms(y=y)[0]
    features['rms_mean'] = float(np.mean(rms))
    features['rms_std'] = float(np.std(rms))
    features['rms_max'] = float(np.max(rms))
    features['rms_min'] = float(np.min(rms))
    
    # Non-silent speech frame detection (Energy threshold)
    energy_threshold = 0.02 * np.max(rms)
    silent_frames = rms < energy_threshold
    features['silence_ratio'] = float(np.mean(silent_frames))
    features['active_speech_ratio'] = float(1.0 - features['silence_ratio'])
    features['speaking_duration'] = float(duration * features['active_speech_ratio'])
    
    # Pause counting (> 300ms contiguous silence)
    frame_length_sec = 512 / sr
    min_pause_frames = int(0.3 / frame_length_sec)
    
    pauses = 0
    current_pause_len = 0
    for is_silent in silent_frames:
        if is_silent:
            current_pause_len += 1
        else:
            if current_pause_len >= min_pause_frames:
                pauses += 1
            current_pause_len = 0
    if current_pause_len >= min_pause_frames:
        pauses += 1
        
    features['pause_count'] = int(pauses)
    features['pauses_per_min'] = float((pauses / (duration + 1e-5)) * 60.0)
    
    # 4. Spectral Features
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features['spectral_centroid_mean'] = float(np.mean(spec_cent))
    features['spectral_centroid_std'] = float(np.std(spec_cent))
    
    spec_flat = librosa.feature.spectral_flatness(y=y)[0]
    features['spectral_flatness_mean'] = float(np.mean(spec_flat))
    features['spectral_flatness_std'] = float(np.std(spec_flat))
    
    spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features['spectral_rolloff_mean'] = float(np.mean(spec_roll))
    features['spectral_rolloff_std'] = float(np.std(spec_roll))
    
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    features['zcr_mean'] = float(np.mean(zcr))
    features['zcr_std'] = float(np.std(zcr))
    
    # 5. MFCC Features (20 Coefficients + Delta)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_delta = librosa.feature.delta(mfcc)
    
    for i in range(20):
        features[f'mfcc_mean_{i}'] = float(np.mean(mfcc[i]))
        features[f'mfcc_std_{i}'] = float(np.std(mfcc[i]))
        features[f'mfcc_delta_mean_{i}'] = float(np.mean(mfcc_delta[i]))
        features[f'mfcc_delta_std_{i}'] = float(np.std(mfcc_delta[i]))
        
    return features
