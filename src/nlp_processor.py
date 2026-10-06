"""
NLP & ASR Feature Processing Module for Grammar Scoring Engine
Transcribes audio files using OpenAI Whisper and extracts grammatical/linguistic features.
Uses thread-safe locking for PyTorch Whisper inference.
"""

import os
import re
import threading
import pandas as pd
import numpy as np
import librosa
import whisper

_WHISPER_MODEL = None
_WHISPER_LOCK = threading.Lock()

def get_whisper_model(model_name="tiny"):
    """Singleton getter for Whisper model."""
    global _WHISPER_MODEL
    with _WHISPER_LOCK:
        if _WHISPER_MODEL is None:
            print(f"Loading Whisper ASR model '{model_name}'...", flush=True)
            _WHISPER_MODEL = whisper.load_model(model_name)
    return _WHISPER_MODEL

def transcribe_audio_file(file_path, model_name="tiny", y_audio=None):
    """
    Thread-safe ASR transcription using OpenAI Whisper.
    """
    try:
        if y_audio is None:
            y_audio, _ = librosa.load(file_path, sr=16000, mono=True)
            
        # Ensure audio array has at least 1 second length (16000 samples)
        if len(y_audio) < 16000:
            y_audio = np.pad(y_audio, (0, 16000 - len(y_audio)))
            
        y_float = y_audio.astype(np.float32)
        
        # Check for invalid NaN/Inf values
        if not np.isfinite(y_float).all():
            y_float = np.nan_to_num(y_float)
            
        model = get_whisper_model(model_name)
        
        with _WHISPER_LOCK:
            result = model.transcribe(y_float, language="en", fp16=False)
            
        return result.get("text", "").strip()
    except Exception as e:
        return ""

def extract_text_grammar_features(text, audio_duration=1.0):
    """
    Extracts NLP, linguistic, and syntactic grammar metrics from transcribed text.
    """
    features = {}
    
    clean_text = text.strip() if text else ""
    words = re.findall(r"\b\w+\b", clean_text.lower())
    n_words = len(words)
    n_chars = len(clean_text)
    
    features["text_length"] = int(n_chars)
    features["word_count"] = int(n_words)
    features["avg_word_length"] = float(n_chars / (n_words + 1e-5))
    features["speech_rate_wpm"] = float((n_words / (audio_duration + 1e-5)) * 60.0)
    
    # Lexical Diversity
    unique_words = set(words)
    features["unique_word_count"] = int(len(unique_words))
    features["type_token_ratio"] = float(len(unique_words) / (n_words + 1e-5))
    
    # Disfluencies & Filler Words
    filler_words = {"um", "uh", "er", "ah", "like", "you know", "mean", "so", "actually"}
    filler_count = sum(1 for w in words if w in filler_words)
    features["filler_word_count"] = int(filler_count)
    features["filler_word_ratio"] = float(filler_count / (n_words + 1e-5))
    
    # Sentence & Punctuation Structure
    sentences = [s.strip() for s in re.split(r"[.!?]+", clean_text) if s.strip()]
    features["sentence_count"] = int(len(sentences))
    features["avg_sentence_length"] = float(n_words / (len(sentences) + 1e-5))
    
    periods = clean_text.count(".")
    commas = clean_text.count(",")
    questions = clean_text.count("?")
    exclamations = clean_text.count("!")
    
    features["comma_count"] = int(commas)
    features["period_count"] = int(periods)
    features["punctuation_density"] = float((commas + periods + questions + exclamations) / (n_words + 1e-5))
    
    # Repetition
    repetitive_words = sum(1 for i in range(len(words)-1) if words[i] == words[i+1])
    features["repeated_word_count"] = int(repetitive_words)
    
    return features
