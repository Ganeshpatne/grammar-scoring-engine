"""
End-to-End Pipeline for Grammar Scoring Engine
Process-isolated parallel feature extraction, 5-Fold Stratified CV,
Training RMSE computation, and competition prediction generation.
"""

import os
import sys
import time
import librosa
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from scipy.stats import pearsonr
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Import local processor modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from audio_processor import extract_acoustic_features
from nlp_processor import transcribe_audio_file, extract_text_grammar_features

DATASET_DIR = "c:/Users/ganes/OneDrive/Desktop/kaggle/Dataset_Final"
CACHE_DIR = "c:/Users/ganes/OneDrive/Desktop/kaggle/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def process_single_file(args):
    """
    Top-level helper function for process-isolated execution in multiprocessing.Pool.
    """
    fname, audio_dir, label_val = args
    audio_path = os.path.join(audio_dir, fname)
    
    try:
        # Load audio once
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # 1. Acoustic Features
        ac_feats = extract_acoustic_features(audio_path)
        
        # 2. ASR Transcript & Text Grammar Features
        transcript = transcribe_audio_file(audio_path, model_name="tiny", y_audio=y)
        txt_feats = extract_text_grammar_features(transcript, audio_duration=ac_feats["duration"])
        
        rec = {"filename": fname}
        if label_val is not None and pd.notna(label_val) and label_val >= 0:
            rec["label"] = float(label_val)
        rec["transcript"] = transcript
        rec.update(ac_feats)
        rec.update(txt_feats)
        return rec
    except Exception as e:
        print(f"Error processing {fname}: {e}", flush=True)
        return None

def build_feature_dataset(df, audio_dir, split_name="train", n_processes=4):
    """
    Builds feature dataset with process isolation and incremental caching.
    """
    cache_path = os.path.join(CACHE_DIR, f"features_{split_name}.csv")
    
    existing_df = None
    processed_files = set()
    records = []
    
    if os.path.exists(cache_path):
        try:
            existing_df = pd.read_csv(cache_path)
            processed_files = set(existing_df["filename"])
            records = existing_df.to_dict("records")
            print(f"Found existing cache with {len(records)} samples for {split_name}.", flush=True)
        except Exception:
            pass
            
    remaining_rows = [row for _, row in df.iterrows() if row["filename"] not in processed_files]
    
    if len(remaining_rows) == 0:
        print(f"All {len(df)} samples for {split_name} already extracted and cached!", flush=True)
        return pd.DataFrame(records)
        
    print(f"Processing remaining {len(remaining_rows)}/{len(df)} samples for {split_name} using {n_processes} worker processes...", flush=True)
    
    task_args = [
        (row["filename"], audio_dir, row.get("label", None))
        for row in remaining_rows
    ]
    
    t0 = time.time()
    completed = len(records)
    total_samples = len(df)
    
    with Pool(processes=n_processes) as pool:
        for rec in pool.imap_unordered(process_single_file, task_args, chunksize=1):
            if rec is not None:
                records.append(rec)
            completed += 1
            
            if completed % 20 == 0 or completed == total_samples:
                # Incremental Save to Cache CSV
                df_curr = pd.DataFrame(records)
                df_curr.to_csv(cache_path, index=False)
                
                elapsed = time.time() - t0
                speed = (completed - len(processed_files)) / (elapsed + 1e-5)
                remaining_sec = (total_samples - completed) / (speed + 1e-5)
                print(f"  [{split_name}] Completed {completed}/{total_samples} samples ({speed:.2f} samples/sec, ~{remaining_sec:.1f}s remaining)...", flush=True)
                
    res_df = pd.DataFrame(records)
    res_df.to_csv(cache_path, index=False)
    print(f"Saved {split_name} feature dataset with shape {res_df.shape} to {cache_path}", flush=True)
    return res_df

def compute_metrics(y_true, y_pred):
    """Computes RMSE and Pearson Correlation Coefficient."""
    rmse = float(root_mean_squared_error(y_true, y_pred))
    corr, _ = pearsonr(y_true, y_pred)
    return rmse, float(corr)

def train_and_evaluate_cv(df_train, df_test, feature_cols, n_splits=5):
    """
    Trains 5-Fold Stratified Cross-Validation Ensemble (LightGBM, XGBoost, CatBoost, Ridge).
    Calculates Training RMSE, Validation OOF RMSE, and Pearson Correlation.
    """
    X = df_train[feature_cols].copy()
    y = df_train["label"].values
    X_test = df_test[feature_cols].copy()
    
    # Fill any missing values
    X = X.fillna(X.median())
    X_test = X_test.fillna(X.median())
    
    # Stratified K-Fold setup based on binned continuous target
    y_binned = pd.qcut(y, q=10, labels=False, duplicates='drop')
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    oof_preds_lgb = np.zeros(len(df_train))
    oof_preds_xgb = np.zeros(len(df_train))
    oof_preds_cat = np.zeros(len(df_train))
    oof_preds_ridge = np.zeros(len(df_train))
    oof_preds_blend = np.zeros(len(df_train))
    
    test_preds_blend = np.zeros(len(df_test))
    
    train_rmse_list = []
    
    print(f"\n--- Running {n_splits}-Fold Stratified Cross Validation ---", flush=True)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_binned)):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        
        # Feature Scaler
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # 1. LightGBM
        model_lgb = LGBMRegressor(n_estimators=300, learning_rate=0.03, random_state=42, verbose=-1)
        model_lgb.fit(X_tr, y_tr)
        pred_val_lgb = model_lgb.predict(X_val)
        pred_tr_lgb = model_lgb.predict(X_tr)
        
        # 2. XGBoost
        model_xgb = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=4, random_state=42)
        model_xgb.fit(X_tr, y_tr)
        pred_val_xgb = model_xgb.predict(X_val)
        
        # 3. CatBoost
        model_cat = CatBoostRegressor(n_estimators=300, learning_rate=0.03, depth=4, random_state=42, verbose=0)
        model_cat.fit(X_tr, y_tr)
        pred_val_cat = model_cat.predict(X_val)
        
        # 4. Ridge
        model_ridge = Ridge(alpha=10.0)
        model_ridge.fit(X_tr_scaled, y_tr)
        pred_val_ridge = model_ridge.predict(X_val_scaled)
        
        # Fold Blend
        val_blend = 0.35 * pred_val_lgb + 0.30 * pred_val_xgb + 0.20 * pred_val_cat + 0.15 * pred_val_ridge
        val_blend = np.clip(val_blend, 0.0, 5.0)
        
        oof_preds_lgb[val_idx] = pred_val_lgb
        oof_preds_xgb[val_idx] = pred_val_xgb
        oof_preds_cat[val_idx] = pred_val_cat
        oof_preds_ridge[val_idx] = pred_val_ridge
        oof_preds_blend[val_idx] = val_blend
        
        # Calculate Training RMSE for Fold
        tr_blend = 0.35 * pred_tr_lgb + 0.30 * model_xgb.predict(X_tr) + 0.20 * model_cat.predict(X_tr) + 0.15 * model_ridge.predict(X_tr_scaled)
        fold_train_rmse = float(root_mean_squared_error(y_tr, np.clip(tr_blend, 0.0, 5.0)))
        train_rmse_list.append(fold_train_rmse)
        
        fold_val_rmse, fold_val_corr = compute_metrics(y_val, val_blend)
        print(f"Fold {fold+1}: Val RMSE = {fold_val_rmse:.4f}, Val Pearson r = {fold_val_corr:.4f}", flush=True)
        
        # Test Set Prediction Accumulation
        test_p_lgb = model_lgb.predict(X_test)
        test_p_xgb = model_xgb.predict(X_test)
        test_p_cat = model_cat.predict(X_test)
        test_p_ridge = model_ridge.predict(X_test_scaled)
        fold_test_pred = 0.35 * test_p_lgb + 0.30 * test_p_xgb + 0.20 * test_p_cat + 0.15 * test_p_ridge
        test_preds_blend += fold_test_pred / n_splits
        
    mean_train_rmse = float(np.mean(train_rmse_list))
    oof_rmse, oof_corr = compute_metrics(y, oof_preds_blend)
    
    print("\n==============================================", flush=True)
    print(f"MANDATORY TRAINING RMSE SCORE: {mean_train_rmse:.4f}", flush=True)
    print(f"VALIDATION OOF RMSE SCORE:     {oof_rmse:.4f}", flush=True)
    print(f"VALIDATION PEARSON CORR (r):   {oof_corr:.4f}", flush=True)
    print("==============================================", flush=True)
    
    return oof_preds_blend, np.clip(test_preds_blend, 0.0, 5.0), mean_train_rmse, oof_rmse, oof_corr
