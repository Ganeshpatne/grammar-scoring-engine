"""
Fast Acoustic & Prosody Baseline Pipeline
Extracts 100% audio features (MFCCs, spectral centroid, zero crossing rate, energy, pause metrics)
in ~10 seconds, runs 5-Fold Stratified CV, computes Training RMSE, and generates baseline submission.
"""

import os
import sys

# Suppress Loky physical core subprocess warnings on Windows
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

import time
import pandas as pd
import numpy as np
from multiprocessing import Pool
from scipy.stats import pearsonr
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from audio_processor import extract_acoustic_features

DATASET_DIR = "c:/Users/ganes/OneDrive/Desktop/kaggle/Dataset_Final"
CACHE_DIR = "c:/Users/ganes/OneDrive/Desktop/kaggle/cache"
SUBMISSION_DIR = "c:/Users/ganes/OneDrive/Desktop/kaggle/submissions"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

def process_acoustic_file(args):
    """Processes acoustic features for a single file."""
    fname, audio_dir, label_val = args
    audio_path = os.path.join(audio_dir, fname)
    try:
        ac_feats = extract_acoustic_features(audio_path)
        rec = {"filename": fname}
        if label_val is not None and pd.notna(label_val) and label_val >= 0:
            rec["label"] = float(label_val)
        rec.update(ac_feats)
        return rec
    except Exception as e:
        print(f"Error acoustic extraction for {fname}: {e}", flush=True)
        return None

def build_acoustic_dataset(df, audio_dir, split_name="train", n_processes=4):
    """Extracts acoustic features for all samples in parallel."""
    cache_path = os.path.join(CACHE_DIR, f"acoustic_only_{split_name}.csv")
    if os.path.exists(cache_path):
        print(f"Loading cached acoustic features for {split_name}...", flush=True)
        return pd.read_csv(cache_path)
        
    print(f"Extracting fast acoustic features for {split_name} ({len(df)} samples)...", flush=True)
    task_args = [
        (row["filename"], audio_dir, row.get("label", None))
        for _, row in df.iterrows()
    ]
    
    records = []
    t0 = time.time()
    with Pool(processes=n_processes) as pool:
        results = pool.map(process_acoustic_file, task_args)
        records = [r for r in results if r is not None]
        
    print(f"Completed fast acoustic extraction for {split_name} in {time.time()-t0:.2f}s", flush=True)
    res_df = pd.DataFrame(records)
    res_df.to_csv(cache_path, index=False)
    return res_df

def compute_metrics(y_true, y_pred):
    """Computes RMSE and Pearson Correlation Coefficient."""
    rmse = float(root_mean_squared_error(y_true, y_pred))
    corr, _ = pearsonr(y_true, y_pred)
    return rmse, float(corr)

def main():
    print("=== Fast Acoustic & Prosody Baseline Pipeline ===", flush=True)
    train_csv = os.path.join(DATASET_DIR, "train.csv")
    test_csv = os.path.join(DATASET_DIR, "test.csv")
    sub_csv = os.path.join(DATASET_DIR, "sample_submission.csv")
    
    df_train_raw = pd.read_csv(train_csv)
    df_test_raw = pd.read_csv(test_csv)
    df_sub_raw = pd.read_csv(sub_csv)
    
    train_audio_dir = os.path.join(DATASET_DIR, "train")
    test_audio_dir = os.path.join(DATASET_DIR, "test")
    
    df_train_ac = build_acoustic_dataset(df_train_raw, train_audio_dir, split_name="train")
    df_test_ac = build_acoustic_dataset(df_test_raw, test_audio_dir, split_name="test")
    
    ignore_cols = ["filename", "label"]
    feature_cols = [c for c in df_train_ac.columns if c not in ignore_cols and pd.api.types.is_numeric_dtype(df_train_ac[c])]
    
    print(f"\nExtracted {len(feature_cols)} acoustic numeric features.", flush=True)
    
    X = df_train_ac[feature_cols].copy().fillna(df_train_ac[feature_cols].median())
    y = df_train_ac["label"].values
    X_test = df_test_ac[feature_cols].copy().fillna(df_train_ac[feature_cols].median())
    
    y_binned = pd.qcut(y, q=10, labels=False, duplicates='drop')
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(df_train_ac))
    test_preds = np.zeros(len(df_test_ac))
    train_rmse_list = []
    
    print("\n--- Running 5-Fold Stratified Cross Validation (Acoustic Baseline) ---", flush=True)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_binned)):
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]
        
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        model_lgb = LGBMRegressor(n_estimators=300, learning_rate=0.03, random_state=42, verbose=-1, n_jobs=1)
        model_lgb.fit(X_tr, y_tr)
        p_val_lgb = model_lgb.predict(X_val)
        p_tr_lgb = model_lgb.predict(X_tr)
        
        model_xgb = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=4, random_state=42, n_jobs=1)
        model_xgb.fit(X_tr, y_tr)
        p_val_xgb = model_xgb.predict(X_val)
        
        model_cat = CatBoostRegressor(n_estimators=300, learning_rate=0.03, depth=4, random_state=42, verbose=0, thread_count=1)
        model_cat.fit(X_tr, y_tr)
        p_val_cat = model_cat.predict(X_val)
        
        model_ridge = Ridge(alpha=10.0)
        model_ridge.fit(X_tr_scaled, y_tr)
        p_val_ridge = model_ridge.predict(X_val_scaled)
        
        val_blend = 0.35 * p_val_lgb + 0.30 * p_val_xgb + 0.20 * p_val_cat + 0.15 * p_val_ridge
        val_blend = np.clip(val_blend, 0.0, 5.0)
        oof_preds[val_idx] = val_blend
        
        tr_blend = 0.35 * p_tr_lgb + 0.30 * model_xgb.predict(X_tr) + 0.20 * model_cat.predict(X_tr) + 0.15 * model_ridge.predict(X_tr_scaled)
        fold_train_rmse = float(root_mean_squared_error(y_tr, np.clip(tr_blend, 0.0, 5.0)))
        train_rmse_list.append(fold_train_rmse)
        
        v_rmse, v_corr = compute_metrics(y_val, val_blend)
        print(f"Fold {fold+1}: Val RMSE = {v_rmse:.4f}, Val Pearson r = {v_corr:.4f}", flush=True)
        
        t_lgb = model_lgb.predict(X_test)
        t_xgb = model_xgb.predict(X_test)
        t_cat = model_cat.predict(X_test)
        t_ridge = model_ridge.predict(X_test_scaled)
        t_blend = 0.35 * t_lgb + 0.30 * t_xgb + 0.20 * t_cat + 0.15 * t_ridge
        test_preds += t_blend / 5.0
        
    mean_train_rmse = float(np.mean(train_rmse_list))
    oof_rmse, oof_corr = compute_metrics(y, oof_preds)
    
    print("\n==============================================", flush=True)
    print(f"MANDATORY TRAINING RMSE SCORE: {mean_train_rmse:.4f}", flush=True)
    print(f"VALIDATION OOF RMSE SCORE:     {oof_rmse:.4f}", flush=True)
    print(f"VALIDATION PEARSON CORR (r):   {oof_corr:.4f}", flush=True)
    print("==============================================", flush=True)
    
    # Save Baseline Submission
    df_test_ac["pred_label"] = np.clip(test_preds, 0.0, 5.0)
    sub_map = dict(zip(df_test_ac["filename"], df_test_ac["pred_label"]))
    
    sub_df = df_sub_raw.copy()
    sub_df["label"] = sub_df["filename"].map(sub_map)
    sub_df["label"] = sub_df["label"].fillna(df_train_raw["label"].mean())
    sub_df["label"] = np.clip(sub_df["label"], 0.0, 5.0)
    
    out_sub_path = os.path.join(SUBMISSION_DIR, "submission_baseline.csv")
    sub_df.to_csv(out_sub_path, index=False)
    print(f"Saved baseline submission to {out_sub_path} (shape: {sub_df.shape})", flush=True)

if __name__ == "__main__":
    main()
