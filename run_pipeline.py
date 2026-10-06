"""
Main Execution Script for Kaggle Grammar Scoring Engine
Runs end-to-end feature extraction, 5-Fold Stratified CV, training RMSE calculation, and submission generation.
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from pipeline import build_feature_dataset, train_and_evaluate_cv

DATASET_DIR = "c:/Users/ganes/OneDrive/Desktop/kaggle/Dataset_Final"
SUBMISSION_DIR = "c:/Users/ganes/OneDrive/Desktop/kaggle/submissions"
os.makedirs(SUBMISSION_DIR, exist_ok=True)

def main():
    print("=== Kaggle Grammar Scoring Engine Pipeline ===")
    
    train_csv = os.path.join(DATASET_DIR, "train.csv")
    test_csv = os.path.join(DATASET_DIR, "test.csv")
    sub_csv = os.path.join(DATASET_DIR, "sample_submission.csv")
    
    df_train_raw = pd.read_csv(train_csv)
    df_test_raw = pd.read_csv(test_csv)
    df_sub_raw = pd.read_csv(sub_csv)
    
    train_audio_dir = os.path.join(DATASET_DIR, "train")
    test_audio_dir = os.path.join(DATASET_DIR, "test")
    
    # Extract / Load Features
    df_train_feat = build_feature_dataset(df_train_raw, train_audio_dir, split_name="train")
    df_test_feat = build_feature_dataset(df_test_raw, test_audio_dir, split_name="test")
    
    # Identify feature columns
    ignore_cols = ["filename", "label", "transcript"]
    feature_cols = [c for c in df_train_feat.columns if c not in ignore_cols and pd.api.types.is_numeric_dtype(df_train_feat[c])]
    
    print(f"\nExtracted {len(feature_cols)} numeric features per sample:")
    print("Features:", feature_cols[:10], "... and more.")
    
    # Train 5-Fold Stratified CV Ensemble
    oof_preds, test_preds, mean_train_rmse, oof_rmse, oof_corr = train_and_evaluate_cv(
        df_train_feat, df_test_feat, feature_cols, n_splits=5
    )
    
    # Save predictions to test dataframe
    df_test_feat["pred_label"] = test_preds
    
    # Map predictions to test.csv format (216 test rows)
    sub_map = dict(zip(df_test_feat["filename"], df_test_feat["pred_label"]))
    
    sub_df = df_test_raw[["filename", "label"]].copy()
    sub_df["label"] = sub_df["filename"].map(sub_map)
    
    # Fallback for any unmapped test files
    global_mean = df_train_raw["label"].mean()
    sub_df["label"] = sub_df["label"].fillna(global_mean)
    sub_df["label"] = np.clip(sub_df["label"], 0.0, 5.0)
    
    out_sub_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    sub_df.to_csv(out_sub_path, index=False)
    print(f"\nSaved final competition submission to {out_sub_path} (shape: {sub_df.shape})")
    print("Submission sample head:")
    print(sub_df.head(5))

if __name__ == "__main__":
    main()
