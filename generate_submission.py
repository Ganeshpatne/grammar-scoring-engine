import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import StratifiedKFold

DATASET_DIR = "Dataset_Final"
CACHE_DIR = "cache"
SUBMISSION_DIR = "submissions"
os.makedirs(SUBMISSION_DIR, exist_ok=True)

df_train_raw = pd.read_csv(os.path.join(DATASET_DIR, "train.csv"))
df_test_raw = pd.read_csv(os.path.join(DATASET_DIR, "test.csv"))

df_train_ac = pd.read_csv(os.path.join(CACHE_DIR, "acoustic_only_train.csv"))
df_test_ac = pd.read_csv(os.path.join(CACHE_DIR, "acoustic_only_test.csv"))

ignore_cols = ["filename", "label"]
feature_cols = [c for c in df_train_ac.columns if c not in ignore_cols and pd.api.types.is_numeric_dtype(df_train_ac[c])]

X = df_train_ac[feature_cols].copy().fillna(df_train_ac[feature_cols].median())
y = df_train_ac["label"].values

X_test = df_test_ac[feature_cols].copy().fillna(df_train_ac[feature_cols].median())

y_binned = pd.qcut(y, q=10, labels=False, duplicates='drop')
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

test_preds = np.zeros(len(df_test_ac))

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_binned)):
    X_tr, y_tr = X.iloc[train_idx], y[train_idx]
    
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_test_scaled = scaler.transform(X_test)
    
    model_lgb = LGBMRegressor(n_estimators=300, learning_rate=0.03, random_state=42, verbose=-1, n_jobs=1)
    model_lgb.fit(X_tr, y_tr)
    
    model_xgb = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=4, random_state=42, n_jobs=1)
    model_xgb.fit(X_tr, y_tr)
    
    model_cat = CatBoostRegressor(n_estimators=300, learning_rate=0.03, depth=4, random_state=42, verbose=0, thread_count=1)
    model_cat.fit(X_tr, y_tr)
    
    model_ridge = Ridge(alpha=10.0)
    model_ridge.fit(X_tr_scaled, y_tr)
    
    t_lgb = model_lgb.predict(X_test)
    t_xgb = model_xgb.predict(X_test)
    t_cat = model_cat.predict(X_test)
    t_ridge = model_ridge.predict(X_test_scaled)
    
    t_blend = 0.35 * t_lgb + 0.30 * t_xgb + 0.20 * t_cat + 0.15 * t_ridge
    test_preds += t_blend / 5.0

df_test_ac["pred_label"] = np.clip(test_preds, 0.0, 5.0)
sub_map = dict(zip(df_test_ac["filename"], df_test_ac["pred_label"]))

sub_df = df_test_raw[["filename", "label"]].copy()
sub_df["label"] = sub_df["filename"].map(sub_map).fillna(df_train_raw["label"].mean())
sub_df["label"] = np.clip(sub_df["label"], 0.0, 5.0)

out_sub_path = os.path.join(SUBMISSION_DIR, "submission.csv")
sub_df.to_csv(out_sub_path, index=False)
print(f"SUCCESS: Saved submission with {len(sub_df)} rows to {out_sub_path}")
print(sub_df.head(5))
print(sub_df.tail(5))
