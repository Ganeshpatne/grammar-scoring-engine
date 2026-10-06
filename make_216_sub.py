import os
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

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

model_lgb = LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbose=-1, n_jobs=-1)
model_lgb.fit(X, y)
preds_lgb = model_lgb.predict(X_test)

model_ridge = Ridge(alpha=10.0)
model_ridge.fit(X, y)
preds_ridge = model_ridge.predict(X_test)

preds = np.clip(0.7 * preds_lgb + 0.3 * preds_ridge, 0.0, 5.0)

df_test_ac["pred_label"] = preds
sub_map = dict(zip(df_test_ac["filename"], df_test_ac["pred_label"]))

sub_df = df_test_raw[["filename", "label"]].copy()
sub_df["label"] = sub_df["filename"].map(sub_map).fillna(df_train_raw["label"].mean())
sub_df["label"] = np.clip(sub_df["label"], 0.0, 5.0)

out_path = os.path.join(SUBMISSION_DIR, "submission.csv")
sub_df.to_csv(out_path, index=False)
print("SUCCESS! File saved to:", out_path)
print("Row count:", len(sub_df))
print(sub_df.head(3))
print(sub_df.tail(3))
