# generate_60k.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
import os

# 1. Load original data
ORIGINAL_PATH = "data/insurance.csv"
df_orig = pd.read_csv(ORIGINAL_PATH)
TARGET = "charges"

# 2. Fit a small regression model to capture relationships
y_orig = df_orig[TARGET]
X_orig = df_orig.drop(columns=[TARGET])

# Encode categoricals the same way the training pipeline will do
num_cols = X_orig.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X_orig.select_dtypes(include=["object"]).columns.tolist()

enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_enc = pd.DataFrame(
    enc.fit_transform(X_orig[cat_cols]),
    columns=enc.get_feature_names_out(cat_cols),
)
X_full = pd.concat([X_orig[num_cols], X_enc], axis=1)

reg = LinearRegression()
reg.fit(X_full, y_orig)

# 3. Sampling hyper-parameters
N_TARGET = 60_000
N_ORIG = len(df_orig)
np.random.seed(42)

# 4. Generate numeric features by resampling + small Gaussian noise
def jitter(series, noise_scale=0.02):
    return series + np.random.normal(0, noise_scale * series.std(), size=len(series))

numeric_samples = {}
for col in num_cols:
    base = np.random.choice(df_orig[col], size=N_TARGET)
    numeric_samples[col] = jitter(base)

# 5. Generate categorical features by resampling
categorical_samples = {}
for col in cat_cols:
    categorical_samples[col] = np.random.choice(df_orig[col], size=N_TARGET)

# 6. Build new dataframe
df_new = pd.DataFrame({**numeric_samples, **categorical_samples})

# 7. Encode new categorical features
X_new_enc = pd.DataFrame(
    enc.transform(df_new[cat_cols]),
    columns=enc.get_feature_names_out(cat_cols),
)
X_new_full = pd.concat([df_new[num_cols], X_new_enc], axis=1)

# 8. Generate target using the fitted model + Gaussian noise
y_new = reg.predict(X_new_full) + np.random.normal(
    0, scale=y_orig.std() * 0.08, size=N_TARGET
)
y_new = np.abs(y_new)  # charges can't be negative

df_new[TARGET] = y_new

# 9. Re-order columns to match original
df_new = df_new[list(X_orig.columns) + [TARGET]]

# 10. Save
OUT_FILE = "data/insurance_60k.csv"
df_new.to_csv(OUT_FILE, index=False)
print(f"Created {OUT_FILE} with {len(df_new):,} rows.")