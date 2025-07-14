import joblib
import os
import glob

BEST_MODEL_DIR = 'artifacts/best_model'

joblib_files = glob.glob(os.path.join(BEST_MODEL_DIR, "*.joblib"))

print(joblib_files[0])    