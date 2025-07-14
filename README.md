# Medical Charges MLOps Demo

1. Clone repo & place `insurance.csv` into `data/`
2. Install deps: `pip install -r requirements.txt`
3. Train models & log with MLflow:
   ```bash
   python model/train.py


---

## 4. Extra Notes

- The **best model** is chosen automatically by **lowest RMSE** on the hold-out set.  
- All runs are tracked inside `artifacts/mlruns`; open MLflow UI with  
  `mlflow ui --backend-store-uri file:./artifacts/mlruns`  
- To retrain simply run `python model/train.py` again; the FastAPI container will pick the new best model on restart.
