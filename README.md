
# InsureIQ: Real-Time Medical-Charge Predictor

InsureIQ predicts an individual's annual Medical Charge based on certain parameters (age, bmi, smoker, number of children and region).
Dataset source: from kaggle




## Run Locally

Clone the project

```bash
  git clone https://github.com/younus8imran/InsureIQ.git
```

Go to the project directory

```bash
  cd InsureIQ
```

Install dependencies

```bash
  pip install -r requirement.txt
```

Train models & log with MLflow

```bash
  python model/train.py

```
Run the project locally using docker 
Build the image
```bash
#run once or whenever the code/Dockerfile changes
docker build -t insureiq .
# run the container
docker run -p 8000:8000 medical-cost-forecaster

```
Access the fast api docs here ```http://127.0.0.1:8000/docs```
and the UI can be accessed here : ```http://127.0.0.1:8000/```

The best model is chosen automatically using GridSearchCV, with lowest rmse value.

All runs are tracked inside ```artifacts/mlruns ```; 
Open MLflow UI with 
```bash
 mlflow ui --backend-store-uri file:./artifacts/mlruns
 ```


