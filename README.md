
# InsureIQ: Real-Time Medical-Charge Predictor

InsureIQ predicts insurance expenses using demographic and health-related attributes from a dataset containing 10000 records.

🎯 Problem Statement:
The medical insurance industry needs to accurately predict insurance costs based on customer attributes to:

* Set appropriate premium rates
* Identify key risk factors
* Optimize business profitability
* Ensure fair pricing policies

This project analyzes and predicts insurance expenses using demographic and health-related attributes from a dataset containing 10000 records.

📊 Dataset Overview
The dataset contains the following features:

* age: Age of the policyholder
* gender: Gender of the policyholder (male/female)
* bmi: Body Mass Index (BMI) of the policyholder
* children: Number of children/dependents
* discount_eligibility: Whether the policyholder is eligible for discounts (yes/no)
* region: Geographic region (northeast, northwest, southeast, southwest)
* expenses: Actual cost incurred by policyholders (Target Variable)


## Quick Start

 1. Clone Repository

```bash
  git clone https://github.com/younus8imran/InsureIQ.git
  cd InsureIQ
```
2. Setup Environment

```bash
  # Create virtual environment
  python -m venv venv

  # acivate the environment
  source venv/bin/activate

  # install all dependencies
  pip install -r requirements.txt
```
3.Train models & log with MLflow

```bash
  python model/train.py

```
4.Run the project locally using docker 
```bash
# Build and start MLflow and API services
docker-compose up --build

```
Access the fast api docs here ```http://127.0.0.1:8000/docs```
and the UI can be accessed here : ```http://127.0.0.1:8000/```

The best model is chosen automatically using GridSearchCV, with lowest rmse value.

All runs are tracked inside ```artifacts/mlruns ```; 
Open MLflow UI with 
```bash
 mlflow ui --backend-store-uri file:./artifacts/mlruns
 ```