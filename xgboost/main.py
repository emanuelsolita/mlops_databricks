# Databricks notebook source
# MAGIC %pip install xgboost

# COMMAND ----------
# Running this cell to restart the Python process and ensure that the newly installed xgboost package is available for import in subsequent cells.
dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import mlflow.xgboost
import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from datetime import datetime


# COMMAND ----------

def get_current_user():
    current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
    print(f"Current user: {current_user}")
    return current_user

# COMMAND ----------

mlflow.autolog()

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")  # Unity Catalog (recommended) - Default as of MLFlow 3.0
model_name = "emanuel_db.gold.xgboost_model"
timestamp = datetime.strftime(datetime.utcnow(), "%Y%m%d-%H%M%S")
run_name = f"run-churn-emhol-xgb-{timestamp}"
experiment_name = f"churn_for_mlops_emhol"

user = get_current_user()
experiment_name = f"/Users/{user}/{experiment_name}"
exp = mlflow.set_experiment(experiment_name)

print(f"Experiment name: {experiment_name}, experiment ID: {exp.experiment_id}, run name: {run_name}.")

# COMMAND ----------

data = spark.read.table("emanuel_db.gold.train_data_with_label")
data.limit(10).display()
data = data.toPandas()

# COMMAND ----------

# Clean up some metadata
data = data.iloc[:, :-2]

# COMMAND ----------

X = pd.get_dummies(
    data.drop(columns=["churn"]),
    drop_first=True
)
y = data["churn"]

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

# COMMAND ----------

with mlflow.start_run(
    experiment_id=exp.experiment_id, run_name=run_name, tags={"Environment" :"dev"}):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)

    # Log params + metrics
    mlflow.log_params(model.get_params())
    mlflow.log_metric("roc_auc", auc)

    # Log model
    model_info = mlflow.xgboost.log_model(
        model,
        name="model",
        input_example=X_train.iloc[:5],
        registered_model_name=model_name
    )

    print("ROC AUC:", auc)


# COMMAND ----------

# If model is not registered
#result = mlflow.register_model(
#    model_uri=model_info.model_uri,
#    name=model_name
#)

# COMMAND ----------

from mlflow.tracking import MlflowClient
client = MlflowClient()

client.search_model_versions(f"name='{model_name}'")

# COMMAND ----------

client.set_registered_model_alias(
    name=model_name,
    alias="champion",
    version=1
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Serve model
# MAGIC - Done via ClickOPS for this example
# MAGIC - Endpoint `https://adb-7405619060695703.3.azuredatabricks.net/serving-endpoints/emhol-xgboost/invocations`
# MAGIC
# MAGIC
# MAGIC So now, lets test it out. 
# MAGIC

# COMMAND ----------

import requests
import json

endpoint_url = 'https://adb-7405619060695703.3.azuredatabricks.net/serving-endpoints/emhol-xgboost/invocations'
headers = {'Authorization': f'Bearer {dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()}'}



response = requests.post(
    endpoint_url,
    headers=headers,
    json={
        "dataframe_split": X_test.iloc[:4].to_dict(orient="split")
    }
)

print(response.json())

# COMMAND ----------

y_test.iloc[:4]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Champion / Challenger workflow
# MAGIC - Train new challenger model

# COMMAND ----------

# DBTITLE 1,Cell 18
from mlflow.models import infer_signature
pred_proba = {'proba': model.predict_proba(X_train)[:,1]}
signature = infer_signature(X_train, 
            pd.DataFrame(pred_proba))
with mlflow.start_run(run_name="run-churn-emhol-xgb-challenger", tags={"Environment" :"dev"}):

    challenger = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=99
    )

    challenger.fit(X_train, y_train)
    preds = challenger.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)

    mlflow.log_metric("roc_auc", auc)

    model_info = mlflow.xgboost.log_model(
        challenger,
        name="model",
        input_example=X_train.head(3),
        signature=signature, 
        registered_model_name=model_name
    )

    # Set challenger alias to the new model version
    client.set_registered_model_alias(
        name=model_name,
        alias="challenger",
        version=model_info.registered_model_version
    )
    
    print(f"Model version {model_info.registered_model_version} registered with 'challenger' alias")
    print(f"ROC AUC: {auc}")

# COMMAND ----------

champion_version = mlf_client.get_model_version_by_alias(
            model_name, "champion"
            ).run_id

champion_auc = mlf_client.get_metric_history(champion_version, "roc_auc")[-1].value

challenger_version = mlf_client.get_model_version_by_alias(
            model_name, "challenger"
            ).run_id
challenger_auc = mlf_client.get_metric_history(challenger_version, "roc_auc")[-1].value

print(f"AUC for Challenger is {challenger_auc}")
print(f"AUC for Champion is {champion_auc}")
if challenger_auc > champion_auc:
    
    print("Challenger is better than Champion")

    #mlf_client = mlflow.MlflowClient()
    model_version = mlf_client.get_model_version_by_alias(model_name, "challenger").version
    mlf_client.set_registered_model_alias(model_name, "champion", model_version)
    mlf_client.delete_registered_model_alias(model_name, "challenger")
    new_version = mlf_client.get_model_version_by_alias(model_name, "champion").version

    print(f"Challenger is promoted to Champion and version is now {new_version}") 
    
else:
    print("Champion is better than Challenger")
   

# COMMAND ----------


