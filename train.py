import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import json
import pandas as pd
import numpy as np
import requests
import zipfile
import io

import plotly.offline as py #working offline
import plotly.graph_objs as go

from evidently.pipeline.column_mapping import ColumnMapping
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

py.init_notebook_mode()

"""This notebook shows how you can use Evidently to check the data for data drift.

Acknowledgments:

The dataset used in the example is from: https://www.kaggle.com/c/bike-sharing-demand/data?select=train.csv
Fanaee-T, Hadi, and Gama, Joao, 'Event labeling combining ensemble detectors and background knowledge', Progress in Artificial Intelligence (2013): pp. 1-15, Springer Berlin Heidelberg
More information about the dataset can be found in UCI machine learning repository: https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset

## Load Data
"""

content = requests.get("https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip").content
with zipfile.ZipFile(io.BytesIO(content)) as arc:
    raw_data = pd.read_csv(arc.open("day.csv"), header=0, sep=',', parse_dates=['dteday'], index_col='dteday')

#observe data structure
raw_data.head()

#set column mapping for Evidently Profile
data_columns = ColumnMapping()
data_columns.numerical_features = ['weathersit', 'temp', 'atemp', 'hum', 'windspeed']

#set reference dates
reference_dates = ('2011-01-01 00:00:00','2011-01-28 23:00:00')

#set experiment batches dates
experiment_batches = [
    ('2011-02-01 00:00:00','2011-02-28 23:00:00'),
    ('2011-03-01 00:00:00','2011-03-31 23:00:00'),
    ('2011-04-01 00:00:00','2011-04-30 23:00:00'),
    ('2011-05-01 00:00:00','2011-05-31 23:00:00'),  
    ('2011-06-01 00:00:00','2011-06-30 23:00:00'), 
    ('2011-07-01 00:00:00','2011-07-31 23:00:00'), 
]

"""## Functions to clculate drift with Evidently"""

#evaluate data drift with Evidently Profile
def detect_dataset_drift(reference, production, column_mapping, get_ratio=False):
    """
    Returns True if Data Drift is detected, else returns False.
    If get_ratio is True, returns ration of drifted features.
    The Data Drift detection depends on the confidence level and the threshold.
    For each individual feature Data Drift is detected with the selected confidence (default value is 0.95).
    Data Drift for the dataset is detected if share of the drifted features is above the selected threshold (default value is 0.5).
    """
    
    data_drift_profile = Profile(sections=[DataDriftProfileSection()])
    data_drift_profile.calculate(reference, production, column_mapping=column_mapping)
    report = data_drift_profile.json()
    json_report = json.loads(report)
        
    n_features = json_report["data_drift"]["data"]["metrics"]["n_features"]
    n_drifted_features = json_report["data_drift"]["data"]["metrics"]["n_drifted_features"]
    
    if get_ratio:
        return n_drifted_features / n_features
    else:
        return json_report["data_drift"]["data"]["metrics"]["dataset_drift"]

#evaluate data drift with Evidently Profile
def detect_features_drift(reference, production, column_mapping, get_scores=False):
    """
    Returns 1 if Data Drift is detected, else returns 0. 
    If get_scores is True, returns scores value (like p-value) for each feature.
    The Data Drift detection depends on the confidence level and the threshold.
    For each individual feature Data Drift is detected with the selected confidence (default value is 0.95).
    """
    
    data_drift_profile = Profile(sections=[DataDriftProfileSection()])
    data_drift_profile.calculate(reference, production, column_mapping=column_mapping)
    report = data_drift_profile.json()
    json_report = json.loads(report)
    
    drifts = []
    num_features = column_mapping.numerical_features if column_mapping.numerical_features else []
    cat_features = column_mapping.categorical_features if column_mapping.categorical_features else []
    for feature in num_features + cat_features:
        drift_score = json_report['data_drift']['data']['metrics'][feature]['drift_score']
        if get_scores:
            drifts.append((feature, drift_score))
        else:
            drifts.append((feature, json_report['data_drift']['data']['metrics'][feature]['drift_detected']))
             
    return drifts

"""## Features Drift"""

features_historical_drift = []

for date in experiment_batches:
    drifts = detect_features_drift(raw_data.loc[reference_dates[0]:reference_dates[1]], 
                           raw_data.loc[date[0]:date[1]], 
                           column_mapping=data_columns)
    
    features_historical_drift.append([x[1] for x in drifts])
    
features_historical_drift_frame = pd.DataFrame(features_historical_drift, 
                                               columns = data_columns.numerical_features)

fig = go.Figure(data=go.Heatmap(
                   z = features_historical_drift_frame.astype(int).transpose(),
                   x = [x[1] for x in experiment_batches],
                   y = data_columns.numerical_features,
                   hoverongaps = False,
                   xgap = 1,
                   ygap = 1,
                   zmin = 0,
                   zmax = 1,
                   showscale = False,
                   colorscale = 'Bluered'
))

fig.update_xaxes(side="top")

fig.update_layout(
    xaxis_title = "Timestamp",
    yaxis_title = "Feature Drift"
)

fig.show()

features_historical_drift_pvalues = []

for date in experiment_batches:
    drifts = detect_features_drift(raw_data.loc[reference_dates[0]:reference_dates[1]], 
                           raw_data.loc[date[0]:date[1]],
                           column_mapping=data_columns,
                           get_scores=True)
    
    features_historical_drift_pvalues.append([x[1] for x in drifts])
    
features_historical_drift_pvalues_frame = pd.DataFrame(features_historical_drift_pvalues, 
                                                       columns = data_columns.numerical_features)

fig = go.Figure(data=go.Heatmap(
                   z = features_historical_drift_pvalues_frame.transpose(),
                   x = [x[1] for x in experiment_batches],
                   y = features_historical_drift_pvalues_frame.columns,
                   hoverongaps = False,
                   xgap = 1,
                   ygap = 1,
                   zmin = 0,
                   zmax = 1,
                   colorscale = 'reds_r'
                   )
               )

fig.update_xaxes(side="top")

fig.update_layout(
    xaxis_title = "Timestamp",
    yaxis_title = "p-value"
)

fig.show()

"""## Dataset Drift"""

dataset_historical_drift = []

for date in experiment_batches:
    dataset_historical_drift.append(detect_dataset_drift(raw_data.loc[reference_dates[0]:reference_dates[1]], 
                           raw_data.loc[date[0]:date[1]], 
                           column_mapping=data_columns))

fig = go.Figure(data=go.Heatmap(
                   z = [[1 if x == True else 0 for x in dataset_historical_drift]],
                   x = [x[1] for x in experiment_batches],
                   y = [''],
                   hoverongaps = False,
                   xgap = 1,
                   ygap = 1,
                   zmin = 0,
                   zmax = 1,
                   colorscale = 'Bluered',
                   showscale = False
                   )
               )

fig.update_xaxes(side="top")

fig.update_layout(
    xaxis_title = "Timestamp",
    yaxis_title = "Dataset Drift"
)
fig.show()

dataset_historical_drift_ratio = []

for date in experiment_batches:
    dataset_historical_drift_ratio.append(detect_dataset_drift(raw_data.loc[reference_dates[0]:reference_dates[1]], 
                           raw_data.loc[date[0]:date[1]],
                           column_mapping=data_columns,
                           get_ratio=True))

fig = go.Figure(data=go.Heatmap(
                   z = [dataset_historical_drift_ratio],
                   x = [x[1] for x in experiment_batches],
                   y = [''],
                   hoverongaps = False,
                   xgap = 1,
                   ygap = 1,
                   zmin = 0.5,
                   zmax = 1,
                   colorscale = 'reds'
                  )
               )

fig.update_xaxes(side="top")

fig.update_layout(
    xaxis_title = "Timestamp",
    yaxis_title = "Dataset Drift"
)
fig.show()

"""## Log Dataset Drift in MLFlow"""

#log into MLflow
client = MlflowClient()

#set experiment
mlflow.set_experiment('Dataset Drift Analysis with Evidently')

#start new run
for date in experiment_batches:
    with mlflow.start_run() as run: 
        
        # Log parameters
        mlflow.log_param("begin", date[0])
        mlflow.log_param("end", date[1])

        # Log metrics
        metric = detect_dataset_drift(raw_data.loc[reference_dates[0]:reference_dates[1]], 
                           raw_data.loc[date[0]:date[1]],
                           column_mapping=data_columns,
                           get_ratio=True)
        
        mlflow.log_metric('dataset drift', metric)

        print(run.info)
