import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import autots
from autots import AutoTS

import logging
from pathlib import Path
from datetime import datetime


def get_train_test_split(df, start_year_month=(2017, 1), end_year_month=(2023, 1)):
    """Function to split dataset into train and test

    Args:
        df (_type_): _description_
        start_year_month (tuple, optional): _description_. Defaults to (2017,1).
        end_year_month (tuple, optional): _description_. Defaults to (2022,1).

    Returns:
        _type_: _description_
    """
    train_start_date = datetime(*start_year_month, 1)
    train_end_date = datetime(*end_year_month, 1)
    train = df[(df["date"] >= train_start_date) & (df["date"] < train_end_date)]
    test = df[(df["date"] >= train_end_date)]
    return train, test


FILE_PATH = Path(__file__).resolve()
DIR_FOLDER = FILE_PATH.parents[0]
DATASET = DIR_FOLDER / "dataset.xlsx"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    centre, variety = sys.argv[1], sys.argv[2]
    if not (centre and variety):
        raise ValueError("Centre and Variety are required")
    logger.info(f"Centre: {centre}, Variety: {variety}")

    try:
        data = pd.read_excel(DATASET)
        data = data[(data["centre"] == centre) & (data["variety"] == variety)]
        logger.info(f"Data shape: {data.shape}")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s",
            e,
        )
        raise
    else:
        logger.info("Data downloaded successfully")

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = get_train_test_split(data)

    with mlflow.start_run():
        logger.info("Starting AutoTS model training")
        model = AutoTS(
            forecast_length=3,
            frequency="infer",
            prediction_interval=0.9,
            ensemble="simple",
            model_list="fast",  # "superfast", "default", "fast_parallel"
            transformer_list="fast",  # "superfast",
            max_generations=5,
            num_validations=2,
            validation_method="backwards",
        )

        model = model.fit(train, date_col="date", value_col="value", id_col="centre")
        validations = model.results("validation")
        best_model = validations[validations["ID"] == model.best_model_id]
        (model_name, rmse, mae, smape, auto_ts_score) = (
            best_model["Model"].values[0],
            best_model["rmse"].values[0],
            best_model["mae"].values[0],
            best_model["smape"].values[0],
            best_model["Score"].values[0],
        )

        logger.info(f"Best model: {model_name}")
        logger.info(f"RMSE: {rmse}")
        logger.info(f"MAE: {mae}")
        logger.info(f"SMAPE: {smape}")
        logger.info(f"AutoTS Score: {auto_ts_score}")

        mlflow.log_param("forecast_length", 3)
        mlflow.log_param("Centre", centre)
        mlflow.log_param("Variety", variety)
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("smape", smape)
        mlflow.log_metric("auto_ts_score", auto_ts_score)
        logging.info("AutoTS  model training finished")

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        logger.info(f"Tracking URL type store: {tracking_url_type_store}")

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(model, "model", registered_model_name="Auto-TS")
        else:
            mlflow.sklearn.log_model(model, "model")
        logger.info("Model logged successfully")
