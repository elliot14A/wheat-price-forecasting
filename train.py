from datetime import datetime
import sys
from pathlib import Path
from urllib.parse import urlparse
import autots
from autots import AutoTS
import mlflow
import pandas as pd
import logging
import numpy as np
import warnings


class TimeSeriesModel:
    def __init__(self,
                 file_path, forecast_length=3,
                 frequency="infer", prediction_interval=0.9,
                 ensemble="simple", model_list="fast",
                 transformer_list="fast", max_generations=5,
                 num_validations=2, validation_method="backwards"):
        self.file_path = file_path
        self.forecast_length = forecast_length
        self.frequency = frequency
        self.prediction_interval = prediction_interval
        self.ensemble = ensemble
        self.model_list = model_list
        self.transformer_list = transformer_list
        self.max_generations = max_generations
        self.num_validations = num_validations
        self.validation_method = validation_method
        self.logger = logging.getLogger(__name__)

    def _get_train_test_split(self, df, start_year_month=(2017, 1), end_year_month=(2023, 1)):
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
        train = df[(df["date"] >= train_start_date)
                   & (df["date"] < train_end_date)]
        test = df[(df["date"] >= train_end_date)]
        return train, test

    def train(self, centre, variety):
        """
        Train the AutoTS model on the given centre and variety.

        Args:
            centre (str): The name of the centre.
            variety (str): The name of the variety.

        Returns:
            None
        """

        warnings.filterwarnings("ignore")
        np.random.seed(40)

        self.logger.info(f"Centre: {centre}, Variety: {variety}")

        try:
            data = pd.read_excel(self.file_path)
            data = data[(data["centre"] == centre) &
                        (data["variety"] == variety)]
            self.logger.info(f"Data shape: {data.shape}")
        except Exception as e:
            self.logger.exception(
                "Unable to read dataset, check the file path. Error: %s",
                e
            )
            raise
        else:
            self.logger.info("Data read successfully")

        train, test = self._get_train_test_split(data)

        with mlflow.start_run():
            self.logger.info("Starting AutoTS model training")
            model = AutoTS(forecast_length=self.forecast_length,
                           frequency=self.frequency,
                           prediction_interval=self.prediction_interval,
                           ensemble=self.ensemble,
                           model_list=self.model_list,
                           transformer_list=self.transformer_list,
                           max_generations=self.max_generations,
                           num_validations=self.num_validations,
                           validation_method=self.validation_method
                           )
            model = model.fit(train, date_col="date",
                              value_col="value", id_col="centre")
            validations = model.results("validation")
            best_model = validations[validations["ID"] == model.best_model_id]
            (model_name, rmse, mae, smape, auto_ts_score) = (
                best_model["Model"].values[0],
                best_model["rmse"].values[0],
                best_model["mae"].values[0],
                best_model["smape"].values[0],
                best_model["Score"].values[0]
            )

            self.logger.info(f"Best model: {model_name}")
            self.logger.info(f"RMSE: {rmse}")
            self.logger.info(f"MAE: {mae}")
            self.logger.info(f"SMAPE: {smape}")
            self.logger.info(f"AutoTS Score: {auto_ts_score}")

            mlflow.log_param("forecast_length", self.forecast_length)
            mlflow.log_param("Centre", centre)
            mlflow.log_param("Variety", variety)
            mlflow.log_param("Model", model_name)
            mlflow.log_param("RMSE", rmse)
            mlflow.log_param("MAE", mae)
            mlflow.log_param("SMAPE", smape)
            self.logger.info("AutoTS model training finished")

            tracking_url_type_store = urlparse(
                mlflow.get_tracking_uri()).scheme
            self.logger.info(f"Tracking URL: {tracking_url_type_store}")

            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(
                    model, "model", registered_model_name="Auto-TS")
            else:
                mlflow.sklearn.log_model(model, "model")
            self.logger.info("Model logged successfully")


if __name__ == "__main__":
    filepath = Path(__file__).resolve().parents[0] / "dataset.xlsx"

    if not filepath.exists():
        raise ValueError("dataset.xlsx file not found")

    centre, variety = sys.argv[1], sys.argv[2]
    if not (centre and variety):
        raise ValueError("Centre and Variety not provided")

    timeseriesmodel = TimeSeriesModel(filepath)

    timeseriesmodel.train(centre, variety)
