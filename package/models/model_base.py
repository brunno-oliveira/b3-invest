import json
import logging
import os
import pickle
from abc import abstractmethod
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
)

from model_grid_search import GridSearch
from model_type import ModelType

sns.set_theme(style="darkgrid")

log = logging.getLogger(__name__)

TRAIN_MAX_DATE = "2021-05-18"


class ModelBase(GridSearch):
    def __init__(
        self, group_name: str, model_name: str, model_folder: str, model_type: ModelType
    ):
        """Classe base que todos modelos devem herdar. Contém todas as
            implementações necessárias.

        Args:
            group_name (str): Informação para o wandb
            model_name (str): Informação para o wandb
            model_folder (str): Nome da pasta que o modelo se encontra
            model_type (ModelType): Tipo do modelo que será gerado.
                ModelType.WITH_FEATURES: Todas as features carregadas.
                ModelType.WITHOUT_FEATURES: Somente dados de data.
        """

        self.group_name = group_name
        self.model_name = model_name
        self.model_type = model_type
        self.model_folder = model_folder
        # Path
        self.root_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.data_path = os.path.join(self.root_path, "data")
        self.current_path = os.path.dirname(__file__)
        self.model_path = os.path.join(self.current_path, self.model_folder)

        super().__init__(self.model_path, self.model_type)

        # Models
        self.model = None
        self.df: pd.DataFrame = None
        self.y_data: np.ndarray = None

        # Predicted
        self.predicted_1_day: np.ndarray = None
        self.predicted_7_days: np.ndarray = None
        self.predicted_14_days: np.ndarray = None
        self.predicted_28_days: np.ndarray = None

        # Template de dicionário para as métricas
        self.model_result: Dict = {
            "1_day": {
                "data": "",
                "metrics": {
                    "mape": "",
                    "mae": "",
                    "mse": "",
                    "rmse": "",
                    "r2": "",
                    "msle": "",
                },
            },
            "7_days": {
                "data": "",
                "metrics": {
                    "mape": "",
                    "mae": "",
                    "mse": "",
                    "rmse": "",
                    "r2": "",
                    "msle": "",
                },
            },
            "14_days": {
                "data": "",
                "metrics": {
                    "mape": "",
                    "mae": "",
                    "mse": "",
                    "rmse": "",
                    "r2": "",
                    "msle": "",
                },
            },
            "28_days": {
                "data": "",
                "metrics": {
                    "mape": "",
                    "mae": "",
                    "mse": "",
                    "rmse": "",
                    "r2": "",
                    "msle": "",
                },
            },
        }

    def load_data(self):
        log.info("Start")
        self.df = pd.read_parquet(
            os.path.join(self.data_path, "df_consolidado.parquet")
        )

        if self.model_type == ModelType.WITHOUT_FEATURES:
            log.info("Excutando com colunas limitadas")
            valid_columns = ["close", "ticker", "date", "year", "month", "day"]
            for col in self.df.columns:
                if "ticker." in col:
                    valid_columns.append(col)
            self.df = self.df[valid_columns]
        else:
            log.info("Excutando com todas as colunas")
        self.train_test_split(self.model_type)

    @property
    @abstractmethod
    def set_model(self):
        pass

    def fit_and_predict(self):
        log.info("Start")
        self.fit()
        self.predict()

    def fit(self):
        log.info("Start")
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        log.info("Predict 1 day..")
        self.predicted_1_day = self.model.predict(self.X_test_1_day)
        self.test_data_1_day["predicted"] = self.predicted_1_day
        self.model_result["1_day"].update({"data": self.test_data_1_day.to_dict()})

        log.info("Predict 7 days..")
        self.predicted_7_days = self.model.predict(self.X_test_7_days)
        self.test_data_7_days["predicted"] = self.predicted_7_days
        self.model_result["7_days"].update({"data": self.test_data_7_days.to_dict()})

        log.info("Predict 14 days..")
        self.predicted_14_days = self.model.predict(self.X_test_14_days)
        self.test_data_14_days["predicted"] = self.predicted_14_days
        self.model_result["14_days"].update({"data": self.test_data_14_days.to_dict()})

        log.info("Predict 28 days..")
        self.predicted_28_days = self.model.predict(self.X_test_28_days)
        self.test_data_28_days["predicted"] = self.predicted_28_days
        self.model_result["28_days"].update({"data": self.test_data_28_days.to_dict()})

        log.info("Done")

    def run_metrics(self):
        log.info("Running Metrics...")
        self.create_metrics()
        self.log_metrics()
        self.plot_graphs()
        self.save_results()

    def save_results(self):
        log.info("Start")
        pickle_path = os.path.join(self.result_path(), "result.pickle")
        with open(pickle_path, "wb") as handle:
            pickle.dump(self.model_result, handle)

        metrics_path = os.path.join(self.result_path(), "rmse")
        metrics = {
            "1_day": self.model_result["1_day"]["metrics"],
            "7_days": self.model_result["7_days"]["metrics"],
            "14_days": self.model_result["14_days"]["metrics"],
            "28_days": self.model_result["28_days"]["metrics"],
        }

        with open(f"{metrics_path}.json", "a") as f:
            json.dump(metrics, f, indent=4)

    def log_metrics(self):
        log.info("Start")
        for key, _ in self.model_result.items():
            log.info(f"----- {key} metrics -----")
            for metric, value in self.model_result[key]["metrics"].items():
                log.info(f"{metric}: {value}")

    def result_path(self) -> str:
        data_path = os.path.join(self.root_path, "data")
        result_path = os.path.join(data_path, "results")
        result_path = os.path.join(result_path, self.model_folder)

        if self.model_type == ModelType.WITH_FEATURES:
            feature_path = "with_features"
        else:
            feature_path = "wo_features"

        return os.path.join(result_path, feature_path)

    def plot_graphs(self):
        log.info("Start")

        for key, _ in self.model_result.items():
            fig, ax = plt.subplots(figsize=(30, 6))

            x = [_ for _ in range(len(pd.DataFrame(self.model_result[key]["data"])))]

            sns.lineplot(x=x, y=pd.DataFrame(self.model_result[key]["data"])["close"])
            ax.plot(pd.DataFrame(self.model_result[key]["data"])["predicted"])

            fig.savefig(os.path.join(self.result_path(), f"{key}.jpeg"))

    def create_metrics(self):
        """Carrega todas as métricas para os 4 experimentos em um dicionário"""

        def root_mean_squared_error(predicted, y):
            return mean_squared_error(predicted, y, squared=False)

        func_metrics = [
            mean_absolute_percentage_error,
            mean_absolute_error,
            mean_squared_error,
            root_mean_squared_error,
            r2_score,
            mean_squared_log_error,
        ]

        key_metrics = ["mape", "mae", "mse", "rmse", "r2", "msle"]

        for key, _ in self.model_result.items():
            for func, metric_key in zip(func_metrics, key_metrics):
                self.model_result[key]["metrics"][metric_key] = round(
                    func(
                        pd.DataFrame(self.model_result[key]["data"])["predicted"],
                        pd.DataFrame(self.model_result[key]["data"])["close"],
                    ),
                    4,
                )
