import json
import logging
import os
from abc import abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from numpy.core.numeric import roll
import pandas as pd
import seaborn as sns
import wandb
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
        # Path
        root_path = root_path = os.path.dirname(
            os.path.dirname(os.path.dirname(__file__))
        )
        self.data_path = os.path.join(root_path, "data")
        self.current_path = os.path.dirname(__file__)
        self.model_path = os.path.join(self.current_path, model_folder)

        super().__init__(self.model_path, self.model_type)

        # Models
        self.model = None
        self.df: pd.DataFrame = None
        self.y_data: np.ndarray = None
        self.predicted: np.ndarray = None

        # Metrics
        self.mape_score: float = None
        self.mae_score: float = None
        self.mse_score: float = None
        self.rmse_score: float = None
        self.r_square_score: float = None
        self.mean_squared_log_error: float = None

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
        return self.model, self.predicted

    def fit(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        self.predicted = self.model.predict(self.X_test)

    def plot_wandb(self):
        if None in [
            self.mape_score,
            self.mae_score,
            self.mse_score,
            self.rmse_score,
            self.r_square_score,
            self.mean_squared_log_error
        ]:
            error = "Error: Empty Metrics. Run plot_metrics before plot_wandb"
            log.error(error)
            raise Exception(error)

        wandb.init(
            project="b3-invest",
            entity="brunno-oliveira",
            group=self.group_name,
            name=self.model_name,
        )

        wandb.log(
            {
                "mape_score": self.mape_score,
                "mae_score": self.mae_score,
                "mse_score": self.mse_score,
                "rmse_score": self.rmse_score,
                "r2_score": self.r_square_score,
                "msle_score": self.mean_squared_log_error
            }
        )

        wandb.finish()

    def plot_metrics(self, plot_graph: bool = False):
        self.mape_score = round(
            mean_absolute_percentage_error(self.predicted, self.y_test), 4
        )
        self.mae_score = round(mean_absolute_error(self.predicted, self.y_test), 4)
        self.mse_score = round(mean_squared_error(self.predicted, self.y_test), 4)
        self.rmse_score = round(mean_squared_error(self.predicted, self.y_test, squared=True), 4)
        self.r_square_score = round(r2_score(self.predicted, self.y_test), 4)
        self.self.mean_squared_log_error = round(mean_squared_log_error(self.predicted, self.y_test), 4)

        log.info(f"mape_score : {self.mape_score}")
        log.info(f"mae_score : {self.mae_score}")
        log.info(f"mse_score : {self.mse_score}")
        log.info(f"rmse_score : {self.rmse_score}")
        log.info(f"r2_score : {self.r_square_score}")
        log.info(f"msle_score : {self.mean_squared_log_error}")

        if plot_graph:
            fig, ax = plt.subplots(figsize=(30, 6))

            sns.lineplot(x=list(self.y_data["ticker"]), y=list(self.y_data["close"]))
            ax.plot(self.predicted)
            plt.xticks(rotation=90)

            plt.show()
