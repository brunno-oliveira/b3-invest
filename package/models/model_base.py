from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from data_split import DataSplit
from model_grid_search import GridSearch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import logging
import wandb
import os
from abc import abstractmethod

sns.set_theme(style="darkgrid")

log = logging.getLogger(__name__)

TRAIN_MAX_DATE = "2021-05-18"


class ModelBase(GridSearch):
    def __init__(self, group_name: str, model_name: str, model_folder: str):
        # model_type = model_type.lower()

        # if model_type not in ['best', 'simple']:
        #     msg = f'Atencao! model_type: {model_type} nao configurado'
        #     log.error(msg)
        #     raise Exception(msg)

        self.group_name = group_name
        self.model_name = model_name
        # Path
        root_path = root_path = os.path.dirname(
            os.path.dirname(os.path.dirname(__file__))
        )
        self.data_path = os.path.join(root_path, "data")
        self.current_path = os.path.dirname(__file__)
        self.model_path = os.path.join(self.current_path, model_folder)

        super().__init__(self.model_path)

        # Models
        self.model = None
        self.df: pd.DataFrame = None
        self.y_data: np.ndarray = None
        self.predicted: np.ndarray = None

        # Metrics
        self.mape_score: float = None
        self.mae_score: float = None
        self.mse_score: float = None
        self.r_square_score: float = None

    def load_data(self):
        log.info("Start")
        self.df = pd.read_parquet(
            os.path.join(self.data_path, "df_consolidado.parquet")
        )
        self.train_test_split()

    @property
    @abstractmethod
    def set_model(self, model_type: str = 'best'):
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
            self.r_square_score,
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
                "r2_score": self.r_square_score,
            }
        )

        wandb.finish()

    def plot_metrics(self, plot_graph: bool = False):
        self.mape_score = round(
            mean_absolute_percentage_error(self.predicted, self.y_test), 4
        )
        self.mae_score = round(mean_absolute_error(self.predicted, self.y_test), 4)
        self.mse_score = round(mean_squared_error(self.predicted, self.y_test), 4)
        self.r_square_score = round(r2_score(self.predicted, self.y_test), 4)

        log.info(f"mape_score : {self.mape_score}")
        log.info(f"mae_score : {self.mae_score}")
        log.info(f"mse_score : {self.mse_score}")
        log.info(f"r2_score : {self.r_square_score}")

        if plot_graph:
            fig, ax = plt.subplots(figsize=(30, 6))

            sns.lineplot(x=list(self.y_data["ticker"]), y=list(self.y_data["close"]))
            ax.plot(self.predicted)
            plt.xticks(rotation=90)

            plt.show()
