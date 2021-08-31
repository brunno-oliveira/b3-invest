from pandas.core.frame import DataFrame
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import multiprocessing

from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import datetime as dt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import logging
import wandb
import os

from typing import Dict
from abc import abstractmethod

sns.set_theme(style="darkgrid")

log = logging.getLogger(__name__)


class ModelBase:
    def __init__(self, group_name: str, model_name: str, model_folder: str):
        # Path
        root_path = root_path = os.path.dirname(
            os.path.dirname(os.path.dirname(__file__))
        )
        self.data_path = os.path.join(root_path, "data")
        self.current_path = os.path.dirname(__file__)
        self.model_path = os.path.join(self.current_path, model_folder)

        # Models
        self.model = None
        self.df: pd.DataFrame = None
        self.X_train: pd.DataFrame = None
        self.y_train: pd.Series = None
        self.X_tes: pd.DataFramet = None
        self.y_test: pd.Series = None
        self.y_data: np.ndarray = None
        self.predicted: np.ndarray = None

        # Grid Search
        self.gs: GridSearchCV = None
        self.gs_params: Dict = None
        self.gs_result: DataFrame = None

        # Metrics
        self.mape_score: float = None
        self.mae_score: float = None
        self.mse_score: float = None
        self.r_square_score: float = None

        wandb.init(
            project="b3-invest",
            entity="brunno-oliveira",
            group=group_name,
            name=model_name,
        )

    def load_data(self):
        log.info("Start")
        self.df = pd.read_parquet(
            os.path.join(self.data_path, "df_consolidado.parquet")
        )

        # Coluna para facilitar a busca por tickers
        tickers = self.df.iloc[:, 0]
        self.df = self.df.iloc[:, 2:].copy()  # Remove ticker columns

        max_date = self.df["date"].max()
        # Previsao para o ultimo dia valido, removendo a primeira coluna (TARGET)
        self.X_train = self.df[self.df["date"] < max_date].iloc[:, 1:]
        self.y_train = self.df[self.df["date"] < max_date].iloc[:, 0]

        self.X_test = self.df[self.df["date"] == max_date].iloc[:, 1:]
        self.y_test = self.df[self.df["date"] == max_date].iloc[:, 0]

        # Devolvendo a coluna de ticker
        self.df["ticker"] = tickers

        # DF para facilitar a validacao
        self.y_data = self.df[self.df["date"] == self.df["date"].max()][
            ["ticker", "date", "close"]
        ].tail(len(self.X_test))

        self.transform_date()

    def load_grid(self):
        # Arquivo utilizado para grid search
        grid_path = os.path.join(self.model_path, "grid.json")
        with open(grid_path) as json_file:
            self.gs_params = json.load(json_file)["params"]

    def grid_search(self):
        log.info("Start")
        if self.gs_params is None:
            self.load_grid()
        self.gs = GridSearchCV(
            estimator=self.model,
            param_grid=self.gs_params,
            n_jobs=multiprocessing.cpu_count(),
            verbose=2,
        )

        self.gs.fit(self.X_train, self.y_train)
        self.gs_result = pd.DataFrame(self.gs.cv_results_)
        self.gs_result.to_csv(os.path.join(self.model_path, "gs_results.csv"))

    @property
    @abstractmethod
    def set_model(self):
        pass

    def fit_and_predict(self):
        log.info("Start")
        self.fit()
        self.predict()
        return self.model, self.predicted

    @property
    @abstractmethod
    def fit(self):
        pass

    @property
    @abstractmethod
    def predict(self):
        pass

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
        wandb.log(
            {
                "mape_score": self.mape_score,
                "mae_score": self.mae_score,
                "mse_score": self.mse_score,
                "r2_score": self.r_square_score,
            }
        )
        wandb.finish()

    def transform_date(self):
        """O modelo nao trabalha com o tipo datetime"""
        self.X_train["date"] = self.X_train["date"].map(dt.datetime.toordinal)
        self.X_test["date"] = self.X_test["date"].map(dt.datetime.toordinal)

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
