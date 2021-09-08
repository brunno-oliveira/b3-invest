from pandas.core.frame import DataFrame
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

import multiprocessing
from data_split import DataSplit
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import logging
import wandb
import os
import io

from typing import Dict
from abc import abstractmethod

sns.set_theme(style="darkgrid")

log = logging.getLogger(__name__)

TRAIN_MAX_DATE = "2021-05-18"


class ModelBase(DataSplit):
    def __init__(self, group_name: str, model_name: str, model_folder: str):
        super().__init__()
        self.group_name = group_name
        self.model_name = model_name
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
        self.y_data: np.ndarray = None
        self.predicted: np.ndarray = None

        # Grid Search
        self.gs: GridSearchCV = None
        self.gs_params: Dict = None
        self.gs_result: DataFrame = None
        self.gs_best_params_path = os.path.join(self.model_path, "best_params.json")

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

        # Save results
        log.info("Saving GridSearch results and best params..")
        self.gs_result = pd.DataFrame(self.gs.cv_results_)

        round_columns = [
            "mean_fit_time",
            "std_fit_time",
            "mean_score_time",
            "std_score_time",
            "split0_test_score",
            "split1_test_score",
            "split2_test_score",
            "split3_test_score",
            "split4_test_score",
            "mean_test_score",
            "std_test_score",
        ]

        for col in round_columns:
            self.gs_result = round(self.gs_result, 4)

        self.gs_result.to_csv(os.path.join(self.model_path, "gs_results.csv"))

        # Save best params
        gs_best_params = json.dumps(self.gs.best_params_)
        with io.open(self.gs_best_params_path, "w") as f:
            f.write(gs_best_params)

    def grid_search_splitter(self):
        log.info("Start")

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
