import io
import json
import logging
import multiprocessing
import os
from typing import Dict, Tuple

import pandas as pd
from pandas.core.frame import DataFrame
from pandas.core.indexes.numeric import Int64Index
from sklearn.model_selection import GridSearchCV

from data_split import DataSplit
from model_type import ModelType

log = logging.getLogger(__name__)


class GridSearch(DataSplit):
    def __init__(self, model_path: str, model_type: ModelType):
        super().__init__()
        self.model_path = model_path
        self.model_type = model_type
        self.gs: GridSearchCV = None
        self.gs_params: Dict = None
        self.gs_result: DataFrame = None

    def load_grid(self):
        """Carrega o JSON com os parâmetros a serem utilizados
        no grid search"""
        grid_path = os.path.join(self.model_path, "gs_params.json")
        with open(grid_path) as json_file:
            self.gs_params = json.load(json_file)["params"]

    def train_test_index(self) -> Tuple[Int64Index]:
        """Train e Test set customizados com base nos experimentos.
        O Train set é sempre o mesmo, o que muda são os tests"""
        n_experiments = 4
        train = tuple(self.X_train.index for _ in range(n_experiments))

        test = (
            self.y_test_1_day.index,
            self.y_test_7_days.index,
            self.y_test_14_days.index,
            self.y_test_28_days.index,
        )

        return zip(train, test)

    def grid_search(self):
        log.info("Start")
        if self.gs_params is None:
            self.load_grid()

        self.gs = GridSearchCV(
            estimator=self.model,
            param_grid=self.gs_params,
            cv=self.train_test_index(),
            n_jobs=multiprocessing.cpu_count(),
            scoring="neg_root_mean_squared_error",
            verbose=2,
        )

        self.gs.fit(self.X_train_gs, self.y_train_gs)

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

        self.gs_result = round(self.gs_result, 2)

        # Save results and best params
        if self.model_type == ModelType.WITHOUT_FEATURES:
            gs_path = os.path.join(self.model_path, "wo_features")
        elif self.model_type == ModelType.WITH_FEATURES:
            gs_path = os.path.join(self.model_path, "with_features")

        self.gs_result.to_csv(os.path.join(gs_path, "gs_results.csv"), inedx=False)

        gs_best_params = json.dumps(self.gs.best_params_)
        with io.open(os.path.join(gs_path, "best_params.json"), "w") as f:
            f.write(gs_best_params)

    def grid_search_splitter(self):
        log.info("Start")
