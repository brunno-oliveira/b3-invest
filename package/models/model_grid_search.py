import io
import json
import logging
import multiprocessing
import os
from typing import Dict

import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.model_selection import GridSearchCV

from data_split import DataSplit

log = logging.getLogger(__name__)


class GridSearch(DataSplit):
    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path
        self.gs: GridSearchCV = None
        self.gs_params: Dict = None
        self.gs_result: DataFrame = None
        self.gs_best_params_path = os.path.join(self.model_path, "best_params.json")

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
