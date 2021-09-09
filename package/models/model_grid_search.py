import io
import json
import logging
import multiprocessing
import os
from typing import Dict, List

import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.model_selection import GridSearchCV, PredefinedSplit

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
        # Arquivo utilizado para grid search
        grid_path = os.path.join(self.model_path, "gs_params.json")
        with open(grid_path) as json_file:
            self.gs_params = json.load(json_file)["params"]

    def test_index(self) -> List[int]:
        """Funcao que gera a lista de index que da base self.df
            sue era gerada. 
            0: Treino
            1: Test
            -1: Skip 
        """
        index_list = [0 for _ in range(len(self.df))]
        index_test_list = list(self.df[self.df["date"] > "2021-05-18"].index)            

        for index in index_test_list:
            index_list[index] = 1  

        return index_list    


    def grid_search(self):
        log.info("Start")
        if self.gs_params is None:
            self.load_grid()

        cv_split = PredefinedSplit(self.test_index())

        self.gs = GridSearchCV(
            estimator=self.model,
            param_grid=self.gs_params,
            cv=cv_split,
            n_jobs=multiprocessing.cpu_count(),
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

        self.gs_result.to_csv(os.path.join(gs_path, "gs_results.csv"))

        gs_best_params = json.dumps(self.gs.best_params_)
        with io.open(os.path.join(gs_path, "best_params.json"), "w") as f:
            f.write(gs_best_params)

    def grid_search_splitter(self):
        log.info("Start")
