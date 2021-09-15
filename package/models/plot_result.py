import os
import logging
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict

sns.set_theme(style="darkgrid")

log = logging.getLogger(__name__)


class PlotResults:
    def show_results(self):
        log.info("Start")
        results = self.consolidate_results()
        df_metric = self.consolidade_metric(results)
        log.info("Finished")

    @staticmethod
    def consolidate_results() -> Dict:
        log.info("Start")
        results = {}
        root_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        result_path = os.path.join(root_path, "data", "results")
        for model in os.listdir(result_path):
            for model_type in os.listdir(os.path.join(result_path, model)):
                pickle_path = os.path.join(
                    result_path, model, model_type, "result.pickle"
                )
                with open(pickle_path, "rb") as handle:
                    result = pickle.load(handle)
                if not (model in results):
                    results[model] = {}
                results[model][model_type] = {}
                results[model][model_type].update(result)
        return results

    @staticmethod
    def consolidade_metric(results: Dict) -> pd.DataFrame:
        log.info("Start")
        return pd.concat(
            [
                # 1 day
                pd.DataFrame(
                    {
                        "experiment": "1_day",
                        "model": ["decision_tree_regressor"],
                        "model_type": ["with_features"],
                        "rmse": [
                            results["decision_tree_regressor"]["with_features"][
                                "1_day"
                            ]["metrics"]["rmse"]
                        ],
                    }
                ),
                pd.DataFrame(
                    {
                        "experiment": "1_day",
                        "model": ["decision_tree_regressor"],
                        "model_type": ["wo_features"],
                        "rmse": [
                            results["decision_tree_regressor"]["wo_features"]["1_day"][
                                "metrics"
                            ]["rmse"]
                        ],
                    }
                ),
                pd.DataFrame(
                    {
                        "experiment": "1_day",
                        "model": ["random_forest_regressor"],
                        "model_type": ["with_features"],
                        "rmse": [
                            results["random_forest_regressor"]["with_features"][
                                "1_day"
                            ]["metrics"]["rmse"]
                        ],
                    }
                ),
                pd.DataFrame(
                    {
                        "experiment": "1_day",
                        "model": ["random_forest_regressor"],
                        "model_type": ["wo_features"],
                        "rmse": [
                            results["random_forest_regressor"]["wo_features"]["1_day"][
                                "metrics"
                            ]["rmse"]
                        ],
                    }
                ),
                # 7 days
                pd.DataFrame(
                    {
                        "experiment": "7_days",
                        "model": ["decision_tree_regressor"],
                        "model_type": ["with_features"],
                        "rmse": [
                            results["decision_tree_regressor"]["with_features"][
                                "7_days"
                            ]["metrics"]["rmse"]
                        ],
                    }
                ),
                pd.DataFrame(
                    {
                        "experiment": "7_days",
                        "model": ["decision_tree_regressor"],
                        "model_type": ["wo_features"],
                        "rmse": [
                            results["decision_tree_regressor"]["wo_features"]["7_days"][
                                "metrics"
                            ]["rmse"]
                        ],
                    }
                ),
                pd.DataFrame(
                    {
                        "experiment": "7_days",
                        "model": ["random_forest_regressor"],
                        "model_type": ["with_features"],
                        "rmse": [
                            results["random_forest_regressor"]["with_features"][
                                "7_days"
                            ]["metrics"]["rmse"]
                        ],
                    }
                ),
                pd.DataFrame(
                    {
                        "experiment": "7_days",
                        "model": ["random_forest_regressor"],
                        "model_type": ["wo_features"],
                        "rmse": [
                            results["random_forest_regressor"]["wo_features"]["7_days"][
                                "metrics"
                            ]["rmse"]
                        ],
                    }
                ),
                # 14 days
                pd.DataFrame(
                    {
                        "experiment": "14_days",
                        "model": ["decision_tree_regressor"],
                        "model_type": ["with_features"],
                        "rmse": [
                            results["decision_tree_regressor"]["with_features"][
                                "14_days"
                            ]["metrics"]["rmse"]
                        ],
                    }
                ),
                pd.DataFrame(
                    {
                        "experiment": "14_days",
                        "model": ["decision_tree_regressor"],
                        "model_type": ["wo_features"],
                        "rmse": [
                            results["decision_tree_regressor"]["wo_features"][
                                "14_days"
                            ]["metrics"]["rmse"]
                        ],
                    }
                ),
                pd.DataFrame(
                    {
                        "experiment": "14_days",
                        "model": ["random_forest_regressor"],
                        "model_type": ["with_features"],
                        "rmse": [
                            results["random_forest_regressor"]["with_features"][
                                "14_days"
                            ]["metrics"]["rmse"]
                        ],
                    }
                ),
                pd.DataFrame(
                    {
                        "experiment": "14_days",
                        "model": ["random_forest_regressor"],
                        "model_type": ["wo_features"],
                        "rmse": [
                            results["random_forest_regressor"]["wo_features"][
                                "14_days"
                            ]["metrics"]["rmse"]
                        ],
                    }
                ),
                # 28 days
                pd.DataFrame(
                    {
                        "experiment": "28_days",
                        "model": ["decision_tree_regressor"],
                        "model_type": ["with_features"],
                        "rmse": [
                            results["decision_tree_regressor"]["with_features"][
                                "28_days"
                            ]["metrics"]["rmse"]
                        ],
                    }
                ),
                pd.DataFrame(
                    {
                        "experiment": "28_days",
                        "model": ["decision_tree_regressor"],
                        "model_type": ["wo_features"],
                        "rmse": [
                            results["decision_tree_regressor"]["wo_features"][
                                "28_days"
                            ]["metrics"]["rmse"]
                        ],
                    }
                ),
                pd.DataFrame(
                    {
                        "experiment": "28_days",
                        "model": ["random_forest_regressor"],
                        "model_type": ["with_features"],
                        "rmse": [
                            results["random_forest_regressor"]["with_features"][
                                "28_days"
                            ]["metrics"]["rmse"]
                        ],
                    }
                ),
                pd.DataFrame(
                    {
                        "experiment": "28_days",
                        "model": ["random_forest_regressor"],
                        "model_type": ["wo_features"],
                        "rmse": [
                            results["random_forest_regressor"]["wo_features"][
                                "28_days"
                            ]["metrics"]["rmse"]
                        ],
                    }
                ),
            ]
        )
