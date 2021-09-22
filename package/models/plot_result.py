import logging
import os
import pickle
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml

sns.set_theme(style="darkgrid")

log = logging.getLogger(__name__)


class PlotResults:
    def __init__(self):
        self.root_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.data_path = os.path.join(self.root_path, "data")
        self.docs_path = os.path.join(self.root_path, "docs")
        self.docs_imagens_path = os.path.join(self.docs_path, "imagens")

        with open(
            os.path.join(self.root_path, "package", "config.yml"), "r"
        ) as ymlfile:
            self.cfg = yaml.safe_load(ymlfile)

        self.dict_results: Dict = None
        self.df: pd.DataFrame = None
        self.df_metric: pd.DataFrame = None

    def show_results(self):
        log.info("Start")
        self.load_data()
        self.plot_treino_teste_data()
        log.info("Finished")

    def load_data(self):
        log.info("Start")
        self.df = pd.read_parquet(
            os.path.join(self.data_path, "df_consolidado.parquet")
        )
        self.dict_results = self.consolidate_results()
        self.df_metric = self.consolidade_metric()

    def plot_test_example(self, ticker: str = "PETR4"):
        log.info("Start")

    def plot_treino_teste_data(self):
        """Histórico de fechamento com marcação para a baixa devido a covid,
        e o período de testes.
        """
        log.info("Start")
        df_train_test = self.df[["ticker", "close", "date"]].copy()
        df_train_test["date"] = df_train_test["date"].dt.strftime("%Y-%m-%d")
        df_train_test = df_train_test.sort_values(by="date")
        df_train_test = df_train_test.reset_index()
        df_train_test.drop(columns="index", inplace=True)

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.grid(True)
        sns.lineplot(
            x=df_train_test["date"],
            y=df_train_test["close"],
            label="Valor do fechamento",
        )
        # Se não for explicar não plota
        # ax.axvline(205, 0, 1, color="r")

        cfg_grid = self.cfg["model"]["grid_search"]["experiments"]["28_days"]
        cfg_test = self.cfg["model"]["predict"]["experiments"]["28_days"]
        ax.axvspan(
            list(df_train_test["date"].unique()).index(str(cfg_grid["start_date"])),
            list(df_train_test["date"].unique()).index(str(cfg_grid["end_date"])),
            alpha=0.3,
            color="green",
            label="Datas de validação",
        )

        ax.axvspan(
            list(df_train_test["date"].unique()).index(str(cfg_test["start_date"])),
            list(df_train_test["date"].unique()).index(str(cfg_test["end_date"])),
            alpha=0.3,
            color="blue",
            label="Datas de teste",
        )

        ax.xaxis.set_major_locator(plt.MaxNLocator(20))
        plt.xticks(rotation=60)
        plt.ylabel("R$")
        plt.xlabel("Data")
        plt.title("Histórico de fechamento diário")
        plt.legend()
        fig.savefig(
            os.path.join(self.docs_imagens_path, "train_validation_test_data.jpeg"),
            bbox_inches="tight",
        )

    def consolidate_results(self) -> Dict:
        log.info("Start")
        results = {}
        result_path = os.path.join(self.root_path, "data", "results")
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

    def consolidade_metric(self):
        log.info("Start")
        models = [
            "decision_tree_regressor",
            "random_forest_regressor",
            "neural_network",
            "xgb_regressor",
        ]

        experiments = ["1_day", "7_days", "14_days", "28_days"]

        model_types = ["with_features", "wo_features"]

        dfs = []
        for model in models:
            for experiment in experiments:
                for model_type in model_types:

                    dfs.append(
                        pd.DataFrame(
                            {
                                "experiment": experiment,
                                "model": [model],
                                "model_type": [model_type],
                                "rmse": [
                                    self.dict_results[model][model_type][experiment][
                                        "metrics"
                                    ]["rmse"]
                                ],
                            }
                        )
                    )
        df = pd.concat(dfs)

        df = df.reset_index()
        df.drop(columns=["index"], inplace=True)
        return df
