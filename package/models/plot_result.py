import logging
import os
import pickle
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from typing import List
from tqdm import tqdm

from sklearn.metrics import (
    mean_squared_error,
    r2_score,
)

sns.set_theme(style="darkgrid")

log = logging.getLogger(__name__)


class PlotResults:
    def __init__(self):
        self.root_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.data_path = os.path.join(self.root_path, "data")
        self.docs_path = os.path.join(self.root_path, "docs")
        self.docs_imagens_path = os.path.join(self.docs_path, "imagens")
        self.experimentos_path = os.path.join(self.docs_path, "experimentos")

        with open(
            os.path.join(self.root_path, "package", "config.yml"), "r"
        ) as ymlfile:
            self.cfg = yaml.safe_load(ymlfile)

        self.dict_results: Dict = None
        self.df: pd.DataFrame = None
        self.df_metric: pd.DataFrame = None
        self.df_experiments: pd.DataFrame = None

    def show_results(self):
        log.info("Start")
        self.load_data()
        self.df_experiments.to_parquet('df_experiments.parquet')
        # Não é necessário reprocessar esses plots abaixo
        # self.plot_treino_teste_data()
        # self.metrics_example()
        self.plot_experiments()
        log.info("Finished")

    def load_data(self):
        log.info("Start")
        self.df = pd.read_parquet(
            os.path.join(self.data_path, "df_consolidado.parquet")
        )
        self.dict_results = self.consolidate_result_file()
        self.df_metric, self.df_experiments = self.consolidade_data()
        self.metrics_by_ticker()

    def metrics_by_ticker(self) -> None:
        log.info("Start")
        df_rmse: List[pd.DataFrame] = []
        for experiment in self.df_experiments["experiment"].unique():
            log.info(f"experiment: {experiment}")
            for model in self.df_experiments["model"].unique():
                log.info(f"model: {model}")
                for model_type in self.df_experiments["model_type"].unique():
                    log.info(f"model_type: {model_type}")
                    for ticker in tqdm(list(self.df_experiments["ticker"].unique())):
                        df_close_predicted = self.df_experiments[
                            (self.df_experiments["experiment"] == experiment)
                            & (self.df_experiments["model"] == model)
                            & (self.df_experiments["model_type"] == model_type)
                            & (self.df_experiments["ticker"] == ticker)
                        ].copy()

                        if df_close_predicted.empty:
                            log.warning(
                                f"Sem dados para: {experiment} | {model} | {model_type} | {ticker}"
                            )
                            break

                        df_close_predicted["ticker_rmse"] = round(
                            self.root_mean_squared_error(
                                df_close_predicted["close"],
                                df_close_predicted["predicted"],
                            ),
                            4,
                        )
                        df_rmse.append(df_close_predicted.copy())
                        del df_close_predicted

        self.df_experiments = pd.concat(df_rmse)

    def metrics_example(self):
        log.info("Start")
        df = pd.DataFrame(
            self.dict_results["decision_tree_regressor"]["with_features"]["14_days"][
                "data"
            ]
        )
        df = df[df["ticker"] == "INEP3"]
        df = df.reset_index().drop(columns=["index"]).sort_values(by=["date"])
        df["date"] = df["date"].dt.date
        df = round(df, 4)

        r2 = round(r2_score(df["close"], df["predicted"]), 4)
        mse = round(mean_squared_error(df["close"], df["predicted"]), 4)
        rmse = round(self.root_mean_squared_error(df["close"], df["predicted"]), 4)
        log.info(f"r2: {r2}")
        log.info(f"mse: {mse}")
        log.info(f"rmse: {rmse}")
        df.rename(
            columns={
                "ticker": "Código Ação",
                "date": "Data",
                "close": "Fechamento",
                "predicted": "Predito",
            },
            inplace=True,
        )

        df.to_csv(os.path.join(self.docs_path, "exemplo_metricas.csv"), index=False)

    def plot_experiments(self):
        log.info("Start")
        for experiment in self.df_metric["experiment"].unique():
            log.info(f"Ploting {experiment} experiment")
            df = self.df_metric[self.df_metric["experiment"] == experiment][
                ["model", "wo_features", "with_features"]
            ]
            df.rename(
                columns={
                    "model": "Modelo",
                    "wo_features": "Sem Features",
                    "with_features": "Com Features",
                },
                inplace=True,
            )
            df.to_csv(
                os.path.join(self.experimentos_path, f"{experiment}.csv"), index=False
            )

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

    def consolidate_result_file(self) -> Dict:
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

    def consolidade_data(self) -> pd.DataFrame:
        """[summary]

        Returns:
            pd.DataFrame: Dataframe com as métricas
        """

        log.info("Start")
        models = [
            "decision_tree_regressor",
            "random_forest_regressor",
            "neural_network",
            "xgb_regressor",
        ]

        experiments = ["1_day", "7_days", "14_days", "28_days"]

        model_types = ["with_features", "wo_features"]

        dfs_metrica = []
        dfs_data = []
        for model in models:
            for experiment in experiments:
                for model_type in model_types:

                    dfs_metrica.append(
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

                    df_tmp_data = pd.DataFrame(
                        self.dict_results[model][model_type][experiment]["data"]
                    )
                    df_tmp_data["experiment"] = experiment
                    df_tmp_data["model"] = model
                    df_tmp_data["model_type"] = model_type
                    dfs_data.append(df_tmp_data.copy())

        return self.generate_metrics(dfs_metrica), self.generate_experiments(dfs_data)

    def generate_metrics(self, dfs_metrica: List[pd.DataFrame]) -> pd.DataFrame:
        df_metrica = pd.concat(dfs_metrica)

        df_metrica = df_metrica.reset_index()
        df_metrica.drop(columns=["index"], inplace=True)

        # Transorm RMSE into two columns, with_features and wo_features
        df_metrica = df_metrica.set_index(["experiment", "model", "model_type"])[
            "rmse"
        ].unstack()
        df_metrica = df_metrica.reset_index()
        df_metrica["best_model"] = np.where(
            df_metrica["with_features"] < df_metrica["wo_features"],
            "with_features",
            np.where(
                df_metrica["with_features"] < df_metrica["wo_features"],
                "wo_features",
                "EMPATE",
            ),
        )

        df_metrica = df_metrica.sort_values(["model"])

        return df_metrica

    def generate_experiments(self, dfs_data: List[pd.DataFrame]) -> pd.DataFrame:
        log.info("Start")
        df_data = pd.concat(dfs_data)
        df_data = df_data[
            [
                "experiment",
                "model",
                "model_type",
                "ticker",
                "date",
                "close",
                "predicted",
            ]
        ]
        df_data["predicted"] = round(df_data["predicted"], 2)
        return df_data

    @staticmethod
    def root_mean_squared_error(y_true, y_pred) -> float:
        return mean_squared_error(y_true, y_pred, squared=False)
