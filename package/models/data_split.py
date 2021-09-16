import logging
import os
from typing import List, Tuple

import pandas as pd
import yaml

from model_type import ModelType

log = logging.getLogger(__name__)


class DataSplit:
    def __init__(self):
        # GS Data
        self.X_gs_train: pd.DataFrame = None
        self.y_gs_train: pd.Series = None

        self.test_gs_data_1_day: pd.DataFrame = None
        self.X_gs_test_1_day: pd.DataFrame = None
        self.y_gs_test_1_day: pd.Series = None

        self.test_gs_data_7_days: pd.DataFrame = None
        self.X_gs_test_7_days: pd.DataFrame = None
        self.y_gs_test_7_days: pd.Series = None

        self.test_gs_data_14_days: pd.DataFrame = None
        self.X_gs_test_14_days: pd.DataFrame = None
        self.y_gs_test_14_days: pd.Series = None

        self.test_gs_data_28_days: pd.DataFrame = None
        self.X_gs_test_28_days: pd.DataFrame = None
        self.y_gs_test_28_days: pd.Series = None

        # Predict Data
        self.X_train: pd.DataFrame = None
        self.y_train: pd.Series = None

        self.test_data_1_day: pd.DataFrame = None
        self.X_test_1_day: pd.DataFrame = None
        self.y_test_1_day: pd.Series = None

        self.test_data_7_days: pd.DataFrame = None
        self.X_test_7_days: pd.DataFrame = None
        self.y_test_7_days: pd.Series = None

        self.test_data_14_days: pd.DataFrame = None
        self.X_test_14_days: pd.DataFrame = None
        self.y_test_14_days: pd.Series = None

        self.test_data_28_days: pd.DataFrame = None
        self.X_test_28_days: pd.DataFrame = None
        self.y_test_28_days: pd.Series = None

        root_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        with open(os.path.join(root_path, "package", "config.yml"), "r") as ymlfile:
            cfg = yaml.safe_load(ymlfile)["model"]
        self.cfg_gs = cfg["grid_search"]
        self.cfg_predict = cfg["predict"]

    def train_test_split(self, model_type: ModelType):
        log.info("Start")
        if model_type == ModelType.WITH_FEATURES:
            # Columns: [ticker, symbol, close, date, year, month, ...]
            skip_indexes = 4
            close_column_index = 2
        elif model_type == ModelType.WITHOUT_FEATURES:
            # Columns: [ticker, close, date, year, month, day, ticker.AALR3, ...]
            skip_indexes = 3
            close_column_index = 0

        """O dataset do GridSearch precisa ser diferente 
            pois o objeto PredefinedSplit trabalhar com o index
            para definir o split
        """
        self.set_gs_data(skip_indexes, close_column_index)
        self.set_predict_data(skip_indexes, close_column_index)
        log.info("Finished")

    def set_gs_data(self, skip_indexes: int, close_column_index: int):
        log.info("Start")
        max_date = str(self.cfg_gs["train"]["max_date"])
        # Train set
        self.X_train_gs = self.df[self.df["date"] <= max_date].iloc[:, skip_indexes:]
        self.y_train_gs = self.df[self.df["date"] <= max_date].iloc[
            :, close_column_index
        ]

        # Test set
        (
            self.test_gs_data_1_day,
            self.X_gs_test_1_day,
            self.y_gs_test_1_day,
        ) = self.test_generator(1, skip_indexes, close_column_index, max_date)

        (
            self.test_gs_data_7_days,
            self.X_gs_test_7_days,
            self.y_gs_test_7_days,
        ) = self.test_generator(7, skip_indexes, close_column_index, max_date)

        (
            self.test_gs_data_14_days,
            self.X_gs_test_14_days,
            self.y_gs_test_14_days,
        ) = self.test_generator(14, skip_indexes, close_column_index, max_date)
        (
            self.test_gs_data_28_days,
            self.X_gs_test_28_days,
            self.y_gs_test_28_days,
        ) = self.test_generator(28, skip_indexes, close_column_index, max_date)

    def set_predict_data(self, skip_indexes: int, close_column_index: int):
        log.info("Start")
        max_date = str(self.cfg_predict["train"]["max_date"])
        # Train set
        self.X_train = self.df[self.df["date"] <= max_date].iloc[:, skip_indexes:]
        self.y_train = self.df[self.df["date"] <= max_date].iloc[:, close_column_index]

        # Test Sets
        (
            self.test_data_1_day,
            self.X_test_1_day,
            self.y_test_1_day,
        ) = self.test_generator(1, skip_indexes, close_column_index, max_date)

        (
            self.test_data_7_days,
            self.X_test_7_days,
            self.y_test_7_days,
        ) = self.test_generator(7, skip_indexes, close_column_index, max_date)

        (
            self.test_data_14_days,
            self.X_test_14_days,
            self.y_test_14_days,
        ) = self.test_generator(14, skip_indexes, close_column_index, max_date)
        (
            self.test_data_28_days,
            self.X_test_28_days,
            self.y_test_28_days,
        ) = self.test_generator(28, skip_indexes, close_column_index, max_date)

    def test_generator(
        self, days: int, skip_indexes: int, close_column_index: int, max_date: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Gerador de Dataframe com base na quantidade de dias.
        Os dias começam a contar DEPOIS (>) do TRAIN_MAX_DATE"""
        log.info(f"Generating test data with {days} days")
        test_data: List[pd.DataFrame] = []
        x_test: List[pd.DataFrame] = []
        y_test: List[pd.DataFrame] = []

        tickers = list(self.df["ticker"].unique())

        for ticker in tickers:
            test_data.append(
                self.df[
                    (self.df["ticker"] == ticker) & (self.df["date"] > max_date)
                ].head(days)[["ticker", "date", "close"]]
            )

            x_test.append(
                self.df[(self.df["ticker"] == ticker) & (self.df["date"] > max_date)]
                .head(days)
                .iloc[:, skip_indexes:]
            )

            y_test.append(
                self.df[(self.df["ticker"] == ticker) & (self.df["date"] > max_date)]
                .head(days)
                .iloc[:, close_column_index]
            )

        return pd.concat(test_data), pd.concat(x_test), pd.concat(y_test)
