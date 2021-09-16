import logging
import os
from datetime import date
from typing import List, Tuple

import pandas as pd
import yaml

from model_type import ModelType

log = logging.getLogger(__name__)


class DataSplit:
    def __init__(self):
        # GS Data
        self.gs_data: pd.DataFrame = None
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
        self.train_data: pd.DataFrame = None
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

        max_train_date = str(self.cfg_gs["train"]["max_date"])
        # Full GS Set
        self.gs_data = self.df[self.df["date"] <= max_train_date].copy()
        self.gs_data = self.gs_data.reset_index()
        self.gs_data.drop(columns=["index"], inplace=True)

        # Train set
        self.X_gs_train = self.gs_data.iloc[:, skip_indexes:]
        self.y_gs_train = self.gs_data.iloc[:, close_column_index]

        # Test set
        cfg_exp = self.cfg_gs["experiments"]
        cfg_exp_1_day = cfg_exp["1_day"]
        cfg_exp_7_days = cfg_exp["7_days"]
        cfg_exp_14_days = cfg_exp["14_days"]
        cfg_exp_28_days = cfg_exp["28_days"]
        (
            self.test_gs_data_1_day,
            self.X_gs_test_1_day,
            self.y_gs_test_1_day,
        ) = self.test_generator(
            self.gs_data,
            1,
            skip_indexes,
            close_column_index,
            cfg_exp_1_day["start_date"],
            cfg_exp_1_day["end_date"],
        )

        (
            self.test_gs_data_7_days,
            self.X_gs_test_7_days,
            self.y_gs_test_7_days,
        ) = self.test_generator(
            self.gs_data,
            7,
            skip_indexes,
            close_column_index,
            cfg_exp_7_days["start_date"],
            cfg_exp_7_days["end_date"],
        )

        (
            self.test_gs_data_14_days,
            self.X_gs_test_14_days,
            self.y_gs_test_14_days,
        ) = self.test_generator(
            self.gs_data,
            14,
            skip_indexes,
            close_column_index,
            cfg_exp_14_days["start_date"],
            cfg_exp_14_days["end_date"],
        )
        (
            self.test_gs_data_28_days,
            self.X_gs_test_28_days,
            self.y_gs_test_28_days,
        ) = self.test_generator(
            self.gs_data,
            28,
            skip_indexes,
            close_column_index,
            cfg_exp_28_days["start_date"],
            cfg_exp_28_days["end_date"],
        )

    def set_predict_data(self, skip_indexes: int, close_column_index: int):
        log.info("Start")
        max_train_date = str(self.cfg_predict["train"]["max_date"])

        # Full Train set
        self.train_data = self.df[self.df["date"] <= max_train_date]
        self.train_data = self.train_data.reset_index()
        self.train_data.drop(columns=["index"], inplace=True)

        # Train set
        self.X_train = self.train_data.iloc[:, skip_indexes:]
        self.y_train = self.train_data.iloc[:, close_column_index]

        # Test set
        cfg_exp = self.cfg_predict["experiments"]
        cfg_exp_1_day = cfg_exp["1_day"]
        cfg_exp_7_days = cfg_exp["7_days"]
        cfg_exp_14_days = cfg_exp["14_days"]
        cfg_exp_28_days = cfg_exp["28_days"]
        (
            self.test_data_1_day,
            self.X_test_1_day,
            self.y_test_1_day,
        ) = self.test_generator(
            self.df,
            1,
            skip_indexes,
            close_column_index,
            cfg_exp_1_day["start_date"],
            cfg_exp_1_day["end_date"],
        )

        (
            self.test_data_7_days,
            self.X_test_7_days,
            self.y_test_7_days,
        ) = self.test_generator(
            self.df,
            7,
            skip_indexes,
            close_column_index,
            cfg_exp_7_days["start_date"],
            cfg_exp_7_days["end_date"],
        )

        (
            self.test_data_14_days,
            self.X_test_14_days,
            self.y_test_14_days,
        ) = self.test_generator(
            self.df,
            14,
            skip_indexes,
            close_column_index,
            cfg_exp_14_days["start_date"],
            cfg_exp_14_days["end_date"],
        )
        (
            self.test_data_28_days,
            self.X_test_28_days,
            self.y_test_28_days,
        ) = self.test_generator(
            self.df,
            28,
            skip_indexes,
            close_column_index,
            cfg_exp_28_days["start_date"],
            cfg_exp_28_days["end_date"],
        )

    @staticmethod
    def test_generator(
        df: pd.DataFrame,
        days: int,
        skip_indexes: int,
        close_column_index: int,
        start_date: date,
        end_date: date,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Gerador de Dataframe com base na quantidade de dias.
        Os dias começam a contar DEPOIS (>) do TRAIN_MAX_DATE"""
        log.info(f"Generating test data with {days} days")
        start_date = str(start_date)
        end_date = str(end_date)
        test_data: List[pd.DataFrame] = []
        x_test: List[pd.DataFrame] = []
        y_test: List[pd.DataFrame] = []

        tickers = list(df["ticker"].unique())

        for ticker in tickers:
            test_data.append(
                df[
                    (df["ticker"] == ticker)
                    & (df["date"] >= start_date)
                    & (df["date"] <= end_date)
                ][["ticker", "date", "close"]]
            )

            x_test.append(
                df[
                    (df["ticker"] == ticker)
                    & (df["date"] >= start_date)
                    & (df["date"] <= end_date)
                ].iloc[:, skip_indexes:]
            )

            y_test.append(
                df[
                    (df["ticker"] == ticker)
                    & (df["date"] >= start_date)
                    & (df["date"] <= end_date)
                ].iloc[:, close_column_index]
            )

        return pd.concat(test_data), pd.concat(x_test), pd.concat(y_test)
