import logging
from typing import List, Tuple

import pandas as pd

from model_type import ModelType

log = logging.getLogger(__name__)

TRAIN_MAX_DATE = "2021-05-18"


class DataSplit:
    def __init__(self):
        self.X_train_gs: pd.DataFrame = None
        self.y_train_gs: pd.Series = None

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
        self.X_train_gs = self.df.iloc[:, skip_indexes:]
        self.y_train_gs = self.df.iloc[:, close_column_index]

        # Train set
        self.X_train = self.df[self.df["date"] <= TRAIN_MAX_DATE].iloc[:, skip_indexes:]
        self.y_train = self.df[self.df["date"] <= TRAIN_MAX_DATE].iloc[
            :, close_column_index
        ]

        # Test Sets
        (
            self.test_data_1_day,
            self.X_test_1_day,
            self.y_test_1_day,
        ) = self.test_generator(1, skip_indexes, close_column_index)

        (
            self.test_data_7_days,
            self.X_test_7_days,
            self.y_test_7_days,
        ) = self.test_generator(7, skip_indexes, close_column_index)

        (
            self.test_data_14_days,
            self.X_test_14_days,
            self.y_test_14_days,
        ) = self.test_generator(14, skip_indexes, close_column_index)
        (
            self.test_data_28_days,
            self.X_test_28_days,
            self.y_test_28_days,
        ) = self.test_generator(28, skip_indexes, close_column_index)
        log.info("Finished")

    def test_generator(
        self, days: int, skip_indexes: int, close_column_index: int
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
                    (self.df["ticker"] == ticker) & (self.df["date"] > TRAIN_MAX_DATE)
                ].head(days)[["ticker", "date", "close"]]
            )

            x_test.append(
                self.df[
                    (self.df["ticker"] == ticker) & (self.df["date"] > TRAIN_MAX_DATE)
                ]
                .head(days)
                .iloc[:, skip_indexes:]
            )

            y_test.append(
                self.df[
                    (self.df["ticker"] == ticker) & (self.df["date"] > TRAIN_MAX_DATE)
                ]
                .head(days)
                .iloc[:, close_column_index]
            )

        return pd.concat(test_data), pd.concat(x_test), pd.concat(y_test)
