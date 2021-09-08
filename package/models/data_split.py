import pandas as pd
import logging

log = logging.getLogger(__name__)

TRAIN_MAX_DATE = "2021-05-18"


class DataSplit:
    def __init__(self):
        self.X_train: pd.DataFrame = None
        self.y_train: pd.Series = None

        self.X_test_1_day: pd.Series = None
        self.y_test_1_day: pd.Series = None

        self.X_test_7_day: pd.Series = None
        self.y_test_7_day: pd.Series = None

        self.X_test_14_day: pd.Series = None
        self.y_test_14_day: pd.Series = None

        self.X_test_28_days: pd.Series = None
        self.y_test_28_days: pd.Series = None

    def train_test_split(self):
        log.info("Start")
        self.X_train = self.df[self.df["date"] <= TRAIN_MAX_DATE].iloc[:, 1:]
        self.y_train = self.df[self.df["date"] <= TRAIN_MAX_DATE].iloc[:, 0]

        self.X_test = self.df[self.df["date"] > TRAIN_MAX_DATE].iloc[:, 1:]
        self.y_test = self.df[self.df["date"] > TRAIN_MAX_DATE].iloc[:, 0]
        self.y_data = self.df[self.df["date"] > TRAIN_MAX_DATE]

        self.X_test_28_days = self.df[self.df["date"] > TRAIN_MAX_DATE].iloc[:, 1:]
        self.y_test_28_days = self.df[self.df["date"] > TRAIN_MAX_DATE].iloc[:, 0]
        self.y_data_28_days = self.df[self.df["date"] > TRAIN_MAX_DATE]
