from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import datetime as dt
import pandas as pd
import numpy as np
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="[%(process)-5d][%(asctime)s][%(filename)-10s][%(funcName)-20s][%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(), logging.FileHandler(filename="extract.log")],
)


class ModelBase:
    def __init__(self):
        root_path = root_path = os.path.dirname(
            os.path.dirname(os.path.dirname(__file__))
        )
        self.data_path = os.path.join(root_path, "data")
        self.model = None
        self.df: pd.DataFrame = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.predicted = None
        self.tickers = None

    def load_data(self, df: pd.DataFrame = None):
        if df is None:
            self.df = pd.read_parquet(
                os.path.join(self.data_path, "df_consolidado.parquet")
            )

            # Coluna para facilitar a busca por tickers
            self.tickers = self.df.iloc[:, 0]
            self.df = self.df.iloc[:, 2:].copy()  # Remove ticker columns
        else:
            self.df = df
        max_date = self.df["date"].max()
        # Previsao para o ultimo dia valido, removendo a primeira coluna (TARGET)
        self.X_train = self.df[self.df["date"] < max_date].iloc[:, 1:]
        self.y_train = self.df[self.df["date"] < max_date].iloc[:, 0]

        self.X_test = self.df[self.df["date"] == max_date].iloc[:, 1:]
        self.y_test = self.df[self.df["date"] == max_date].iloc[:, 0]

        self.transform_date()

    def set_model(self):
        pass

    def fit_and_predict(self):
        logging.info("Start")
        self.fit()
        self.predict()
        return self.model, self.predicted

    def fit(self):
        pass

    def predict(self):
        pass

    def transform_date(self):
        """O modelo nao trabalha com o tipo datetime"""
        self.X_train["date"] = self.X_train["date"].map(dt.datetime.toordinal)
        self.X_test["date"] = self.X_test["date"].map(dt.datetime.toordinal)

    def plot_metrics(self):
        logging.info(f"r2_score : {round(r2_score(self.predicted, self.y_test),4)}")
        logging.info(
            f"mean_squared_error: {round(mean_squared_error( self.predicted, self.y_test, squared=False),4)}"
        )
        logging.info(
            f"mean_absolute_error: {round(mean_absolute_error( self.predicted, self.y_test),4)}"
        )

        fig, ax = plt.subplots(figsize=(25, 8))
        ax.plot(self.predicted)
        ax.plot(np.array(self.y_test))
        plt.show()
