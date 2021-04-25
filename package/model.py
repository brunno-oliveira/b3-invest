import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

logging.basicConfig(
    level=logging.INFO,
    format="[%(process)-5d][%(asctime)s][%(filename)-20s][%(levelname)-8s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class Model:
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

    def load_data(self, df: pd.DataFrame = None):
        if df is None:
            self.df = pd.read_parquet(os.path.join(self.data_path, 'df_consolidado.parquet'))
            self.df = self.df.iloc[:, :-1].copy()  # Remove ticker column
        else:
            self.df = df

    def set_model(self):
        logging.info("Start")
        self.model = DecisionTreeRegressor(random_state=42)
        return self.model

    def 

    def fit_and_predict(self):
        logging.info("Start")
        self.fit()
        self.predict()
        return self.model, self.predicted

    def fit(self):
        logging.info("Start")
        self.model.fit(self.X_train, self.y_train)
        return self.model

    def predict(self):
        logging.info("Start")
        self.predicted = self.model.predict(self.X_test)
        return self.predicted

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


if __name__ == "__main__":
    root_path = root_path = os.path.dirname(os.path.dirname(__file__))
    model = Model()
    model.set_model()
    model.fit_and_predict()
    model.plot_metrics()
