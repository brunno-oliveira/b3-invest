import os
import json
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[%(process)-5d][%(asctime)s][%(filename)-20s][%(levelname)-8s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class Transform:
    def __init__(self):
        root_path = root_path = os.path.dirname(
            os.path.dirname(os.path.dirname(__file__))
        )
        data_path = os.path.join(root_path, "data")
        self.ticker_setorial_path = os.path.join(data_path, "b3_setorial.csv")
        self.setor_path = os.path.join(data_path, "setor.json")

        self.setor_map = None
        self.df_setorial = None

    def load_data(self):
        with open(self.setor_path, encoding="utf-8") as json_file:
            self.setor_map = json.load(json_file)["setor_map"]

        self.df_setorial = pd.read_csv(
            self.ticker_setorial_path, sep=";", encoding="latin"
        )

    def transform(self):
        self.load_data()
        self.df_setorial["setor"] = self.df_setorial["setor"].map(self.setor_map)
        setor_dummies = pd.get_dummies(
            self.df_setorial.setor, prefix="setor", prefix_sep="."
        ).drop(columns="setor.outros")

        self.df_setorial = pd.concat([self.df_setorial, setor_dummies], axis=1).drop(
            columns="setor"
        )

        print(self.df_setorial.head())


if __name__ == "__main__":
    Transform().transform()