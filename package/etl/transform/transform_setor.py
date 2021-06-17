import os
import json
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[%(process)-5d][%(asctime)s][%(filename)-10s][%(funcName)-25s][%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(filename="data/log/transform_setor.log"),
    ],
)


class Transform:
    def __init__(self):
        root_path = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )
        data_path = os.path.join(root_path, "data")
        self.ticker_setorial_path = os.path.join(data_path, "b3_setorial.csv")
        self.setor_path = os.path.join(data_path, "setor.json")
        self.subsetor_path = os.path.join(data_path, "subsetor.json")
        self.segmento_path = os.path.join(data_path, "segmento.json")

        self.df_setorial: pd.DataFrame = None
        self.setor_map: pd.DataFrame = None
        self.subsetor_map: pd.DataFrame = None
        self.segmento_map: pd.DataFrame = None

        self.output_consolidado_path = os.path.join(data_path, "df_setor.parquet")

    def transform(self):
        logging.info("Start")
        self.load_data()
        self.transform_setor()
        self.transform_subsetor()
        self.transform_segmento()
        self.df_setorial.to_parquet(self.output_consolidado_path)
        logging.info("Finished")
        return self.df_setorial

    def load_data(self):
        logging.info("Start")
        with open(self.setor_path, encoding="utf-8") as json_file:
            self.setor_map = json.load(json_file)["setor_map"]

        with open(self.subsetor_path, encoding="utf-8") as json_file:
            self.subsetor_map = json.load(json_file)["subsetor_map"]

        with open(self.segmento_path, encoding="utf-8") as json_file:
            self.subsetor_map = json.load(json_file)["segmento_map"]

        self.df_setorial = pd.read_csv(
            self.ticker_setorial_path, sep=";", encoding="latin"
        )

    def transform_setor(self):
        """
        Criando as colunas dummies para SETOR
        """
        logging.info("Start")
        self.df_setorial["setor"] = self.df_setorial["setor"].map(self.setor_map)

        setor_dummies = pd.get_dummies(
            self.df_setorial.setor, prefix="setor", prefix_sep="."
        ).drop(columns="setor.outros")

        self.df_setorial = pd.concat([self.df_setorial, setor_dummies], axis=1).drop(
            columns="setor"
        )

    def transform_subsetor(self):
        """
        Criando as colunas dummies para SUBSETOR
        """
        logging.info("Start")
        self.df_setorial["subsetor"] = self.df_setorial["subsetor"].map(
            self.subsetor_map
        )
        setor_dummies = pd.get_dummies(
            self.df_setorial.subsetor, prefix="subsetor", prefix_sep="."
        ).drop(columns="subsetor.outros")
        self.df_setorial = pd.concat([self.df_setorial, setor_dummies], axis=1).drop(
            columns="subsetor"
        )

    def transform_segmento(self):
        """
        Criando as colunas dummies para SEGMENTO
        """
        logging.info("Start")
        self.df_setorial["segmento"] = self.df_setorial["segmento"].map(self.setor_map)
        setor_dummies = pd.get_dummies(
            self.df_setorial.segmento, prefix="segmento", prefix_sep="."
        ).drop(columns="segmento.outros")
        self.df_setorial = pd.concat([self.df_setorial, setor_dummies], axis=1).drop(
            columns="segmento"
        )

        self.df_setorial.drop(columns="listagem_segmento", inplace=True)


Transform().transform()
