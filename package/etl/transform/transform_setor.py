import os
import json
import logging
import pandas as pd
from unidecode import unidecode

logging.basicConfig(
    level=logging.INFO,
    format="[%(process)-5d][%(asctime)s][%(filename)-10s][%(funcName)-25s][%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(filename="data/log/transform/transform_setor.log"),
    ],
)


class Transform:
    def __init__(self):
        root_path = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )
        data_path = os.path.join(root_path, "data")
        self.ticker_setorial_path = os.path.join(data_path, "b3_setorial.csv")
        self.df_setorial: pd.DataFrame = None
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
        self.df_setorial = pd.read_csv(
            self.ticker_setorial_path, sep=";", encoding="latin"
        )

    def transform_setor(self):
        """
        Criando as colunas dummies para SETOR
        """
        logging.info("Start")
        self.df_setorial = self.transform_column(self.df_setorial, "setor")

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
        self.df_setorial = self.transform_column(self.df_setorial, "subsetor")
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
        self.df_setorial = self.transform_column(self.df_setorial, "segmento")
        setor_dummies = pd.get_dummies(
            self.df_setorial.segmento, prefix="segmento", prefix_sep="."
        ).drop(columns="segmento.outros")
        self.df_setorial = pd.concat([self.df_setorial, setor_dummies], axis=1).drop(
            columns="segmento"
        )

        self.df_setorial.drop(columns="listagem_segmento", inplace=True)

    @staticmethod
    def transform_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Padroniza os valores das colunas com:
        unidecode, lower, replace , e ' '
        """

        def replace_method(x: str) -> str:
            x = unidecode(x).lower()
            x = x.replace(",", "")
            x = x.replace(" ", "_")
            return x

        df[col] = df[col].apply(lambda x: replace_method(x))
        return df


Transform().transform()
