import os
import json
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[%(process)-5d][%(asctime)s][%(filename)-10s][%(funcName)-25s][%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class Transform:
    def __init__(self):
        root_path = root_path = os.path.dirname(
            os.path.dirname(os.path.dirname(__file__))
        )
        data_path = os.path.join(root_path, "data")
        self.history_path = os.path.join(data_path, "history")
        self.ticker_setorial_path = os.path.join(data_path, "b3_setorial.csv")
        self.setor_path = os.path.join(data_path, "setor.json")
        self.subsetor_path = os.path.join(data_path, "subsetor.json")
        self.segmento_path = os.path.join(data_path, "segmento.json")

        self.consolidado: pd.DataFrame = None
        self.df_setorial: pd.DataFrame = None
        self.setor_map: pd.DataFrame = None
        self.subsetor_map: pd.DataFrame = None
        self.segmento_map: pd.DataFrame = None

        self.output_consolidado_path = os.path.join(data_path, "df_consolidado.parquet")

    def transform(self):
        logging.info("Start")
        self.load_data()
        self.transform_setor()
        self.transform_subsetor()
        self.transform_segmento()
        self.join_files()
        self.merge_setorial_consolidado()
        self.transform_ticker()
        self.consolidado.to_parquet(self.output_consolidado_path)
        logging.info("Finished")
        return self.consolidado

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

    def join_files(self, target: str = "close") -> pd.DataFrame:
        logging.info("Start")
        df_list = []
        for file in os.listdir(self.history_path):
            if ".parquet" in file:
                file_path = os.path.join(self.history_path, file)
                df_list.append(pd.read_parquet(file_path))
        self.consolidado = pd.concat(df_list)
        self.consolidado = self.consolidado.reset_index()
        self.consolidado.columns = [
            "date",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
            "ticker",
        ]

        self.consolidado["codigo"] = self.consolidado["ticker"].str.slice(0, 4)
        self.consolidado = self.consolidado[[target, "date", "ticker", "codigo"]].copy()

        logging.info(f"self.consolidado.shape: {self.consolidado.shape}")

    def merge_setorial_consolidado(self):
        logging.info("Start")
        self.consolidado = self.consolidado.merge(
            self.df_setorial, how="inner", left_on="codigo", right_on="codigo"
        )

        self.consolidado = self.consolidado.sort_values(by=["date"], ascending=True)
        self.consolidado["date"] = self.consolidado["date"].astype(str)
        self.consolidado["date"] = self.consolidado["date"].str.replace("-", "")
        self.consolidado["date"] = self.consolidado["date"].astype(int)

        self.consolidado.reset_index(drop=True, inplace=True)

        logging.info(f"self.consolidado.shape: {self.consolidado.shape}")

    def transform_ticker(self):
        logging.info("Start")
        ticker_dummies = pd.get_dummies(
            self.consolidado.ticker, prefix="ticker", prefix_sep="."
        )
        self.consolidado = pd.concat([self.consolidado, ticker_dummies], axis=1)

        # Movendo a coluna codigo para ultima posicao
        self.consolidado = self.consolidado.drop(columns=["codigo"])
        df_ticker = self.consolidado.pop("ticker")
        self.consolidado["ticker"] = df_ticker
        self.consolidado = self.consolidado.dropna()


if __name__ == "__main__":
    df = Transform().transform()
    print(df.shape)