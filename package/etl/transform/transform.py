import os
import json
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[%(process)-5d][%(asctime)s][%(filename)-10s][%(funcName)-15s][%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(filename="data/log/transform/transform.log"),
    ],
)


class Transform:
    def __init__(self):
        root_path = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )
        data_path = os.path.join(root_path, "data")
        self.df_history_path = os.path.join(data_path, "df_history.parquet")
        self.df_setor_path = os.path.join(data_path, "df_setor.parquet")
        self.df_fundamentalista_path = os.path.join(
            data_path, "df_fundamentalista.parquet"
        )

        self.df_consolidado: pd.DataFrame = None
        self.df_history: pd.DataFrame = None
        self.df_setor: pd.DataFrame = None
        self.df_fundamentalista: pd.DataFrame = None

        self.output_consolidado_path = os.path.join(data_path, "df_consolidado.parquet")

    def transform(self):
        logging.info("Start")
        self.load_data()
        self.merge()
        self.df_consolidado.to_parquet(self.output_consolidado_path)
        logging.info("Finished")
        return self.df_consolidado

    def merge(self):
        logging.info("Start")
        # TODO: Validar os TICKERS que nao batem. Pelo visto sao acoes fora da B3
        self.df_consolidado = self.df_history.merge(
            self.df_setor, left_on="symbol", right_on="codigo", how="inner"
        )
        self.df_consolidado.drop(columns="codigo", inplace=True)
        logging.info(f"df_history merge df_setor: {self.df_consolidado.shape}")
        self.df_consolidado = self.df_consolidado.merge(
            self.df_fundamentalista, on="symbol", how="inner"
        )
        logging.info(
            f"df_consolidado merge df_fundamentalista: {self.df_consolidado.shape}"
        )
        del self.df_setor
        del self.df_history
        del self.df_fundamentalista

    def load_data(self):
        logging.info("Start")
        self.df_setor = pd.read_parquet(self.df_setor_path)
        self.df_fundamentalista = pd.read_parquet(self.df_fundamentalista_path)
        self.df_history = pd.read_parquet(self.df_history_path)
        logging.info(f"self.df_setor.shape: {self.df_setor.shape}")
        logging.info(f"self.df_fundamentalista.shape: {self.df_fundamentalista.shape}")
        logging.info(f"self.df_history.shape: {self.df_history.shape}")


if __name__ == "__main__":
    Transform().transform()
