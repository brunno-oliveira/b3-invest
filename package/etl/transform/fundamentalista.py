import os
import json
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[%(process)-5d][%(asctime)s][%(filename)-10s][%(funcName)-10s][%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(filename="data/log/transform_fundamentalista.log"),
    ],
)


class TransformFundamentalista:
    def __init__(self):
        logging.info("Start")
        root_path = root_path = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )
        data_path = os.path.join(root_path, "data")
        self.data_fundamentalista_path = os.path.join(data_path, "fundamentalista")
        self.output_consolidado_path = os.path.join(data_path, "df_consolidado.parquet")
        self.tickers_path = os.path.join(data_path, "tickers.json")
        self.consolidado: pd.DataFrame = None

    def load_tickers(self):
        logging.info("Start")
        with open(self.tickers_path, encoding="utf-8") as json_file:
            tickers = json.load(json_file)["tickers"]
        tickers = [ticker[0:4] for ticker in tickers]
        tickers = set(tickers)
        logging.info(f"{len(tickers)} unique tickers")

    def load_data(self):
        dfs = []
        for file in os.listdir(self.data_fundamentalista_path):
            if ".parquet" in file:
                file_path = os.path.join(self.data_fundamentalista_path, file)
                dfs.append(pd.read_parquet(file_path))
        self.consolidado = pd.concat(dfs)
        del dfs
        self.consolidado = self.consolidado.reset_index()
        self.consolidado.to_parquet(self.output_consolidado_path)
        logging.info(f"self.consolidado.shape: {self.consolidado.shape}")

    def transform(self):
        logging.info("Start")
        self.load_tickers()
        # self.load_data()


TransformFundamentalista().transform()