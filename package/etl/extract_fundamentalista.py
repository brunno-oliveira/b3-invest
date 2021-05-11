import os
import json
import logging
import pandas as pd
from tqdm import tqdm
from yahooquery import Ticker as YTicker

logging.basicConfig(
    level=logging.INFO,
    format="[%(process)-5d][%(asctime)s][%(filename)-20s][%(levelname)-8s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(), logging.FileHandler(filename="extract.log")],
)


class ExtractFundamentalista:
    def __init__(self):
        logging.info("Start")
        root_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        data_path = os.path.join(root_path, "data")
        tickers_file_name = "tickers.json"

        self.tikers_path = os.path.join(data_path, tickers_file_name)
        self.output_consolidado_path = os.path.join(
            data_path, "fundamentalista", "df_fundamentalista.parquet"
        )

    def clear_data(self):
        logging.info("Start")
        for file in os.listdir(self.output_path):
            if ".parquet" in file:
                os.remove(os.path.join(self.output_consolidado_path, file))

    def get_finantial_data(self, ticker: str) -> pd.DataFrame:
        # logging.info(f"---- {ticker} ----")
        df = YTicker(ticker).all_financial_data(frequency="q")
        if type(df) != pd.DataFrame and df is not None:
            logging.error(f"Ticker {ticker} not found!")
            return None
        return df

    def run(self):
        with open(self.tikers_path) as json_file:
            tickers = json.load(json_file)["tickers"]

        df_consolidado = []
        for ticker in tqdm(tickers):
            df_consolidado.append(self.get_finantial_data(f"{ticker}.SA"))

        df_consolidado = pd.concat(df_consolidado)
        df_consolidado.to_parquet(self.output_consolidado_path)


ExtractFundamentalista().run()