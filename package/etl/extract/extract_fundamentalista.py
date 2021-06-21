import os
import json
import logging
import pandas as pd
from tqdm import tqdm
from yahooquery import Ticker as YTicker

logging.basicConfig(
    level=logging.INFO,
    format="[%(process)-5d][%(asctime)s][%(filename)-10s][%(funcName)-25s][%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(filename="data/log/extract_fundamentalista.log"),
    ],
)


class ExtractFundamentalista:
    def __init__(self):
        logging.info("Start")
        root_path = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )
        data_path = os.path.join(root_path, "data")
        tickers_file_name = "tickers.json"

        self.tikers_path = os.path.join(data_path, tickers_file_name)
        self.output_path = os.path.join(data_path, "fundamentalista")
        self.output_consolidado_path = os.path.join(
            data_path, "df_fundamentalista.parquet"
        )

    def clear_data(self):
        logging.info("Start")
        for file in os.listdir(self.output_path):
            if ".parquet" in file:
                os.remove(os.path.join(self.output_path, file))

        if os.path.exists(self.output_consolidado_path):
            os.remove(self.output_consolidado_path)

    def get_finantial_data(self, ticker: str):
        # logging.info(f"---- {ticker} ----")
        df = YTicker(ticker).all_financial_data(frequency="q")
        if type(df) != pd.DataFrame and df is not None:
            logging.error(f"Ticker {ticker} not found!")
            return None

        df.to_parquet(f"{os.path.join(self.output_path, ticker.lower())}.parquet")

    def run(self):
        with open(self.tikers_path) as json_file:
            tickers = json.load(json_file)["tickers"]

        for ticker in tqdm(tickers):
            self.get_finantial_data(f"{ticker}.SA")


ExtractFundamentalista().run()