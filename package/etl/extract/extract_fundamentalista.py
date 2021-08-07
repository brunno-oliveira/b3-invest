import io
import os
import json
import logging
import pandas as pd
from tqdm import tqdm
from typing import List
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
        tickers_not_found = "tickers_financial_not_found.json"
        self.tikers_path = os.path.join(data_path, tickers_file_name)
        self.tikers_financial_not_found = os.path.join(data_path, tickers_not_found)
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

        if os.path.exists(self.tikers_financial_not_found):
            os.remove(self.tikers_financial_not_found)

    def get_finantial_data(self, tickers: List[str]):
        not_found = []
        for ticker in tqdm(tickers):
            df = YTicker(f"{ticker}.SA").all_financial_data(frequency="q")
            if type(df) != pd.DataFrame and df is not None:
                not_found.append(ticker)
            else:
                df.to_parquet(
                    f"{os.path.join(self.output_path, ticker.lower())}.parquet"
                )

        # Save not_found tickers
        logging.error(f"NO DATA FOUND: {not_found}")
        low_data_json = json.dumps({"no_financial_data": not_found})
        with io.open(self.tikers_financial_not_found, "w") as f:
            f.write(low_data_json)

    def run(self):
        logging.info("Start")
        with open(self.tikers_path) as json_file:
            tickers = json.load(json_file)["tickers"]
        self.get_finantial_data(tickers)
        logging.info("Done")


ExtractFundamentalista().run()