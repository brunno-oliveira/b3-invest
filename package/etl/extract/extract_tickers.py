import io
import json
import logging
import os
from datetime import datetime

import yfinance as yf
from dateutil.relativedelta import relativedelta

logging.basicConfig(
    level=logging.INFO,
    format="[%(process)-5d][%(asctime)s][%(filename)-10s][%(funcName)-25s][%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(filename="data/log/extract_tickers.log"),
    ],
)


class ExtractTickers:
    def __init__(self):
        logging.info("Start")
        root_path = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )
        data_path = os.path.join(root_path, "data")

        tickers_file_name = "tickers.json"
        ticker_low_data = "tickers_low_data.json"
        ticker_failed_data = "tickers_failed_data.json"
        self.tikers_path = os.path.join(data_path, tickers_file_name)
        self.ticker_low_data = os.path.join(data_path, ticker_low_data)
        self.ticker_failed_data = os.path.join(data_path, ticker_failed_data)
        self.output_path = os.path.join(data_path, "history")

    def clear_data(self):
        logging.info("Start")
        for file in os.listdir(self.output_path):
            if ".parquet" in file:
                os.remove(os.path.join(self.output_path, file))

        if os.path.exists(self.ticker_low_data):
            os.remove(self.ticker_low_data)

    def download(self):
        logging.info("Start")
        dias_uteis_em_um_ano = 254
        sucessful = []
        failed = []
        low_data = []

        end_date = datetime(2021, 5, 18).date()
        start_date = end_date - relativedelta(years=2)

        with open(self.tikers_path) as json_file:
            tickers = json.load(json_file)["tickers"]
        for ticker in tickers:
            logging.info(f"---- {ticker} ----")

            df_history = yf.download(
                tickers=f"{ticker}.SA", start=str(start_date), end=str(end_date)
            )
            df_history["ticker"] = ticker

            if df_history.shape[0] == 0:
                failed.append(ticker)
            else:
                df_history.to_parquet(
                    f"{os.path.join(self.output_path, ticker.lower())}.parquet"
                )
                if df_history.shape[0] >= dias_uteis_em_um_ano:
                    sucessful.append(ticker)
                else:
                    low_data.append(ticker)

        logging.info("----------------------")
        logging.warning(f"{len(failed)} REMOVED DUE TO FAILED: {failed}")
        logging.warning(f"{len(low_data)} REMOVED DUE TO LOW DATA: {low_data}")

        # Save low_data tickers
        low_data_json = json.dumps({"low_data_tickers": low_data})
        with io.open(self.ticker_low_data, "w") as f:
            f.write(low_data_json)

        # Save low_failed tickers
        failed_data_json = json.dumps({"failed_data_tickers": failed})
        with io.open(self.ticker_failed_data, "w") as f:
            f.write(failed_data_json)


if __name__ == "__main__":
    extract = ExtractTickers()
    extract.clear_data()
    extract.download()
