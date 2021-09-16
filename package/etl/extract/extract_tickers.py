import io
import json
import logging
import os
from time import sleep

import yaml
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="[%(process)-5d][%(asctime)s][%(filename)-10s][%(funcName)-25s][%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(filename="data/log/extract/extract_tickers.log"),
    ],
)


class ExtractTickers:
    def __init__(self):
        logging.info("Start")
        root_path = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )

        with open(os.path.join(root_path, "package", "config.yml"), "r") as ymlfile:
            self.cfg = yaml.safe_load(ymlfile)["extract"]["history"]

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

        with open(self.tikers_path) as json_file:
            tickers = json.load(json_file)["tickers"]
        for ticker in tickers:
            logging.info(f"---- {ticker} ----")
            df_history = yf.download(
                tickers=f"{ticker}.SA",
                start=str(self.cfg["start_date"]),
                end=str(self.cfg["end_date"]),
            )
            sleep(2)  # não da para configar muito nessa api
            df_history["ticker"] = ticker

            if df_history.shape[0] == 0:
                failed.append(ticker)
            else:
                df_history.to_parquet(
                    f"{os.path.join(self.output_path, ticker.lower())}.parquet"
                )

                # A validação de low data deve ser feita contando os dados de treino somente
                # fmt: off
                if len(
                    df_history[df_history.index <= str(self.cfg["validation_days_max_date"])]
                ) >= dias_uteis_em_um_ano:
                    sucessful.append(ticker)
                else:
                    low_data.append(ticker)
                # fmt: on

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
