import os
import json
import logging
import yfinance as yf


logging.basicConfig(
    level=logging.INFO,
    format="[%(process)-5d][%(asctime)s][%(filename)-20s][%(levelname)-8s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class ExtractTickers:
    def __init__(self):
        logging.info("Init")
        root_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        data_path = os.path.join(root_path, "data")

        tickers_file_name = "tickers.json"
        self.tikers_path = os.path.join(data_path, tickers_file_name)
        self.output_path = os.path.join(data_path, "history")

    def download(self):
        with open(self.tikers_path) as json_file:
            tickers = json.load(json_file)["tickers"]
        for ticker in tickers:
            logging.info(f"---- {ticker} ----")

            df_history = yf.download(tickers=f"{ticker}.SA", period="2y")
            df_history["ticker"] = ticker
            logging.info(df_history.shape)

            if df_history.shape[0] >= 360:
                df_history.to_parquet(
                    f"{os.path.join(self.output_path, ticker.lower())}.parquet"
                )
            else:
                logging.warning("Skipped because: Data <- 360 days")


if __name__ == "__main__":
    ExtractTickers().download()
