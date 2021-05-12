import os
import json
import logging
import yfinance as yf


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
        self.tikers_path = os.path.join(data_path, tickers_file_name)
        self.output_path = os.path.join(data_path, "history")
        self.output_consolidado_path = os.path.join(data_path, "df_consolidado.parquet")

    def clear_data(self):
        logging.info("Start")
        for file in os.listdir(self.output_path):
            if ".parquet" in file:
                os.remove(os.path.join(self.output_path, file))

        if os.path.exists(self.output_consolidado_path):
            os.remove(self.output_consolidado_path)

    def download(self):
        logging.info("Start")
        sucessful = []
        failed = []
        low_data = []
        with open(self.tikers_path) as json_file:
            tickers = json.load(json_file)["tickers"]
        for ticker in tickers:
            logging.info(f"---- {ticker} ----")

            df_history = yf.download(tickers=f"{ticker}.SA", period="2y")
            df_history["ticker"] = ticker

            if df_history.shape[0] == 0:
                failed.append(ticker)
            elif df_history.shape[0] >= 360:
                df_history.to_parquet(
                    f"{os.path.join(self.output_path, ticker.lower())}.parquet"
                )
                sucessful.append(ticker)
            else:
                low_data.append(ticker)

        logging.info("----------------------")
        logging.warning("FAILED")
        logging.warning(failed)
        logging.warning("LOW DATA")
        logging.warning(low_data)


if __name__ == "__main__":
    extract = ExtractTickers()
    extract.clear_data()
    extract.download()
