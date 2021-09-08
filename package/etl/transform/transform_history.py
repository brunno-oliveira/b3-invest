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
        logging.FileHandler(filename="data/log/transform/transform_history.log"),
    ],
)


class TransformHistory:
    def __init__(self, target: str = "close"):
        self.target = target
        root_path = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )
        data_path = os.path.join(root_path, "data")
        ticker_low_data = "tickers_low_data.json"
        self.ticker_low_data = os.path.join(data_path, ticker_low_data)
        self.history_path = os.path.join(data_path, "history")
        self.df_history: pd.DataFrame = None
        self.output_consolidado_path = os.path.join(data_path, "df_history.parquet")

    def transform(self):
        logging.info("Start")
        self.df_history = self.load_history()
        logging.info(f"self.df_history.shape: {self.df_history.shape}")
        self.transform_ticker()
        self.remove_missing_data()
        self.transform_date()
        self.df_history.to_parquet(self.output_consolidado_path)

    def load_history(self) -> pd.DataFrame:
        logging.info("Start")
        with open(self.ticker_low_data) as json_file:
            low_data_tickers = json.load(json_file)["low_data_tickers"]

        df_list = []
        for file in os.listdir(self.history_path):
            # Skip low data
            if any([x.lower() in file for x in low_data_tickers]):
                logging.info(f"Skiping {file} due to low data")
            elif ".parquet" in file:
                file_path = os.path.join(self.history_path, file)
                df_list.append(pd.read_parquet(file_path))
        return pd.concat(df_list)

    def transform_ticker(self):
        """Transforma os dados de historico das acoes

        Args:
            target (str, optional): Qual sera a coluna predita [close, adj_close].
                Defaults to 'close'.
        """
        logging.info("Start")
        self.df_history = self.df_history.reset_index()
        self.df_history = self.df_history.drop("Volume", axis=1)
        self.df_history.columns = [
            "date",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "ticker",
        ]

        self.df_history["symbol"] = self.df_history["ticker"].str.slice(0, 4)
        self.df_history = self.df_history[
            [self.target, "date", "ticker", "symbol"]
        ].copy()
        ticker_dummies = pd.get_dummies(
            self.df_history.ticker, prefix="ticker", prefix_sep="."
        )
        self.df_history = pd.concat([self.df_history, ticker_dummies], axis=1)

        # Reorganizando as primeiras colunas
        symbol = self.df_history.pop("symbol")
        ticker = self.df_history.pop("ticker")
        self.df_history.insert(0, "symbol", symbol)
        self.df_history.insert(0, "ticker", ticker)
        logging.info(f"self.df_history.shape: {self.df_history.shape}")

    def remove_missing_data(self):
        """As vezes faltam alguns dados diarios"""
        total_missing_rows = len(self.df_history[self.df_history["close"].isna()])
        if total_missing_rows > 0:
            if total_missing_rows > 20:
                raise Exception("Missing too many data!")
            else:
                self.df_history = self.df_history.dropna(subset=["close"])
        logging.info(f"Dropper {total_missing_rows} missing data!")

    def transform_date(self):
        logging.info("Start")
        self.df_history['year'] = self.df_history['date'].dt.year
        self.df_history['month'] = self.df_history['date'].dt.month
        self.df_history['day'] = self.df_history['date'].dt.day
        self.df_history['dayofyear'] = self.df_history['date'].dt.dayofyear
        self.df_history['dayofweek'] = self.df_history['date'].dt.dayofweek
        self.df_history['weekofyear'] = self.df_history['date'].dt.weekofyear

        # Df columns: [ticker, symbol, close, date, ticker.AALR3, ...]
        # A reorganizacão é para manter as colunas de datas próximas, no começo do df
        date_column_index = 4 #Insert after date column
        date_columns = ['weekofyear', 'dayofweek', 'dayofyear', 'day', 'month', 'year']


        for col in date_columns:
            df_col = self.df_history.pop(col)
            self.df_history.insert(date_column_index, col, df_col)
    
TransformHistory().transform()
