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
        logging.FileHandler(filename="data/log/transform.log"),
    ],
)


class Transform:
    def __init__(self):
        root_path = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        )
        data_path = os.path.join(root_path, "data")
        self.history_path = os.path.join(data_path, "history")
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
        self.transform_ticker()
        # self.consolidado.to_parquet(self.output_consolidado_path)
        logging.info("Finished")
        # return self.consolidado

    def load_data(self):
        logging.info("Start")
        self.df_setor = pd.read_parquet(self.df_setor_path)
        self.df_fundamentalista = pd.read_parquet(self.df_fundamentalista_path)
        self.df_history = self.load_history()

    def load_history(self) -> pd.DataFrame:
        logging.info("Start")
        df_list = []
        for file in os.listdir(self.history_path):
            if ".parquet" in file:
                file_path = os.path.join(self.history_path, file)
                df_list.append(pd.read_parquet(file_path))
        return pd.concat(df_list)

    def transform_ticker(self, target: str = "close"):
        """Transforma os dados de historico das acoes

        Args:
            target (str, optional): Qual sera a coluna predita [close, adj_close]. Defaults to 'close'.
        """
        logging.info("Start")
        self.df_history = self.df_history.reset_index()
        self.df_history.columns = [
            "date",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
            "ticker",
        ]

        self.df_history["symbol"] = self.df_history["ticker"].str.slice(0, 4)
        self.df_history = self.df_history[[target, "date", "ticker", "symbol"]].copy()
        ticker_dummies = pd.get_dummies(
            self.df_history.ticker, prefix="ticker", prefix_sep="."
        )
        self.df_history = pd.concat([self.df_history, ticker_dummies], axis=1)

        # Movendo colunas para primeira posicao
        symbol = self.df_history.pop("symbol")
        ticker = self.df_history.pop("ticker")
        self.df_history.insert(0, "symbol", symbol)
        self.df_history.insert(0, "ticker", ticker)
        logging.info(f"self.df_history.shape: {self.df_history.shape}")


if __name__ == "__main__":
    Transform().transform()
    print("opa")
