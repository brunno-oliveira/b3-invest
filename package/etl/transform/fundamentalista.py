import os
import json
import logging
import pandas as pd
from typing import List

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
        self.output_consolidado_path = os.path.join(
            data_path, "df_fundamentalista.parquet"
        )
        self.tickers_path = os.path.join(data_path, "tickers.json")
        self.df_consolidado: pd.DataFrame = None

    @staticmethod
    def get_columns() -> List[str]:
        """Colunas filtradas a partir dos dados fundamentalisa bruto. """
        # fmt: off
        return [
            "symbol", "asOfDate", "NetIncomeFromContinuingOperations", "ReconciledDepreciation",
            "ChangeInCashSupplementalAsReported", "ChangeInWorkingCapital",
            "InvestingCashFlow", "BeginningCashPosition", "FinancingCashFlow",
            "EndCashPosition", "OperatingCashFlow", "LongTermDebtAndCapitalLeaseObligation",
            "ChangesInCash", "FreeCashFlow", "SellingGeneralAndAdministration",
            "TotalDebt", "TaxProvision", "NetPPE", "Payables", "NetInterestIncome",
            "CommonStock", "CapitalStock", "CashAndCashEquivalents", "InvestedCapital",
            "TotalCapitalization", "NetIncomeFromContinuingAndDiscontinuedOperation",
            "NetIncome", "NetIncomeCommonStockholders", "TaxRateForCalcs",
            "TaxEffectOfUnusualItems", "TotalRevenue", "NetIncomeContinuousOperations",
            "PretaxIncome", "OrdinarySharesNumber", "OperatingRevenue",
            "NetIncomeFromContinuingOperationNetMinorityInterest", 
            "NetIncomeIncludingNoncontrollingInterests",
            "NormalizedIncome", "DilutedNIAvailtoComStockholders", "ShareIssued",
            "NetTangibleAssets", "TotalEquityGrossMinorityInterest",
            "TotalAssets", "TangibleBookValue", "CommonStockEquity",
            "TotalLiabilitiesNetMinorityInterest", "StockholdersEquity"
        ]
        # fmt: on

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
        self.df_consolidado = pd.concat(dfs)
        del dfs
        self.df_consolidado = self.df_consolidado.reset_index()
        logging.info(f"self.consolidado.shape: {self.df_consolidado.shape}")

    def remove_duplicates(self):
        """Ex: AALR3.SA -> AALR"""
        self.df_consolidado["symbol"] = self.df_consolidado["symbol"].str[0:4]
        self.df_consolidado = self.df_consolidado.drop_duplicates()
        logging.info(f"self.consolidado.shape: {self.df_consolidado.shape}")

    def transform(self):
        logging.info("Start")
        self.load_tickers()
        self.load_data()
        self.remove_duplicates()
        self.df_consolidado.to_parquet(self.output_consolidado_path)
        logging.info(f"self.consolidado.shape: {self.df_consolidado.shape}")


TransformFundamentalista().transform()