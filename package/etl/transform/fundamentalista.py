import os
import logging
import pandas as pd
from typing import List
from sklearn.preprocessing import MinMaxScaler

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

    def load_data(self):
        logging.info("Start")
        dfs = []
        for file in os.listdir(self.data_fundamentalista_path):
            if ".parquet" in file:
                file_path = os.path.join(self.data_fundamentalista_path, file)
                dfs.append(pd.read_parquet(file_path))
        self.df_consolidado = pd.concat(dfs)
        del dfs
        self.df_consolidado = self.df_consolidado.reset_index()
        self.df_consolidado = self.df_consolidado[self.get_columns()].copy()
        logging.info(f"self.consolidado.shape: {self.df_consolidado.shape}")

    def remove_duplicates(self):
        """Ex: AALR3.SA -> AALR"""
        logging.info("Start")
        self.df_consolidado["symbol"] = self.df_consolidado["symbol"].str[0:4]
        self.df_consolidado = self.df_consolidado.drop_duplicates()
        logging.info(f"self.consolidado.shape: {self.df_consolidado.shape}")

    def scaler(self):
        logging.info("Start")
        scaler = MinMaxScaler()
        min_max = scaler.fit_transform(self.df_consolidado.iloc[:, 2:])
        df_scaler = pd.DataFrame(columns=self.get_columns()[2:], data=min_max)
        df_scaler["symbol"] = self.df_consolidado["symbol"]
        df_scaler["asOfDate"] = self.df_consolidado["asOfDate"]
        cols = df_scaler.columns.tolist()
        cols = cols[-2:] + cols[:-2]
        self.df_consolidado = df_scaler[cols]
        del df_scaler
        logging.info(f"self.consolidado.shape: {self.df_consolidado.shape}")

    def transform(self):
        logging.info("Start")
        self.load_data()
        self.remove_duplicates()
        self.scaler()
        self.df_consolidado.to_parquet(self.output_consolidado_path)


TransformFundamentalista().transform()