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

    def transform(self):
        logging.info("Start")
        self.load_data()
        self.remove_duplicates()
        self.scaler()
        self.prep_rows()
        self.fill_nan()
        self.fill_no_data()
        self.pivot()
        self.df_consolidado.to_parquet(self.output_consolidado_path)

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

    def prep_rows(self):
        """
        A quantidade de dimensoes por feature/coluna sera sempre sempre 4,
        oq eh referente a 4 trimestre.
        Se a quantidade de registro for maior que 4, ira usar as ultimas 4 linhas
        Se for menor, ira repetir a ultima linha (mais atualizada) ate dar 4
        """
        logging.info("Start")
        self.df_consolidado = self.df_consolidado[self.df_consolidado["symbol"].notna()]
        self.df_consolidado = self.df_consolidado.sort_values(by=["symbol", "asOfDate"])

        df_fixed_rows = []
        fixed_rows_size = 4
        for ticket in self.df_consolidado["symbol"].unique():

            df_temp = self.df_consolidado[
                self.df_consolidado["symbol"] == ticket
            ].copy()
            if len(df_temp) == fixed_rows_size:
                df_fixed_rows.append(df_temp)
            elif len(df_temp) > fixed_rows_size:
                df_fixed_rows.append(df_temp.tail(fixed_rows_size))
            else:
                while len(df_temp) < fixed_rows_size:
                    new_row = df_temp.tail(1).copy()
                    df_temp = df_temp.append(new_row)
                df_fixed_rows.append(df_temp)

        self.df_consolidado = pd.concat(df_fixed_rows)
        self.df_consolidado = self.df_consolidado.sort_values(by=["symbol", "asOfDate"])
        logging.info(f"df_fixed_rows.shape: {self.df_consolidado.shape}")
        del df_fixed_rows, df_temp

    def fill_nan(self):
        """
        Preenche o NaN com o proximo valor,
        """
        logging.info("Start")
        df_fixed_nan = []
        for ticket in self.df_consolidado["symbol"].unique():
            df_temp = (
                self.df_consolidado[self.df_consolidado["symbol"] == ticket]
                .copy()
                .reset_index()
            )
            df_temp.drop(columns=["index"], inplace=True)
            columns = df_temp.columns[2:]  # Pulando symbol e asOfDate
            for col in columns:
                if df_temp[col].isnull().any():
                    df_temp[col].fillna(method="ffill", inplace=True)
                    if df_temp[col].isnull().any():
                        df_temp[col].fillna(method="bfill", inplace=True)
            df_fixed_nan.append(df_temp)

        self.df_consolidado = pd.concat(df_fixed_nan)
        self.df_consolidado = self.df_consolidado.sort_values(by=["symbol", "asOfDate"])
        logging.info(f"self.df_consolidado.shape: {self.df_consolidado.shape}")
        del df_fixed_nan, df_temp

    def fill_no_data(self):
        """
        Prenche os NaN que colunas inteiras possuem
        """
        logging.info("Start")
        cols_with_nans = self.df_consolidado.columns[
            self.df_consolidado.isna().any()
        ].tolist()
        for col in cols_with_nans:
            self.df_consolidado[col].fillna(
                self.df_consolidado[col].median(), inplace=True
            )
        logging.info(f"self.df_consolidado.shape: {self.df_consolidado.shape}")

    def pivot(self):
        """
        Transofrm feature rows into columns
        """
        logging.info("Start")
        dfs = []
        for ticket in self.df_consolidado["symbol"].unique():
            row = self.df_consolidado[self.df_consolidado["symbol"] == ticket].copy()
            valid_columns = row.columns[2:]
            df_ticket = pd.DataFrame()
            df_ticket["symbol"] = [ticket]
            for col in valid_columns:
                values = list(row[col])
                values = [[v] for v in values]
                columns = [f"{col}_{index}" for index in range(4)]
                df_temp = pd.DataFrame(dict(zip(columns, values)))
                df_ticket = pd.concat((df_ticket, df_temp), axis=1)
            dfs.append(df_ticket)

        self.df_consolidado = pd.concat(dfs)
        logging.info(f"self.df_consolidado.shape: {self.df_consolidado.shape}")

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


TransformFundamentalista().transform()