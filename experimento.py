# %%
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
pd.set_option("display.float_format", lambda x: "%.5f" % x)

# %%
def get_columns():
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
# %%
df = pd.read_parquet('data/df_fundamentalista.parquet')
df = df[get_columns()]
scaler = MinMaxScaler()
min_max = scaler.fit_transform(df.iloc[:,2:])
# %%
df_transformed = pd.DataFrame(
    columns=get_columns()[2:],
    data=min_max)
# %%
df_transformed['symbol'] = df['symbol']
df_transformed['asOfDate'] = df['asOfDate']
cols = df_transformed.columns.tolist()
cols = cols[-2:] + cols[:-2]
df_transformed = df_transformed[cols]
# %%
df_transformed.head()
# %%
