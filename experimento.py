import numpy as np
import pandas as pd

pd.set_option("display.float_format", lambda x: "%.5f" % x)

df = pd.read_parquet("data/df_fundamentalista.parquet")
df_consolidado = pd.read_parquet("data/df_consolidado.parquet")


dfs = []
for ticket in df["symbol"].unique():
    row = df[df["symbol"] == ticket].copy()
    valid_columns = df.columns[2:]
    df_ticket = pd.DataFrame()
    df_ticket["symbol"] = [ticket]
    for col in valid_columns:
        values = list(row[col])
        values = [[v] for v in values]
        columns = [f"{col}_{index}" for index in range(4)]
        df_temp = pd.DataFrame(dict(zip(columns, values)))
        df_ticket = pd.concat((df_ticket, df_temp), axis=1)
    dfs.append(df_ticket)

df = pd.concat(dfs)
print("opa")
