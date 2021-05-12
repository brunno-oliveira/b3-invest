# %%
from bs4 import BeautifulSoup
import pandas as pd
import requests
import json
import re

pd.set_option("display.float_format", lambda x: "%.5f" % x)
# %%
df = pd.read_parquet("data/fundamentalista/df_fundamentalista.parquet")
# %%
df["currencyCode"].unique()

# %%
df.index("COPH34.SA")
# %%
for col in df.columns:
    print(col)
# %%
df[["asOfDate", "TaxProvision"]]
# %%
