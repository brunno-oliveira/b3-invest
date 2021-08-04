# Changelog
All notable changes to this project will be documented in this file.

## [1.2.0]
* Removido modelo de linear_regression
* Adicionado modelo random_forest
* Removido os ticker que não encontravam dados no fundamentalista [
    "BBDC3", "BBDC4", "BMLC11", "BMOB3", "BSEV3", "CJCT11", "CPFE3",
    "DEVA11", "HBRE3", "MFAI11", "NEMO5", "NEMO6", "RECR11", "TIET11",
    "TIET3", "TIET4", "URPR11"]
* Atualização da lib yfinance para corrigir um erro de Thread

## [1.1.0]
* Removido os tickers ['BBAS11', 'BBAS12', 'BEEF11', 'BPAR3', 'DMMO11', 'GOLL11', 'NEMO3', 'NEMO5', 'NEMO6', 'URPR11']
* Novos tickers [ATMP3, SOMA3, STBP3, STBP3, ESTR4, VALE3, ENMT3, ENMT4, DMMO3, AAPL34, CASN3, CEBR3, ABEV3]

## [1.0.2] - 2021-04-29
* Create simple linear regression model

## [1.0.1] - 2021-04-28
* Split base funciontions from into a model_base module