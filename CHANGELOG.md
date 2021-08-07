# Changelog
All notable changes to this project will be documented in this file.

## [1.2.0]
* Removido modelo de linear_regression
* Adicionado modelo random_forest
* Atualização da lib yfinance para corrigir um erro de Thread
* Salva em JSON a lista de ticker que não possuirem registros superior a 254 dias
* Removido os tickers lower data na tranformação do histórico
* Adicionado a coluna 'volume' na base de histórico

## [1.1.0]
* Removido os tickers ['BBAS11', 'BBAS12', 'BEEF11', 'BPAR3', 'DMMO11', 'GOLL11', 'NEMO3', 'NEMO5', 'NEMO6', 'URPR11']
* Novos tickers [ATMP3, SOMA3, STBP3, STBP3, ESTR4, VALE3, ENMT3, ENMT4, DMMO3, AAPL34, CASN3, CEBR3, ABEV3]

## [1.0.2] - 2021-04-29
* Create simple linear regression model

## [1.0.1] - 2021-04-28
* Split base funciontions from into a model_base module