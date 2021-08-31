# Changelog
All notable changes to this project will be documented in this file.

## [2.1.0]
* Add GridSearch
* Change model folders
* New log folders

## [2.0.0]
* Removido modelo de linear_regression
* Adicionado modelo random_forest
* Atualização da lib yfinance para corrigir um erro de Thread
* Salva em JSON a lista de tickers que deram erro na extração do histórico, 
ações com menos de 254 dias e não encontrados do financeiro
* Removido os tickers lower e failed data na tranformação do histórico
* Atualizado lista de tickers
* Adicionados arquivos de sample utilizado para imagens na documetação
* Experimento de feature engineering com os dados do financeiro
* Modelos 2.0

## [1.1.0]
* Removido os tickers ['BBAS11', 'BBAS12', 'BEEF11', 'BPAR3', 'DMMO11', 'GOLL11', 'NEMO3', 'NEMO5', 'NEMO6', 'URPR11']
* Novos tickers [ATMP3, SOMA3, STBP3, STBP3, ESTR4, VALE3, ENMT3, ENMT4, DMMO3, AAPL34, CASN3, CEBR3, ABEV3]

## [1.0.2] - 2021-04-29
* Create simple linear regression model

## [1.0.1] - 2021-04-28
* Split base funciontions from into a model_base module