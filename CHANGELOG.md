# Changelog
All notable changes to this project will be documented in this file.

## [3.1.0]
* Revisão das datas
* Persistência de novo history.zip

## [3.0.0]
* Arquivo de configuração
* Alteração nas datas de validação e predição
* Fixado a data dos experimentos devido a missing data no hitórico
* Zipado os data sources

## [2.2.1]
* Arredondado close para 2 casas decimais
* Módulo para consolidar todos os resultados
* Salvando as métricas em JSON

## [2.2.0]
* Predict com todos os experimentos
* Adicionado RMSE como scoring do GridSearch
* Refatoração da métricas
* Salvar as imagens de predicted x y_test para cada experimento
* Persistir os resultados em formato pickle de cada experimento
* Remove wandb

## [2.1.0]
* Add GridSearch
* Change model folders
* New log folders
* Fixado a data de ínicio
* Adicionado mais 28 dias a data final, para usar nos experimentos
* Datasets para cada experimento
* Novas colunas de data
* Tipo de modelos: Com ou Sem Features

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