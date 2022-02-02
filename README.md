## MODELO MACHINE LEARNING PARA ANÁLISE FUNDAMENTALISTA DE EMPRESAS LISTADAS NA BOLSA DE VALORES BRASILEIRA

Trabalho de Conclusão apresentado ao Curso de Especialização em Ciência de Dados e Big Data como requisito parcial à obtenção do título de especialista.

----

Criacao de um modelo de ML para predicao de compra ou venda de papeis da bolsa de valores brasileira, baseado em analise fundamentalista.

Este modelo utilizara dados historicos trimestrais de balancos patrimoniais e demonstrativos financeiros como variaveis explicativas.

Para criacao da variavel explicada, target, foi utilizado dados historicos trimestrais de cotacao e indice Ibovespa.

----

Os jupyter notebooks foram divididos de acordo com as etapas de Machine learning, partindo da aquisicao dos dados ate os tunning do modelo de ML.

- **aquisicao.ipynb**: Realiza o carregamento dos dados de balanco patrimonial, demonstrativos financeiros e cotações por empresas do Ibovespa.
- **tratamento_col_target.ipynb**: Faz o tratamento dos dataframes, padronizacao das variaveis de entrada e criacao da coluna target/decisao.
- **analis_descritiva.ipynb**: Realiza a analise descritiva, Feature engineer e Feature selections.
- **analise_ml.ipynb**: Treinamento de tres modelos de machine learning com parametros default.
- **analise_ml_tunning.ipynb**: Realiza o ajuste fino (tunning) dos hiperparametros dos modelos de ML treinados.
