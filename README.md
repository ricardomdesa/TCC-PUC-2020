## MODELO MACHINE LEARNING PARA ANÁLISE FUNDAMENTALISTA DE EMPRESAS LISTADAS NA BOLSA DE VALORES BRASILEIRA

Trabalho de Conclusão apresentado ao Curso de Especialização em Ciência de Dados e Big Data como requisito parcial à obtenção do título de especialista.

----

Criacao de um modelo de ML para predicao de compra ou venda de papeis da bolsa de valores brasileira, baseado em analise fundamentalista.

Este modelo utilizara dados historicos trimestrais de balancos patrimoniais e demonstrativos financeiros como variaveis explicativas.

Para criacao da variavel explicada, target, foi utilizado dados historicos trimestrais de cotacao e indice Ibovespa.

----

Os jupyter notebooks foram divididos de acordo com as etapas de Machine learning, partindo da aquisicao dos dados ate os tunning do modelo de ML.

```
cd notebooks/
```

- **1_coleta_dados.ipynb**: Realiza o carregamento dos dados de balanco patrimonial, demonstrativos financeiros e cotações por empresas do Ibovespa.
- **2_pre_tratamento.ipynb**: Faz o tratamento dos dataframes, padronizacao das variaveis de entrada e criacao da coluna target/decisao.
- **3_analise_exploratoria.ipynb**: Realiza a analise descritiva, Feature engineer e Feature selections.
- **4_modelos_ml_default_.ipynb**: Treinamento de tres modelos de machine learning com parametros default.
- **5_tunning_ml.ipynb**: Realiza o ajuste fino (tunning) dos hiperparametros dos modelos de ML treinados.
- **final_predict_test/6_teste_resultados.ipynb**: Realiza teste com novos dados utilizando o modelo ML.

Os dados de entrada necessarios para o primeiro notebook, de coleta de dados, encontram-se na pasta **dados/balancos** e **dados/cotacoes**.
Os arquivos ".joblib" sao arquivos gerados pelo notebook a fim de guardar os DataFrames com os dados de excel ja carregados para agilizar o processo em caso de uma nova execucao. 
No final de cada notebook tambem é gerado arquivos ".joblib" na pasta **out** para serem lidos no proximo notebook.