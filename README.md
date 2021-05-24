# Segundo projeto da disciplina de Tópicos Especias do Curso Superior de Tecnologia em Sistemas para Internet do IFPB

## Descrição

O objetivo é comparar os resultados de algoritmos de Machine Learning com problemas de classificação.

## Tecnologias

As seguintes ferramentas foram usadas na construção do projeto:

- [Python](https://www.python.org/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [Pandas](https://pandas.pydata.org/)

## Datasets

Foram escolhidas três bases de dados do [UCI](https://archive.ics.uci.edu/ml/index.php) para a comparação entre os algoritmos.

- [Wine Data Set](https://archive.ics.uci.edu/ml/datasets/Wine): Possui dados que são resultados de uma análise química de vinhos cultivados na mesma região na Itália, mas derivados de três cultivares diferentes;
- [Glass Identification Data Set](https://archive.ics.uci.edu/ml/datasets/Glass+Identification): Possui dados do Serviço de Ciência Forense dos EUA; tipos de vidro; definido em termos de seu teor de óxido (ou seja, Na, Fe, K, etc).
- [Breast Cancer Coimbra Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Coimbra): Possui dados sobre câncer de mama. As características clínicas foram observadas ou medidas em 64 pacientes com câncer de mama e 52 saudáveis.

## Algoritmos Testados

- Árvore de Busca (gini e entropy);
- kNN (k igual a 5 e 10);
- MLP com layer_sizes = (5, 4) e (6, 6), activation = tanh e relu e 3000 iterações;
- K-Means.

## Protocolo Experimental

Um k-fold cross-validation (k = 10), com 90% dos dados para Treinamento e o restante para Testes. Cada uma das divisões dos conjuntos foi utilizada para treinar cada algoritmo.

## Relatório

Tabela com as taxas de erro/acerto, que é a média dos 10 folds de teste, para cada algoritmo treinado. Valores em percentual.

- [Tabela com os resultados](https://docs.google.com/document/d/1r-IP3soIv6cGVw8zPRzxfr-vfBOQjUgoH6We-MEnuI0/edit?usp=sharing)
