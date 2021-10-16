# https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import numpy as np

from base_func import split_dados, avaliar, read_df


def definir_modelos_ml() -> dict:
    return {
        "RandomForest_base": RandomForestClassifier(random_state=42),
        "RandomForest_tunned": RandomForestClassifier(random_state=42, n_estimators= 1400, min_samples_split= 10, min_samples_leaf= 2, max_features= 'sqrt', max_depth= 80, bootstrap= True),
        "RandomForest_tunned_grid": RandomForestClassifier(random_state=42, n_estimators= 200, min_samples_split= 12, min_samples_leaf= 3, max_features= 'sqrt', max_depth= 80, bootstrap= True),

    }


def rodar_imprimir_modelos_ml(modelos, X_treino, X_teste, y_treino, y_teste):
    for nome_modelo in modelos:
        modelo = modelos[nome_modelo]
        modelo.fit(X_treino, y_treino)
        previsoes = modelo.predict(X_teste)
        avaliar(y_teste, previsoes, nome_modelo)
        modelos[nome_modelo] = modelo

    return modelos


def fit_tunning_rand_search_random_forest(X_treino, y_treino):
    print("Tunning Random Forest")
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    print(random_grid)

    rf_random = RandomizedSearchCV(estimator=RandomForestClassifier(),
                                   param_distributions=random_grid,
                                   n_iter=100,
                                   cv=3,
                                   verbose=2,
                                   random_state=42,
                                   n_jobs=-1)

    # best params {'n_estimators': 1400, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 80, 'bootstrap': True}
    # precision 1 - 0.54 0.56 0.55

    # Fit the random search model
    return rf_random.fit(X_treino, y_treino)


def fit_tunning_grid_random_forest(X_treino, y_treino):
    print("Tunning Random Forest after random search cv")

    grid = GridSearchCV(
            estimator=RandomForestClassifier(),
            param_grid={
                'bootstrap': [True],
                'max_depth': [80, 90, 100, 110],
                'max_features': ['sqrt'],
                'min_samples_leaf': [2, 3, 4],
                'min_samples_split': [8, 10, 12],
                'n_estimators': [200, 600, 1000, 1400]
            },
            cv=3,
            n_jobs=-1,
            verbose=2
    )

    return grid.fit(X_treino, y_treino)


def run_random_forest_tunning(df):

    # split database train and test
    X_treino, X_teste, y_treino, y_teste = split_dados(df)

    # run randomized search first, then set params for grid
    resultado_grid = fit_tunning_rand_search_random_forest(X_treino, y_treino)
    # resultado_grid = fit_tunning_grid_random_forest(X_treino, y_treino)

    print("Ajuste random forest feito")

    print("best params", resultado_grid.best_params_)
    modelo_tunado = resultado_grid.best_estimator_
    previsoes = modelo_tunado.predict(X_teste)

    avaliar(y_teste, previsoes, "RandomForest Tunado")


if __name__ == '__main__':
    df = read_df(scaled=False)
    run_random_forest_tunning(df)
