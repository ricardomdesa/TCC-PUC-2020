from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, precision_score

import numpy as np

from base_func import split_dados, avaliar, read_df


def definir_modelos_ml() -> dict:
    return {
        "GradientBoost": GradientBoostingClassifier(random_state=42),
        "GradientBoost_tunned": GradientBoostingClassifier(random_state=42),
    }


def rodar_imprimir_modelos_ml(modelos, X_treino, X_teste, y_treino, y_teste):
    for nome_modelo in modelos:
        modelo = modelos[nome_modelo]
        modelo.fit(X_treino, y_treino)
        previsoes = modelo.predict(X_teste)
        avaliar(y_teste, previsoes, nome_modelo)
        modelos[nome_modelo] = modelo
    
    return modelos


def fit_tunning_rand_search_gradient(X_treino, y_treino):
    print("Tunning randomized search Gradient Boost")
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=700, stop=1500, num=10)]
    # Learning rate default = 0.1
    learning_rate = [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(3)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Criterion
    criterion = ["friedman_mse", "squared_error"]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'learning_rate': learning_rate,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'criterion': criterion}

    print(random_grid)

    rf_random = RandomizedSearchCV(estimator=GradientBoostingClassifier(),
                                   param_distributions=random_grid,
                                   n_iter=100,
                                   cv=3,
                                   verbose=2,
                                   random_state=42,
                                   n_jobs=-1)

    # best params {'n_estimators': 1411, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'auto', 'max_depth': 110, 'learning_rate': 0.025, 'criterion': 'squared_error'}
    # precision 1 - 0.53 0.55 0.54 - score 0.5454

    # Fit the random search model
    return rf_random.fit(X_treino, y_treino)


def fit_tunning_grid_gradient(X_treino, y_treino):
    print("Tunning Gradient boost after random search cv")

    grid = GridSearchCV(
            estimator=GradientBoostingClassifier(),
            param_grid={
                'criterion': ["squared_error"],
                'learning_rate': [0.02, 0.025, 0.03],
                'max_depth': [90, 110, 120],
                'max_features': ['auto'],
                'min_samples_leaf': [2, 3],
                'min_samples_split': [2, 3],
                'n_estimators': [1344, 1411]
            },
            cv=3,
            n_jobs=-1,
            verbose=2
    )

    return grid.fit(X_treino, y_treino)


def run_gradient_boost_tunning(df):
    # split database train and test
    X_treino, X_teste, y_treino, y_teste = split_dados(df)

    # resultado_grid = fit_tunning_rand_search_gradient(X_treino, y_treino)
    resultado_grid = fit_tunning_grid_gradient(X_treino, y_treino)

    print("Ajuste Gradient feito")

    print("Best params", resultado_grid.best_params_)
    print("Best score", resultado_grid.best_score_)

    modelo_tunado = resultado_grid.best_estimator_
    previsoes = modelo_tunado.predict(X_teste)

    avaliar(y_teste, previsoes, "RandomForest Tunado")


if __name__ == '__main__':
    df = read_df(scaled=False)
    run_gradient_boost_tunning(df)
