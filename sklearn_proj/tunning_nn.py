# https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import numpy as np
from sklearn.neural_network import MLPClassifier

from base_func import split_dados, avaliar, read_df


def definir_modelos_ml() -> dict:
    return {
        "NN_base": MLPClassifier(random_state=42, max_iter=100),
        "KNN_tunned": MLPClassifier(solver="sgd", random_state= 2, hidden_layer_sizes= 5, alpha= 0.1, activation= "relu"),
        # "KNN_tunned_grid": RandomForestClassifier(random_state=42, n_estimators= 200, min_samples_split= 12, min_samples_leaf= 3, max_features= 'sqrt', max_depth= 80, bootstrap= True),

    }


def rodar_imprimir_modelos_ml(df):
    X_treino, X_teste, y_treino, y_teste = split_dados(df)
    modelos = definir_modelos_ml()
    for nome_modelo in modelos:
        modelo = modelos[nome_modelo]
        modelo.fit(X_treino, y_treino)
        previsoes = modelo.predict(X_teste)
        avaliar(y_teste, previsoes, nome_modelo)
        modelos[nome_modelo] = modelo

    return modelos


def fit_tunning_rand_search_nn(X_treino, y_treino):
    print("Tunning randomized search NN")
    activation = ['relu', 'logistic', 'tanh']
    # activation = ['relu']
    solver = ['lbfgs', 'sgd', 'adam']
    learning_rate = ["constant", "adaptive"]
    random_state = [2]
    # random_state = np.arange(1, 12)
    solver = ['sgd']
    # max_iter = [2000]
    alpha = 10.0 ** -np.arange(1, 7)
    hidden_layer_sizes = [5]

    # Create the random grid
    random_grid = {'activation': activation,
                   'solver': solver,
                   # 'max_iter': max_iter,
                   'alpha': alpha,
                   'random_state': random_state,
                   # 'learning_rate': learning_rate,
                   'hidden_layer_sizes': hidden_layer_sizes,
                   }

    print(random_grid)

    rf_random = RandomizedSearchCV(estimator=MLPClassifier(random_state=42),
                                   param_distributions=random_grid,
                                   n_iter=1000,
                                   cv=3,
                                   verbose=2,
                                   random_state=42,
                                   scoring="recall_macro",
                                   n_jobs=-1)

    # best params {'n_estimators': 1400, 'min_samples_split': 10, 'min_samples_leaf': 2, 'max_features': 'sqrt', 'max_depth': 80, 'bootstrap': True}
    # precision 1 - 0.54 0.56 0.55

    # Fit the random search model
    return rf_random.fit(X_treino, y_treino)


def fit_tunning_grid_nn(X_treino, y_treino):
    print("Tunning nn after random search cv")

    grid = GridSearchCV(
            estimator=MLPClassifier(),
            param_grid={
                'n_neighbors': list(range(5, 30)),
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'minkowski'],
                # 'algorithm': ["auto", "ball_tree", "kd_tree", "brute"]
            },
            cv=10,
            n_jobs=-1,
            verbose=2,
            scoring="precision"
    )

    return grid.fit(X_treino, y_treino)


def run_nn_tunning(df):

    # split database train and test
    X_treino, X_teste, y_treino, y_teste = split_dados(df)

    # run randomized search first, then set params for grid
    resultado_grid = fit_tunning_rand_search_nn(X_treino, y_treino)
    # resultado_grid = fit_tunning_grid_knn(X_treino, y_treino)
    #
    print("Ajuste random forest feito")

    print("best params", resultado_grid.best_params_)
    print("best score", resultado_grid.best_score_)
    modelo_tunado = resultado_grid.best_estimator_
    previsoes = modelo_tunado.predict(X_teste)

    avaliar(y_teste, previsoes, "RandomForest Tunado")


if __name__ == '__main__':
    df = read_df(scaled=False)
    rodar_imprimir_modelos_ml(df)
    # run_nn_tunning(df)
