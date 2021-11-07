# https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import RandomizedSearchCV

from base_func import avaliar, read_df


def definir_modelos_ml() -> dict:
    return {
        "RandomForest_base": RandomForestClassifier(random_state=42),
        "RandomForest_tunned": RandomForestClassifier(
            random_state=42, n_estimators=1600, min_samples_split=2, min_samples_leaf=5, bootstrap=False),
    }


def rodar_imprimir_modelos_ml(modelos, X_treino, X_teste, y_treino, y_teste):
    for nome_modelo in modelos:
        modelo = modelos[nome_modelo]
        modelo.fit(X_treino, y_treino)
        previsoes = modelo.predict(X_teste)
        avaliar(y_teste, previsoes, nome_modelo)
        modelos[nome_modelo] = modelo

    return modelos


def fit_tunning_rand_search_random_forest(X, y):

    print("Tunning Random Forest")

    # Create the random grid
    params = {'n_estimators': [1200, 1400, 1600],
                   'max_depth': [10, None],
                   'min_samples_split': [2, 4],
                   'min_samples_leaf': [1, 2, 5],
                   'bootstrap': [True, False]}

    print(params)

    rf_random = RandomizedSearchCV(estimator=RandomForestClassifier(),
                                   param_distributions=params,
                                   cv=3,
                                   verbose=2,
                                   scoring='recall',
                                   random_state=42,
                                   n_jobs=-1)

    # best params {'n_estimators': 1600, 'min_samples_split': 2, 'min_samples_leaf': 5, 'max_depth': 10, 'bootstrap': False}
    # precision 1 - 0.51 0.59 0.55

    # Fit the random search model
    return rf_random.fit(X, y)


def run_random_forest_tunning(df):

    X = df.drop(["decisao"], axis=1)
    y = df["decisao"]
    # split database train and test
    from sklearn.model_selection import train_test_split
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.15, random_state=2)

    # run randomized search first, then set params for grid
    resultado_grid = fit_tunning_rand_search_random_forest(X_treino, y_treino)

    print("Ajuste random forest feito")

    print("best params", resultado_grid.best_params_)
    print("best score", resultado_grid.best_score_)
    modelo_tunado = resultado_grid.best_estimator_

    previsoes = modelo_tunado.predict(X_teste)

    avaliar(y_teste, previsoes, "RandomForest Tunado")


if __name__ == '__main__':
    df = read_df(scaled=False)
    run_random_forest_tunning(df)
