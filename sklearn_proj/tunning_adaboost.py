from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import RandomizedSearchCV


from base_func import avaliar, read_df


def definir_modelos_ml() -> dict:
    return {
        "AdaBoostClassifier": AdaBoostClassifier(random_state=42),
        "AdaBoostClassifier_tunned": AdaBoostClassifier(random_state=42, n_estimators=80, learning_rate=0.1),
    }


def rodar_imprimir_modelos_ml(modelos, X_treino, X_teste, y_treino, y_teste):
    for nome_modelo in modelos:
        modelo = modelos[nome_modelo]
        modelo.fit(X_treino, y_treino)
        previsoes = modelo.predict(X_teste)
        avaliar(y_teste, previsoes, nome_modelo)
        modelos[nome_modelo] = modelo
    
    return modelos


def fit_tunning_rand_search_ada(X_treino, y_treino):
    print("Tunning randomized search AdaBoost")

    # Create the random grid
    param = {'n_estimators': [50, 80, 90, 100],
             'learning_rate': [0.2, 1.0, 0.1],
             'algorithm': ["SAMME", "SAMME.R"]}

    print(param)

    rf_random = RandomizedSearchCV(estimator=AdaBoostClassifier(),
                                   param_distributions=param,
                                   cv=3,
                                   verbose=2,
                                   # scoring="balanced_accuracy",
                                   scoring="accuracy",
                                   random_state=42,
                                   n_jobs=-1)

    # best params {'n_estimators': 80, 'learning_rate': 0.1, 'algorithm': 'SAMME.R'}
    # precision 1 - 0.59 0.63 0.61 - score 0.5704

    # Fit the random search model
    return rf_random.fit(X_treino, y_treino)


def run_ada_boost_tunning(df):

    X = df.drop(["decisao"], axis=1)
    y = df["decisao"]

    # split database train and test
    from sklearn.model_selection import train_test_split
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.15, random_state=2)

    resultado_grid = fit_tunning_rand_search_ada(X_treino.values, y_treino.values)

    print("Ajuste Adaboost feito")

    print("Best params", resultado_grid.best_params_)
    print("Best score", resultado_grid.best_score_)

    modelo_tunado = resultado_grid.best_estimator_

    previsoes = modelo_tunado.predict(X_teste)

    avaliar(y_teste, previsoes, "Adaboost Tunado")


if __name__ == '__main__':
    df = read_df(scaled=True)
    run_ada_boost_tunning(df)
