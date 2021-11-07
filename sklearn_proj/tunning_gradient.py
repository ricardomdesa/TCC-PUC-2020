from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import RandomizedSearchCV

from base_func import avaliar, read_df


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

    # Create the random grid
    params = {'n_estimators': [800, 900, 1000],
                   'learning_rate': [0.01, 0.015, 0.009],
                   'max_features': ['auto', 'sqrt'],
                   'min_samples_split': [2, 5, 10],
                   # 'min_samples_leaf': [2, 5, 10],
                   'criterion': ["friedman_mse", "squared_error"]}

    print(params)

    rf_random = RandomizedSearchCV(estimator=GradientBoostingClassifier(),
                                   param_distributions=params,
                                   # n_iter=100,
                                   cv=3,
                                   verbose=2,
                                   # scoring="balanced_accuracy",
                                   scoring="recall",
                                   random_state=42,
                                   n_jobs=-1)

    # best params {'n_estimators': 1411, 'min_samples_split': 2, 'min_samples_leaf': 2, 'max_features': 'auto', 'max_depth': 110, 'learning_rate': 0.025, 'criterion': 'squared_error'}
    # {'n_estimators': 1000, 'min_samples_split': 10, 'max_features': 'auto', 'learning_rate': 0.01, 'criterion': 'squared_error'}
    # precision 1 - 0.57 0.68 0.62 - score 0.5609

    # Fit the random search model
    return rf_random.fit(X_treino, y_treino)


def run_gradient_boost_tunning(df):

    X = df.drop(["decisao"], axis=1)
    y = df["decisao"]

    # split database train and test
    from sklearn.model_selection import train_test_split
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.15, random_state=2)

    resultado_grid = fit_tunning_rand_search_gradient(X_treino, y_treino)

    print("Ajuste Gradient feito")

    print("Best params", resultado_grid.best_params_)
    print("Best score", resultado_grid.best_score_)

    modelo_tunado = resultado_grid.best_estimator_

    previsoes = modelo_tunado.predict(X_teste)

    avaliar(y_teste, previsoes, "RandomForest Tunado")


if __name__ == '__main__':
    df = read_df(scaled=True)
    run_gradient_boost_tunning(df)
