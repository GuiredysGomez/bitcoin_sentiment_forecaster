import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from tqdm import tqdm
from sklearn.metrics import classification_report
from collections import Counter
import random
from sklearn.model_selection import train_test_split
# imports de modelos scikit/xgboost
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Constante para definir la métrica de puntuación a extraer del reporte de clasificación.
# Se usará para optimizar el modelo en Optuna.
SCORE = "f1-score"

def trend_changes_true(y_test: np.array, y_pred: np.array) -> float:
    """
    Calculate the trend changes score based on the test and predicted values.
    
    Args:
        y_test (np.array): True labels.
        y_pred (np.array): Predicted labels.
        
    Returns:
        float: The trend changes score.
    """
    y_df = pd.DataFrame([y_test, y_pred]).T
    y_df.columns = ["y_test", "y_pred"]
    y_df["y_test_shifted"] = y_df["y_test"].shift(-1)
    y_df["is_changed_trend_test"] = y_df["y_test"] != y_df["y_test_shifted"]
    y_df["y_predict_shifted"] = y_df["y_pred"].shift(-1)
    y_df["is_changed_trend_predict"] = y_df["y_pred"] != y_df["y_predict_shifted"]
    report = classification_report(
        y_df["is_changed_trend_test"][:-1],
        y_df["is_changed_trend_predict"][:-1],
        output_dict=True,
        zero_division=0
    )
    return report["True"][SCORE]

class Backtesting:
    """
    Clase para ejecutar un backtesting de estrategia de trading usando Walk-Forward Optimization.
    El modelo se re-optimiza y re-entrena periódicamente a medida que avanzan los datos de prueba.
    """
    def __init__(self, model_name, X_train_val, y_train_val, X_test, y_test, 
                 test_set, window_size=5, initial_capital=10000.0, optuna_trials_initial=150, 
                 optuna_trials=50):
        """
        Inicializa la clase de Backtesting.
        Args:
            X_train_val, y_train_val: Datos de entrenamiento originales (train + val).
            X_test, y_test: Datos de prueba completos sobre los que se ejecutará el backtest.
            test_set (pd.DataFrame): DataFrame de prueba original, para obtener precios y fechas.
            window_size (int): Tamaño de la ventana de prueba antes de re-optimizar el modelo.
            initial_capital (float): Capital inicial para la simulación.
            optuna_trials_initial (int): Número de trials de Optuna para la primera optimización.
            optuna_trials (int): Número de trials de Optuna para las optimizaciones posteriores.
        """
        # Guardar los conjuntos de datos originales
        self.model_name = model_name
        self.X_train_orig = X_train_val
        self.y_train_orig = y_train_val
        self.X_test = X_test
        self.y_test = y_test
        self.test_set = test_set

        # Parámetros de configuración del backtest
        self.window_size = window_size
        self.initial_capital = initial_capital
        self.optuna_trials_initial = optuna_trials_initial
        self.optuna_trials = optuna_trials

        # Mapeo de clases: se requiere clases numéricas (0, 1, 2).
        self.cls_map = {-1: 0, 0: 1, 1: 2}
        self.inv_map = {v: k for k, v in self.cls_map.items()}

        # Aplicar el mapeo a los conjuntos de etiquetas 'y'.
        self.y_train_m_orig = np.vectorize(self.cls_map.get)(self.y_train_orig)
        self.y_test_m = np.vectorize(self.cls_map.get)(self.y_test)

    def _tuning_model(self, X_train_val, y_train_val, n_trials, prev_best_params):
        """
        Realiza la optimización de hiperparámetros y el re-entrenamiento del modelo.
        """
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.20, shuffle=False)  
        # 20% para validation

        # Calcular pesos de clase para el conjunto de entrenamiento actual (para LightGBM).
        cnt = Counter(y_train)
        total = len(y_train)
        class_weights = {c: total / (len(cnt) * n) for c, n in cnt.items()}

        # Crear y ejecutar el estudio de Optuna.
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        if prev_best_params:
            study.enqueue_trial(prev_best_params)

        # Seleccionar la función objetivo según el modelo.
        if "lightgbm" in self.model_name.lower():
            objective_func = lambda trial: self._objective_lgbm(trial, X_train, y_train, X_val, y_val, class_weights)
        else:
            raise ValueError(f"El model_name '{self.model_name}' no está soportado.")

        study.optimize(objective_func, n_trials=n_trials)
        best_params = study.best_params

        # Re-entrenar el modelo con los mejores hiperparámetros.
        if "lightgbm" in self.model_name.lower():
            best_params.update({"objective": "multiclass", "num_class": 3, "metric": "multi_logloss", "verbosity": -1, "seed": 42})
            lgb_train = lgb.Dataset(X_train, label=y_train, weight=[class_weights.get(c, 1.0) for c in y_train])
            lgb_val = lgb.Dataset(X_val, label=y_val, weight=[class_weights.get(c, 1.0) for c in y_val], reference=lgb_train)
            model = lgb.train(best_params, lgb_train, num_boost_round=1000,
                              valid_sets=[lgb_val], valid_names=["val"],
                              callbacks=[lgb.early_stopping(50, verbose=False)])
        else:
            raise ValueError(f"El model_name '{self.model_name}' no está soportado para el entrenamiento.")

        return model, best_params

    def _process_signal(self, signal, price):
        """
        Procesa una señal de trading y actualiza el estado del portafolio.
        """
        if signal == 1 and self.cash > 0:  # Señal de COMPRA
            self.position = self.cash / price
            self.cash = 0.0
            self.num_buys_executed += 1
        elif signal == -1 and self.position > 0:  # Señal de VENTA
            self.cash = self.position * price
            self.position = 0.0
            self.num_sells_executed += 1
        
        # Calcular y registrar el valor del portafolio al final del día.
        self.portfolio_values.append(self.cash + self.position * price)

    def _update_reference_strategies(self, i, price):
        """
        Actualiza los portafolios de las estrategias de referencia (Buy & Hold y DCA).
        """
        # --- Lógica de la Estrategia Buy & Hold ---
        if i == 0: # Comprar todo el primer día
            self.b_and_h_position = self.initial_capital / price
        self.b_and_h_portfolio_values.append(self.b_and_h_position * price)

        # --- Lógica de la Estrategia Dollar Cost Averaging (DCA) ---
        # self.dca_investment_interval es el número de días que deben pasar entre cada inversión.
        # self.num_investments es el número total de inversiones planeadas
        if i % self.dca_investment_interval == 0 and self.dca_investments_made < self.num_investments:
            # Se calcula la cantidad de Bitcoin que se puede comprar con la cantidad fija de inversión.
            btc_bought = self.dca_investment_amount / price
            # La cantidad de Bitcoin recien comprada se suma a la posición total de Bitcoin
            self.dca_position += btc_bought
            # Se resta la cantidad invertida del efectivo disponible
            self.dca_cash -= self.dca_investment_amount
            # Se incrementa el contador de inversiones DCA realizadas
            self.dca_investments_made += 1
        # El valor del portafolio DCA es el valor de la posición + el efectivo restante
        self.dca_portfolio_values.append((self.dca_position * price) + self.dca_cash)

    def _data_update(self, X_train_current, y_train_current_m, X_test, y_test):
        """
        Actualiza los conjuntos de datos de entrenamiento, agregando la nueva predicción y eliminando el dato más antiguo.
        """
        # Actualizar los conjuntos de datos para el siguiente paso.
        X_train_current = np.append(X_train_current, X_test, axis=0)
        y_train_current_m = np.append(y_train_current_m, y_test)

        # 2. Eliminar el dato más antiguo (el primero) para mantener el tamaño de la ventana.
        X_train_current = X_train_current[1:]
        y_train_current_m = y_train_current_m[1:]

        return X_train_current, y_train_current_m

    def _objective_lgbm(self, trial, X_train, y_train, X_val, y_val, class_weights):
        """
        Función objetivo para la optimización de hiperparámetros con Optuna.
        """
        # Definir el espacio de búsqueda de hiperparámetros para LightGBM.
        param = {
            "objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
            "verbosity": -1, 
            "seed": 42,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        }
        # Crear datasets de LightGBM con pesos de clase para manejar el desbalance.
        lgb_train = lgb.Dataset(X_train, label=y_train, 
                                weight=[class_weights.get(c, 1.0) for c in y_train])
        lgb_val = lgb.Dataset(X_val, label=y_val, 
                              weight=[class_weights.get(c, 1.0) for c in y_val], reference=lgb_train)

        # Entrenar el modelo con early stopping para evitar sobreajuste.
        model = lgb.train(param, lgb_train, num_boost_round=1000,
                          valid_sets=[lgb_val], valid_names=["val"],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
        
        # Realizar predicciones en el conjunto de validación.
        y_val_prob = model.predict(X_val, num_iteration=model.best_iteration)
        y_val_pred = np.argmax(y_val_prob, axis=1)

        # Devolver la puntuación de cambio de tendencia, que Optuna maximizará.
        return trend_changes_true(y_val, y_val_pred)

    def run(self):
        """
        Ejecuta el proceso completo de backtesting con Walk-Forward Optimization.
        """
        # Fijar semillas para reproducibilidad.
        np.random.seed(42)
        random.seed(42)

        # Inicializar los conjuntos de datos 'actuales' que se irán actualizando en cada paso.
        X_train_current, y_train_current_m = self.X_train_orig.copy(), self.y_train_m_orig.copy()

        # Preparar el DataFrame para almacenar los resultados del backtest.
        backtest_df = self.test_set.copy()
        backtest_df['signal'] = 0            # decisión tomada hoy para ejecutar mañana
        # backtest_df['exec_signal'] = 0       # señal efectivamente ejecutada hoy

        # Identificar la columna de precios (referencia dia 0).
        price_col = [col for col in self.test_set.columns if 'open_d0' in col][-1]

        # Inicializar variables de la cartera.
        self.cash, self.position, self.portfolio_values = self.initial_capital, 0.0, []
        n_test = self.X_test.shape[0]
        self.num_buys_executed = 0
        self.num_sells_executed = 0
        best_params_from_previous_step = None
        model = None
        pending_signal = 0  # señal decidida ayer, se ejecuta hoy

        # INICIALIZACIÓN DE ESTRATEGIAS DE REFERENCIA ---
        # ----Parámetros para la estrategia Buy & Hold----
        self.b_and_h_position = 0.0
        self.b_and_h_portfolio_values = []
        # ----Parámetros para la estrategia DCA----
        self.num_investments = 12
        # self.dca_investment_amount es la cantidad de dinero a invertir en cada operación DCA.
        self.dca_investment_amount = self.initial_capital / self.num_investments
        # self.dca_investment_interval es el número de días que deben pasar entre cada inversión.
        self.dca_investment_interval = n_test // self.num_investments if n_test > self.num_investments else 1
        # self.dca_cash es el efectivo disponible para las inversiones DCA.
        self.dca_cash = self.initial_capital
        # self.dca_position es la cantidad total de Bitcoin comprada a través de DCA.
        self.dca_position = 0.0
        # self.dca_investments_made es el número de inversiones DCA realizadas.
        self.dca_investments_made = 0
        # self.dca_portfolio_values almacena el valor del portafolio DCA a lo largo del tiempo.
        self.dca_portfolio_values = []

        for i in tqdm(range(0, n_test), desc="Walk-Forward Backtesting"):
            # 0) Ejecutar hoy lo decidido ayer (evita look-ahead)
            price = backtest_df.loc[i, price_col]
            self._process_signal(pending_signal, price) # Check
            # backtest_df.loc[i, 'exec_signal'] = pending_signal # SE PUEDE QUITAR POR LOS MOMENTOS, NO SE UTILIZA

            # 1) Estrategias de referencia con el precio de hoy
            self._update_reference_strategies(i, price) # Check

            # 2) Actualizar el train con el ejemplo i-1 (su target ya es conocido)
            if i > 0:
                X_train_current, y_train_current_m = self._data_update(
                    X_train_current, y_train_current_m,
                    self.X_test[i-1:i], self.y_test_m[i-1:i]
                ) # Check

            # 3) Re-optimizar cuando toque (con el train ya actualizado)
            if  i % self.window_size == 0:
                n_trials = self.optuna_trials_initial if i == 0 else self.optuna_trials
                model, best_params = self._tuning_model(
                    X_train_current, y_train_current_m,
                    n_trials, best_params_from_previous_step
                ) # Check (pero confiando que esta bien el training)
                best_params_from_previous_step = best_params.copy()

            # 4) Decidir la señal de mañana con los datos de hoy (no ejecutar hoy)
            X_pred = self.X_test[i:i+1]
            y_prob = model.predict(X_pred, num_iteration=model.best_iteration)
            y_pred_mapped = np.argmax(y_prob, axis=1)
            pending_signal = self.inv_map[y_pred_mapped[0]]
            backtest_df.loc[i, 'signal'] = pending_signal

        # FINALIZACIÓN Y DEVOLUCIÓN DE RESULTADOS
        backtest_df['walk_forward_portfolio'] = self.portfolio_values
        backtest_df['buy_and_hold_portfolio'] = self.b_and_h_portfolio_values
        backtest_df['dca_portfolio'] = self.dca_portfolio_values
        
        return (backtest_df, X_train_current, y_train_current_m)