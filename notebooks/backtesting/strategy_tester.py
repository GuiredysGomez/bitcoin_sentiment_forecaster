import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from tqdm import tqdm
from sklearn.metrics import classification_report
from collections import Counter
import random

# Constante para definir la métrica de puntuación a extraer del reporte de clasificación.
# Se usará para optimizar el modelo en Optuna.
SCORE = "f1-score"

def trend_changes_true(y_test: np.array, y_pred: np.array) -> float:
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
    if "True" in report:
        return report["True"][SCORE]
    return 0.0

class Backtesting:
    """
    Clase para ejecutar un backtesting de estrategia de trading usando Walk-Forward Optimization.
    El modelo se re-optimiza y re-entrena periódicamente a medida que avanzan los datos de prueba.
    """
    def __init__(self, model_name, X_train, y_train, X_val, y_val, X_test, y_test, test_set, window_size=5, initial_capital=10000.0, optuna_trials_initial=150, optuna_trials=50):
        """
        Inicializa la clase de Backtesting.

        Args:
            X_train, y_train: Datos de entrenamiento originales.
            X_val, y_val: Datos de validación originales.
            X_test, y_test: Datos de prueba completos sobre los que se ejecutará el backtest.
            test_set (pd.DataFrame): DataFrame de prueba original, para obtener precios y fechas.
            window_size (int): Tamaño de la ventana de prueba antes de re-optimizar el modelo.
            initial_capital (float): Capital inicial para la simulación.
            optuna_trials_initial (int): Número de trials de Optuna para la primera optimización.
            optuna_trials (int): Número de trials de Optuna para las optimizaciones posteriores.
        """
        # Guardar los conjuntos de datos originales
        self.model_name = model_name
        self.X_train_orig = X_train
        self.y_train_orig = y_train
        self.X_val_orig = X_val
        self.y_val_orig = y_val
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
        self.y_val_m_orig = np.vectorize(self.cls_map.get)(self.y_val_orig)
        self.y_test_m = np.vectorize(self.cls_map.get)(self.y_test)

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
        lgb_train = lgb.Dataset(X_train, label=y_train, weight=[class_weights.get(c, 1.0) for c in y_train])
        lgb_val = lgb.Dataset(X_val, label=y_val, weight=[class_weights.get(c, 1.0) for c in y_val], reference=lgb_train)

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
        Selecciona el modelo y la lógica de optimización basados en self.model_name.
        """
        # Fijar semillas para reproducibilidad.
        np.random.seed(42)
        random.seed(42)

        # Inicializar los conjuntos de datos 'actuales' que se irán actualizando en cada paso.
        X_train_current, y_train_current_m = self.X_train_orig.copy(), self.y_train_m_orig.copy()
        X_val_current, y_val_current_m = self.X_val_orig.copy(), self.y_val_m_orig.copy()
        
        # Preparar el DataFrame para almacenar los resultados del backtest.
        backtest_df = self.test_set.copy()
        # Columna para almacenar la señal de trading (-1, 0, 1).
        backtest_df['signal'] = 0 
        # Identificar la columna de precios (referencia dia 0).
        price_col = [col for col in self.test_set.columns if 'open_d0' in col][-1]

        # Inicializar variables de la cartera.
        cash, position, portfolio_values = self.initial_capital, 0.0, []
        n_test = self.X_test.shape[0]

        # Inicializar contadores de transacciones y rentabilidad.
        num_buys_executed = 0
        num_sells_executed = 0
        winning_trades = 0
        losing_trades = 0
        total_gains = 0.0
        total_losses = 0.0
        # Para calcular la ganancia/pérdida de una operación cerrada.
        cash_at_last_buy = 0.0

        # Almacenar los mejores parámetros para reutilizarlos como punto de partida en el siguiente paso.
        best_params_from_previous_step = None

        # Bucle principal de Walk-Forward: itera sobre el conjunto de prueba en ventanas.
        for start in tqdm(range(0, n_test, self.window_size), desc="Walk-Forward Backtesting"):
            # --- 1. Re-optimización del Modelo ---

             # Calcular pesos de clase para el conjunto de entrenamiento actual.
            cnt = Counter(y_train_current_m)
            total = len(y_train_current_m)
            class_weights = {c: total / (len(cnt) * n) for c, n in cnt.items()}

            # Determinar el número de trials de Optuna para este paso.
            if start == 0:
                n_trials_for_this_step = self.optuna_trials_initial
            else:
                n_trials_for_this_step = self.optuna_trials

            # Crear y ejecutar el estudio de Optuna.
            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
            if best_params_from_previous_step:
                study.enqueue_trial(best_params_from_previous_step)

            # --- Selección de la función objetivo según el modelo ---
            if "lightgbm" in self.model_name.lower():
                objective_func = lambda trial: self._objective_lgbm(trial, X_train_current, y_train_current_m, X_val_current, y_val_current_m, class_weights)
            else:
                # Si en el futuro se añaden más modelos (ej. XGBoost), se definirían aquí.
                raise ValueError(f"El model_name '{self.model_name}' no está soportado.")

            study.optimize(objective_func, n_trials=n_trials_for_this_step)
            
            # --- 2. Re-entrenamiento del Modelo ---

            best_params = study.best_params
            # Guardar los mejores parámetros para el siguiente paso ---
            best_params_from_previous_step = best_params.copy()
            # Añadir los parámetros fijos necesarios para LightGBM.
            best_params.update({"objective": "multiclass", "num_class": 3, "metric": "multi_logloss", "verbosity": -1, "seed": 42})

            # Crear datasets con los datos actuales y los pesos de clase.
            lgb_train = lgb.Dataset(X_train_current, label=y_train_current_m, weight=[class_weights.get(c, 1.0) for c in y_train_current_m])
            lgb_val = lgb.Dataset(X_val_current, label=y_val_current_m, weight=[class_weights.get(c, 1.0) for c in y_val_current_m], reference=lgb_train)
            
            # --- Lógica de entrenamiento según el modelo ---
            if "lightgbm" in self.model_name.lower():
                # Entrenar el modelo final para esta ventana con los mejores hiperparámetros encontrados.
                model = lgb.train(best_params, lgb_train, num_boost_round=1000,
                                  valid_sets=[lgb_val], valid_names=["val"],
                                  callbacks=[lgb.early_stopping(50, verbose=False)])
            else:
                raise ValueError(f"El model_name '{self.model_name}' no está soportado para el entrenamiento.")

            # --- 3. Predicción y Simulación de Trading ---
            
            # Definir el rango de la ventana actual.
            end = min(start + self.window_size, n_test)
            # Bloque de datos de prueba para esta ventana.
            X_pred_block = self.X_test[start:end]
                    
            # Predecir las señales para el bloque actual.
            y_prob_block = model.predict(X_pred_block, num_iteration=model.best_iteration)
            y_pred_mapped = np.argmax(y_prob_block, axis=1)
            y_pred_signals = np.vectorize(self.inv_map.get)(y_pred_mapped)
            backtest_df.loc[start:end-1, 'signal'] = y_pred_signals
            
            # Iterar día a día dentro de la ventana para simular las operaciones.
            for i in range(start, end):
                # Comprueba si el número de la iteración actual (i) existe como una etiqueta de fila en backtest_df
                if i not in backtest_df.index: continue
                price, signal = backtest_df.loc[i, price_col], backtest_df.loc[i, 'signal']

                # Lógica de trading
                if signal == 1 and cash > 0: # Señal de COMPRA y tenemos efectivo.
                    cash_at_last_buy = cash # Guardar el efectivo en la última compra para calcular el Profit/Loss.
                    position, cash = cash / price, 0.0 # Invertir todo el efectivo.
                    num_buys_executed += 1  # Contar compra ejecutada

                elif signal == -1 and position > 0: # Señal de VENTA y tenemos posición.
                    # Calcular ganancia o pérdida de la operación cerrada (compra-venta).
                    if cash_at_last_buy > 0:
                        profit_or_loss = (position * price) - cash_at_last_buy
                        if profit_or_loss > 0:
                            winning_trades += 1
                            total_gains += profit_or_loss
                        else:
                            losing_trades += 1
                            total_losses += abs(profit_or_loss)
                        cash_at_last_buy = 0.0 # Resetear para el próximo ciclo.

                    # Vender toda la posición.
                    cash, position = position * price, 0.0
                    num_sells_executed += 1  # Contar venta ejecutada

                # Calcular y registrar el valor del portafolio al final del día.    
                portfolio_values.append(cash + position * price)

            # Imprimir progreso al final de cada ventana.
            if portfolio_values:
                current_portfolio_value = portfolio_values[-1]
                print(f"  -> Fin de la ventana (días {start}-{end-1}). Valor del Portafolio: ${current_portfolio_value:,.2f}")
                print(f"  -> Compras ejecutadas: {num_buys_executed}")
                print(f"  -> Ventas ejecutadas: {num_sells_executed}")

            # --- 4. Actualización de los Conjuntos de Datos (Walk-Forward) ---
            if end < n_test:
                # PASO 1: El bloque de 'test' que acabamos de usar se convierte en los datos más nuevos para 'val'.
                new_val_X = self.X_test[start:end]
                new_val_y = self.y_test_m[start:end]

                # PASO 2: El bloque más antiguo de 'val' se convierte en los datos más nuevos para 'train'.
                new_train_X = X_val_current[:self.window_size]
                new_train_y = y_val_current_m[:self.window_size]

                # PASO 3: Se actualiza el conjunto de entrenamiento ('train_current').
                # Se eliminan los datos más antiguos y se añaden los nuevos al final.
                X_train_current = np.vstack([X_train_current[self.window_size:], new_train_X])
                y_train_current_m = np.concatenate([y_train_current_m[self.window_size:], new_train_y])
                
                # PASO 4: Se actualiza el conjunto de validación ('val_current').
                # Se eliminan los datos más antiguos y se añaden los nuevos al final.
                X_val_current = np.vstack([X_val_current[self.window_size:], new_val_X])
                y_val_current_m = np.concatenate([y_val_current_m[self.window_size:], new_val_y])

        # --- 5. Finalización y Devolución de Resultados ---

        # Ajustar el DataFrame de resultados a la longitud de los valores del portafolio calculados.
        backtest_df = backtest_df.iloc[:len(portfolio_values)]
        backtest_df['portfolio_value'] = portfolio_values
        
        # Devolver el DataFrame con los resultados y las métricas de la simulación.
        return backtest_df, price_col, num_buys_executed, num_sells_executed, winning_trades, losing_trades, total_gains, total_losses