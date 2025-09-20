import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from tqdm import tqdm
from sklearn.metrics import classification_report
from collections import Counter
import random
from sklearn.model_selection import train_test_split
# Imports de modelos scikit/xgboost
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
# Imports para redes neuronales
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

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
                 test_set, window_size=5, initial_capital=10000.0, optuna_trials_initial=700, optuna_trials=150):
        """
        Inicializa la clase de Backtesting.
        Args:
            X_train_val, y_train_val: Datos de entrenamiento originales (train + val).
            X_test, y_test: Datos de prueba completos sobre los que se ejecutará el backtest.
            test_set (pd.DataFrame): DataFrame de prueba original, para obtener precios y fechas.
            window_size (int): Tamaño de la ventana de prueba antes de re-optimizar el modelo.
            initial_capital (float): Capital inicial para la simulación.
            optuna_trials (int): Número de trials de Optuna para las optimizaciones posteriores.
            pretrained_model: Modelo preentrenado opcional para usar en el primer paso.
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

        # Identificar la columna de precios (referencia dia 0).
        self.price_col = [col for col in self.test_set.columns if 'open_d0' in col][-1]

        # INICIALIZACIÓN DE ESTRATEGIAS DE REFERENCIA ---
        # ----Parámetros para la estrategia DCA----
        self.num_investments = 12
        # self.dca_investment_amount es la cantidad de dinero a invertir en cada operación DCA.
        self.dca_investment_amount = self.initial_capital / self.num_investments
        # self.dca_investment_interval es el número de días que deben pasar entre cada inversión.
        self.dca_investment_interval = len(self.X_test) // self.num_investments if len(self.X_test) > self.num_investments else 1

        # Variables para el bucle de backtesting.
        self.n_test = self.X_test.shape[0]

    def _initialize_state(self):
        """Inicializa o resetea el estado del backtest antes de una ejecución."""
        # Fijar semillas para reproducibilidad.
        np.random.seed(42)
        random.seed(42)
        tf.random.set_seed(42)

        # Inicializar variables de la cartera principal.
        self.cash = self.initial_capital
        self.position = 0.0
        self.portfolio_values = []
        self.num_buys_executed = 0
        self.num_sells_executed = 0

        # ---- Registro de Trades ----
        self.trades = []
        self.current_trade = {}

        # ---- Estado para la estrategia Buy & Hold ----
        self.b_and_h_position = 0.0
        self.b_and_h_portfolio_values = []

        # ---- Estado para la estrategia DCA ----
        self.dca_cash = self.initial_capital
        self.dca_position = 0.0
        self.dca_investments_made = 0
        self.dca_portfolio_values = []

        # Inicializar los conjuntos de datos 'actuales' que se irán actualizando en cada paso.
        self.X_train_current = self.X_train_orig.copy()
        self.y_train_current_m = self.y_train_m_orig.copy()

        # Variables para el bucle de backtesting.
        self.best_params_from_previous_step = None
        self.model = None
        self.pending_signal = 0  # señal decidida ayer, se ejecuta hoy

    def _get_model_instance(self, model_name, params={}):
        """Crea una instancia de un modelo basado en su nombre."""
        name = model_name.lower()
        if name == 'lightgbm':
            # LightGBM se maneja de forma especial en su entrenamiento
            return None 
        elif name == 'random_forest':
            return RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        elif name == 'gradientboosting':
            return GradientBoostingClassifier(**params, random_state=42)
        elif name == 'histgradientboosting':
            return HistGradientBoostingClassifier(**params, random_state=42)
        elif name == 'xgbclassifier':
            return XGBClassifier(**params, random_state=42, use_label_encoder=False, eval_metric='mlogloss')
        elif name == 'svm':
            return SVC(**params, probability=True, random_state=42)
        elif name == 'regresion_logistic':
            return LogisticRegression(**params, random_state=42, solver='liblinear')
        else:
            # Los modelos Keras se construyen dinámicamente en la función objetivo
            return None

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
        objective_func = lambda trial: self._objective(trial, X_train, y_train, X_val, y_val, class_weights)
        study.optimize(objective_func, n_trials=n_trials)
        best_params = study.best_params

        # Re-entrenar el modelo con los mejores hiperparámetros.
        model_name_lower = self.model_name.lower()
        if model_name_lower == "lightgbm":
            best_params.update({"objective": "multiclass", "num_class": 3, 
                                "metric": "multi_logloss", "verbosity": -1, "seed": 42})
            lgb_train = lgb.Dataset(X_train, label=y_train, 
                                    weight=[class_weights.get(c, 1.0) for c in y_train])
            lgb_val = lgb.Dataset(X_val, label=y_val, 
                                  weight=[class_weights.get(c, 1.0) for c in y_val], reference=lgb_train)
            model = lgb.train(best_params, lgb_train, num_boost_round=1000,
                              valid_sets=[lgb_val], valid_names=["val"],
                              callbacks=[lgb.early_stopping(50, verbose=False)])
        elif model_name_lower in ['multiclass_neural_network', 'lstm_rnn']:
            # Para Keras, el modelo ya fue entrenado con los mejores parámetros dentro del estudio de Optuna.
            # Reconstruimos y re-entrenamos en el conjunto completo de entrenamiento/validación.
            model = self._objective(study.best_trial, X_train, y_train, None, None, class_weights, is_retrain=True)
        else:
            model = self._get_model_instance(self.model_name, best_params)
            model.fit(X_train, y_train)

        return model, best_params

    def _process_signal(self, signal, price, index, date):
        """
        Procesa una señal de trading y actualiza el estado del portafolio.
        """
        if signal == 1 and self.cash > 0:  # Señal de COMPRA
            btc_bought = self.cash / price
            self.position = btc_bought
            self.cash = 0.0
            self.num_buys_executed += 1
            # Iniciar el registro de la nueva operación
            self.current_trade = {
                'fecha_compra': date,
                'indice_compra': index,
                'precio_compra': price,
                'cantidad_BTC_comprado': btc_bought
            }
        elif signal == -1 and self.position > 0:  # Señal de VENTA
            btc_sold = self.position
            self.cash = self.position * price
            self.position = 0.0
            self.num_sells_executed += 1
            # Completar y guardar la operación
            if self.current_trade:
                self.current_trade.update({
                    'fecha_venta': date,
                    'indice_venta': index,
                    'precio_venta': price,
                    'cantidad_BTC_vendido': btc_sold
                })
                self.trades.append(self.current_trade)
                self.current_trade = {} # Resetear para la próxima operación
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

    def _objective(self, trial, X_train, y_train, X_val, y_val, class_weights, is_retrain=False):
        """Función objetivo genérica para modelos Scikit-learn y XGBoost."""
        model_name_lower = self.model_name.lower()

        # --- Modelos de Redes Neuronales (Keras/TensorFlow) ---        
        if model_name_lower == 'lightgbm':
            params = {
                "objective": "multiclass",
                "num_class": 3,
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True), # 0.01, 0.2
                "num_leaves": trial.suggest_int("num_leaves", 20, 100), # 20, 100
                "max_depth": trial.suggest_int("max_depth", 4, 12), # 4, 12
                "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0), # 0.6, 1.0
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0), # 0.6, 1.0
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 7), # 1, 7
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 100), # 5, 100
                "metric": ["multi_logloss"],
                "verbosity": -1,
                "seed": 42
            }
            lgb_train = lgb.Dataset(X_train, label=y_train, weight=[class_weights.get(c, 1.0) for c in y_train])
            lgb_val = lgb.Dataset(X_val, label=y_val, weight=[class_weights.get(c, 1.0) for c in y_val], reference=lgb_train)
            model = lgb.train(params, lgb_train, num_boost_round=1000,
                              valid_sets=[lgb_val], valid_names=["val"],
                              callbacks=[lgb.early_stopping(50, verbose=False)])
            y_val_prob = model.predict(X_val, num_iteration=model.best_iteration)
            y_val_pred = np.argmax(y_val_prob, axis=1)
            return trend_changes_true(y_val, y_val_pred)
        elif model_name_lower == 'random_forest':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 5, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            }    
        elif model_name_lower == 'gradientboosting':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
            }
        elif model_name_lower == 'histgradientboosting':
            params = {
                'max_iter': trial.suggest_int('max_iter', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
            }
        elif model_name_lower == 'xgbclassifier':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            }
        elif model_name_lower == 'svm':
            params = {
                'C': trial.suggest_float('C', 1e-2, 1e2, log=True),
                'gamma': trial.suggest_float('gamma', 1e-4, 1e-1, log=True),
                'kernel': trial.suggest_categorical('kernel', ['rbf']),
            }
        elif model_name_lower == 'regresion_logistic':
            params = {
                'C': trial.suggest_float('C', 1e-2, 1e2, log=True),
                'penalty': trial.suggest_categorical('penalty', ['l1', 'l2']),
            }
        elif model_name_lower in ['multiclass_neural_network', 'lstm_rnn']:
            # Definir hiperparámetros comunes para redes neuronales
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.2, 0.5),
                'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128])
            }
            # Reshape para LSTM si es necesario
            X_train_nn = X_train
            X_val_nn = X_val
            if model_name_lower == 'lstm_rnn':
                X_train_nn = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
                if not is_retrain:
                    X_val_nn = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
            # Construcción del modelo
            model = Sequential()
            if model_name_lower == 'lstm_rnn':
                units = trial.suggest_int('units', 32, 128)
                model.add(Input(shape=(X_train_nn.shape[1], X_train_nn.shape[2])))
                model.add(LSTM(units))
            else: # multiclass_neural_network
                n_layers = trial.suggest_int('n_layers', 1, 3)
                model.add(Input(shape=(X_train_nn.shape[1],)))
                for i in range(n_layers):
                    units = trial.suggest_int(f'n_units_l{i}', 32, 128)
                    model.add(Dense(units, activation='relu'))
            model.add(Dropout(params['dropout_rate']))
            model.add(Dense(3, activation='softmax')) # 3 clases de salida

            model.compile(optimizer=Adam(learning_rate=params['learning_rate']),
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            # Si es re-entrenamiento final, no hay set de validación
            if is_retrain:
                model.fit(X_train_nn, y_train, batch_size=params['batch_size'], epochs=100, verbose=0)
                return model
            # Entrenamiento y evaluación normal dentro de Optuna
            model.fit(X_train_nn, y_train, validation_data=(X_val_nn, y_val),
                      batch_size=params['batch_size'], epochs=100,
                      callbacks=[early_stopping], verbose=0)
            y_val_prob = model.predict(X_val_nn)
            y_val_pred = np.argmax(y_val_prob, axis=1)
            return trend_changes_true(y_val, y_val_pred)
        else:
            raise ValueError(f"Modelo '{self.model_name}' no soportado para optimización.")
        model = self._get_model_instance(self.model_name, params)
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        return trend_changes_true(y_val, y_val_pred)

    def _prediction(self, model_name_lower, X_pred):
        """
        Realiza una predicción con el modelo dado.
        """
        if "lightgbm" in model_name_lower:
            y_prob = self.model.predict(X_pred, num_iteration=self.model.best_iteration)
        elif model_name_lower in ['multiclass_neural_network', 'lstm_rnn']:
            X_pred_nn = X_pred
            if model_name_lower == 'lstm_rnn':
                X_pred_nn = X_pred.reshape((X_pred.shape[0], 1, X_pred.shape[1]))
            y_prob = self.model.predict(X_pred_nn)
        else:
            y_prob = self.model.predict_proba(X_pred)
        y_pred_mapped = np.argmax(y_prob, axis=1)
        self.pending_signal = self.inv_map[y_pred_mapped[0]]
        
        return self.pending_signal

    def _finalize_backtest(self, backtest_df):
        """
        Finaliza el backtest: liquida posiciones abiertas, crea el DataFrame de trades
        y ensambla el DataFrame final de resultados.
        """
        # --- LIQUIDACIÓN FINAL ---
        # Al final del último día, si todavía hay una posición abierta, se vende.
        if self.position > 0:
            final_price = backtest_df.loc[self.n_test - 1, self.price_col]
            final_date = backtest_df.loc[self.n_test - 1, 'date']
            print(f"\nLiquidando posición final de {self.position:.6f} BTC al precio de ${final_price:,.2f} en {final_date.date()}")
            
            # Actualizar el estado del portafolio
            self.cash = self.position * final_price
            btc_sold = self.position
            self.position = 0.0
            self.num_sells_executed += 1
            # Actualizar el valor del portafolio para el último día
            self.portfolio_values[-1] = self.cash

            # Completar el último trade si estaba abierto
            if self.current_trade and 'fecha_compra' in self.current_trade:
                self.current_trade.update({
                    'fecha_venta': final_date,
                    'indice_venta': self.n_test - 1,
                    'precio_venta': final_price,
                    'cantidad_BTC_vendido': btc_sold
                })
                self.trades.append(self.current_trade)
                self.current_trade = {}

        # --- CREACIÓN DE DATAFRAMES FINALES ---
        # Crear el DataFrame de trades
        trades_df = pd.DataFrame(self.trades)
        if not trades_df.empty:
            trades_df['valor_portafolio_en_venta'] = trades_df['cantidad_BTC_vendido'] * trades_df['precio_venta']

        # Ensamblar el DataFrame de resultados del backtest
        backtest_df['walk_forward_portfolio'] = self.portfolio_values
        backtest_df['buy_and_hold_portfolio'] = self.b_and_h_portfolio_values
        backtest_df['dca_portfolio'] = self.dca_portfolio_values

        return backtest_df, trades_df
    
    def run(self):
        """
        Ejecuta el proceso completo de backtesting con Walk-Forward Optimization.
        """
        # 1. Inicializar o resetear el estado del backtest.
        self._initialize_state()

        # Preparar el DataFrame para almacenar los resultados del backtest.
        backtest_df = self.test_set.copy()
        backtest_df['signal'] = 0            # decisión tomada hoy para ejecutar mañana

        for i in tqdm(range(0, self.n_test), desc="Walk-Forward Backtesting"):
            # 0) Ejecutar hoy lo decidido ayer (evita look-ahead)
            price = backtest_df.loc[i, self.price_col]
            date = backtest_df.loc[i, 'date']
            self._process_signal(self.pending_signal, price, i, date) # Check

            # 1) Estrategias de referencia con el precio de hoy
            self._update_reference_strategies(i, price) # Check

            # 2) Actualizar el train con el ejemplo i-1 (su target ya es conocido)
            if i > 0:
                self.X_train_current, self.y_train_current_m = self._data_update(
                    self.X_train_current, self.y_train_current_m,
                    self.X_test[i-1:i], self.y_test_m[i-1:i]
                ) # Check

            # 3) Re-optimizar cuando toque (con el train ya actualizado)
            if i % self.window_size == 0:
                # Determinar el número de trials para esta optimización
                n_trials = self.optuna_trials_initial if i == 0 else self.optuna_trials 
                self.model, best_params = self._tuning_model(
                    self.X_train_current, self.y_train_current_m,
                    n_trials, self.best_params_from_previous_step
                )
                self.best_params_from_previous_step = best_params.copy()

            # 4) Decidir la señal de mañana con los datos de hoy (no ejecutar hoy)
            X_pred = self.X_test[i:i+1]

            # Predecir probabilidades según el tipo de modelo
            model_name_lower = self.model_name.lower()    
            backtest_df.loc[i, 'signal'] = self._prediction(model_name_lower, X_pred)

        # --- FINALIZACIÓN Y DEVOLUCIÓN DE RESULTADOS ---
        backtest_df, trades_df = self._finalize_backtest(backtest_df)

        return (backtest_df, trades_df)