import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from tqdm import tqdm
from sklearn.metrics import classification_report
from collections import Counter

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
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, test_set, window_size=5, initial_capital=10000.0, optuna_trials_initial=150, optuna_trials=50):
        self.X_train_orig = X_train
        self.y_train_orig = y_train
        self.X_val_orig = X_val
        self.y_val_orig = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.test_set = test_set
        self.window_size = window_size
        self.initial_capital = initial_capital
        self.optuna_trials_initial = optuna_trials_initial
        self.optuna_trials = optuna_trials
        self.cls_map = {-1: 0, 0: 1, 1: 2}
        self.inv_map = {v: k for k, v in self.cls_map.items()}
        self.y_train_m_orig = np.vectorize(self.cls_map.get)(self.y_train_orig)
        self.y_val_m_orig = np.vectorize(self.cls_map.get)(self.y_val_orig)
        self.y_test_m = np.vectorize(self.cls_map.get)(self.y_test)

    def _objective(self, trial, X_train, y_train, X_val, y_val, class_weights):
        param = {
            "objective": "multiclass", "num_class": 3, "metric": "multi_logloss",
            "verbosity": -1, "seed": 1234,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        }
        lgb_train = lgb.Dataset(X_train, label=y_train, weight=[class_weights.get(c, 1.0) for c in y_train])
        lgb_val = lgb.Dataset(X_val, label=y_val, weight=[class_weights.get(c, 1.0) for c in y_val], reference=lgb_train)
        
        model = lgb.train(param, lgb_train, num_boost_round=1000,
                          valid_sets=[lgb_val], valid_names=["val"],
                          callbacks=[lgb.early_stopping(50, verbose=False)])

        y_val_prob = model.predict(X_val, num_iteration=model.best_iteration)
        y_val_pred = np.argmax(y_val_prob, axis=1)
        return trend_changes_true(y_val, y_val_pred)

    def lightgbm(self):
        X_train_current, y_train_current_m = self.X_train_orig.copy(), self.y_train_m_orig.copy()
        X_val_current, y_val_current_m = self.X_val_orig.copy(), self.y_val_m_orig.copy()
        
        backtest_df = self.test_set.copy()
        backtest_df['signal'] = 0
        price_col = [col for col in self.test_set.columns if 'open_d0' in col][-1]
        cash, position, portfolio_values = self.initial_capital, 0.0, []
        n_test = self.X_test.shape[0]
        # --- NUEVO: Inicializar contadores de transacciones ---
        num_buys_executed = 0
        num_sells_executed = 0
        # Almacenar los mejores parámetros del paso anterior
        best_params_from_previous_step = None
        for start in tqdm(range(0, n_test, self.window_size), desc="Walk-Forward Backtesting"):
            cnt = Counter(y_train_current_m)
            total = len(y_train_current_m)
            class_weights = {c: total / (len(cnt) * n) for c, n in cnt.items()}
            if start == 0:
                n_trials_for_this_step = self.optuna_trials_initial
            else:
                n_trials_for_this_step = self.optuna_trials
            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
            if best_params_from_previous_step:
                study.enqueue_trial(best_params_from_previous_step)

            study.optimize(lambda trial: self._objective(trial, X_train_current, y_train_current_m, X_val_current, y_val_current_m, class_weights), 
                           n_trials=n_trials_for_this_step)
            best_params = study.best_params
            # --- NUEVO: Guardar los mejores parámetros para el siguiente paso ---
            best_params_from_previous_step = best_params.copy()
            best_params.update({"objective": "multiclass", "num_class": 3, "metric": "multi_logloss", "verbosity": -1, "seed": 1234})

            lgb_train = lgb.Dataset(X_train_current, label=y_train_current_m, weight=[class_weights.get(c, 1.0) for c in y_train_current_m])
            lgb_val = lgb.Dataset(X_val_current, label=y_val_current_m, weight=[class_weights.get(c, 1.0) for c in y_val_current_m], reference=lgb_train)
            model = lgb.train(best_params, lgb_train, num_boost_round=1000,
                              valid_sets=[lgb_val], valid_names=["val"],
                              callbacks=[lgb.early_stopping(50, verbose=False)])

            end = min(start + self.window_size, n_test)
            X_pred_block = self.X_test[start:end]
            y_prob_block = model.predict(X_pred_block, num_iteration=model.best_iteration)
            y_pred_mapped = np.argmax(y_prob_block, axis=1)
            y_pred_signals = np.vectorize(self.inv_map.get)(y_pred_mapped)
            backtest_df.loc[start:end-1, 'signal'] = y_pred_signals
            
            for i in range(start, end):
                if i not in backtest_df.index: continue
                price, signal = backtest_df.loc[i, price_col], backtest_df.loc[i, 'signal']
                if signal == 1 and cash > 0:
                    position, cash = cash / price, 0.0
                    num_buys_executed += 1  # Contar compra ejecutada
                elif signal == -1 and position > 0:
                    cash, position = position * price, 0.0
                    num_sells_executed += 1  # Contar venta ejecutada
                portfolio_values.append(cash + position * price)

            if portfolio_values:
                current_portfolio_value = portfolio_values[-1]
                print(f"  -> Fin de la ventana (días {start}-{end-1}). Valor del Portafolio: ${current_portfolio_value:,.2f}")
                print(f"  -> Compras ejecutadas: {num_buys_executed}")
                print(f"  -> Ventas ejecutadas: {num_sells_executed}")

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

        backtest_df = backtest_df.iloc[:len(portfolio_values)]
        backtest_df['portfolio_value'] = portfolio_values
        return backtest_df, price_col, num_buys_executed, num_sells_executed