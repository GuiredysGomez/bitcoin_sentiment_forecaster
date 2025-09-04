import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- Funciones de Métrica ---
# (Copiadas de LightGBM.ipynb para que la clase sea autocontenida)
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
    # Manejar caso donde 'True' no está en el reporte
    if "True" in report:
        return report["True"][SCORE]
    return 0.0

class Backtesting:
    def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test, test_set, window_size=5, initial_capital=10000.0, optuna_trials=50):
        self.X_train_full = np.vstack([X_train, X_val])
        self.y_train_full = np.concatenate([y_train, y_val])
        self.X_test = X_test
        self.y_test = y_test
        self.test_set = test_set
        self.window_size = window_size
        self.initial_capital = initial_capital
        self.optuna_trials = optuna_trials
        
        self.cls_map = {-1: 0, 0: 1, 1: 2}
        self.inv_map = {v: k for k, v in self.cls_map.items()}
        
        self.y_train_full_m = np.vectorize(self.cls_map.get)(self.y_train_full)
        self.y_test_m = np.vectorize(self.cls_map.get)(self.y_test)

    def _objective(self, trial, X_train, y_train, X_val, y_val):
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
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
        
        model = lgb.train(param, lgb_train, num_boost_round=1000,
                          valid_sets=[lgb_val], valid_names=["val"],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
        
        y_val_prob = model.predict(X_val, num_iteration=model.best_iteration)
        y_val_pred = np.argmax(y_val_prob, axis=1)
        return trend_changes_true(y_val, y_val_pred)

    def lightgbm(self):
        X_train_current = self.X_train_full.copy()
        y_train_current_m = self.y_train_full_m.copy()
        
        backtest_df = self.test_set.copy()
        backtest_df['signal'] = 0
        price_col = [col for col in self.test_set.columns if 'open_d0' in col][-1]
        
        cash = self.initial_capital
        position = 0.0
        portfolio_values = []
        
        n_test = self.X_test.shape[0]
        
        for start in tqdm(range(0, n_test, self.window_size), desc="Backtesting con Optuna"):
            # --- 1. Optimización de Hiperparámetros con Optuna ---
            # Dividir el set de entrenamiento actual para la validación de Optuna
            X_t, X_v, y_t, y_v = train_test_split(X_train_current, y_train_current_m, test_size=0.2, random_state=42)
            
            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(lambda trial: self._objective(trial, X_t, y_t, X_v, y_v), n_trials=self.optuna_trials)
            
            best_params = study.best_params
            best_params.update({"objective": "multiclass", "num_class": 3, "metric": "multi_logloss", "verbosity": -1})

            # --- 2. Reentrenamiento con los mejores hiperparámetros ---
            lgb_train_current = lgb.Dataset(X_train_current, label=y_train_current_m)
            model = lgb.train(best_params, lgb_train_current)
            
            # --- 3. Predicción y Simulación de Trading ---
            end = min(start + self.window_size, n_test)
            X_pred_block = self.X_test[start:end]
            y_prob_block = model.predict(X_pred_block)
            y_pred_mapped = np.argmax(y_prob_block, axis=1)
            y_pred_signals = np.vectorize(self.inv_map.get)(y_pred_mapped)
            
            backtest_df.loc[start:end-1, 'signal'] = y_pred_signals
            
            for i in range(start, end):
                price = backtest_df.loc[i, price_col]
                signal = backtest_df.loc[i, 'signal']
                if signal == 1 and cash > 0:
                    position = cash / price
                    cash = 0.0
                elif signal == -1 and position > 0:
                    cash = position * price
                    position = 0.0
                current_portfolio_value = cash + position * price
                portfolio_values.append(current_portfolio_value)

            # --- 4. Actualización de la Ventana Deslizante ---
            if end < n_test:
                X_train_current = np.vstack([X_train_current[self.window_size:], self.X_test[start:end]])
                y_train_current_m = np.concatenate([y_train_current_m[self.window_size:], self.y_test_m[start:end]])

        backtest_df['portfolio_value'] = portfolio_values
        return backtest_df, price_col