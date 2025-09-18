import os

from xgboost import XGBClassifier

class ModelLoader:
    def __init__(self, model_name, model_file):
        # normaliza nombre
        self.model_name = (model_name or "").lower().replace("-", "_").replace(" ", "_")
        self.model_file = model_file

    def load(self):
        if not os.path.exists(self.model_file):
            print(f"No se encontró archivo para el modelo: {self.model_file}")
            return None
        # 1) LightGBM (Booster)
        if self.model_name == 'lightgbm':
            import lightgbm as lgb
            return lgb.Booster(model_file=self.model_file)
        # 2) Scikit-learn (joblib.dump)
        elif self.model_name in [
            'random_forest',
            'histgradientboosting',
            'gradientboosting',
            'naive_model',
            'regresion_logistic',
            'svm'
        ]:
            import joblib
            return joblib.load(self.model_file)
        # 3) XGBoost guardado como XGBClassifier.json (o formatos nativos)
        elif self.model_name == 'xgbclassifier':
            from xgboost import XGBClassifier
            model = XGBClassifier()
            model.load_model(self.model_file)
            return model
        # 4) Keras (.keras/.h5)
        elif self.model_name in [
            'multiclass_neural_network',
            'lstm_rnn'
        ]:
            from tensorflow import keras
            return keras.models.load_model(self.model_file)
        else:
            print(f"Tipo de modelo '{self.model_name}' no soportado para carga automática.")
            return None