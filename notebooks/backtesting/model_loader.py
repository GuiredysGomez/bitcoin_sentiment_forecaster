import os

class ModelLoader:
    def __init__(self, model_name, model_file):
        self.model_name = model_name.lower()
        self.model_file = model_file

    def load(self):
        if not os.path.exists(self.model_file):
            print(f"No se encontró archivo para el modelo: {self.model_file}")
            return None

        if self.model_name == 'lightgbm':
            import lightgbm as lgb
            return lgb.Booster(model_file=self.model_file)
        elif self.model_name in [
            'random_forest',
            'hist_gradient_boosting',
            'gradient_boosting',
            'naive_model',
            'regresion_logistic',
            'svm'
        ]:
            import joblib
            return joblib.load(self.model_file)
        elif self.model_name == 'xgboost':
            import xgboost as xgb
            model = xgb.Booster()
            model.load_model(self.model_file)
            return model
        elif self.model_name in [
            'multiclass_neural_network',
            'lstm'
        ]:
            from tensorflow import keras
            return keras.models.load_model(self.model_file)
        else:
            print(f"Tipo de modelo '{self.model_name}' no soportado para carga automática.")
            return None