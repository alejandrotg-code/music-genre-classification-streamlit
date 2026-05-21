import os

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np


class ModelHandler:
    def __init__(self, model_path=None):
        self.model = None
        self.genres = sorted([
            "blues", "classical", "country", "disco", "hiphop",
            "jazz", "lofi", "metal", "pop", "reggae",
            "reggaeton", "rock", "trap"
        ])

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def load_model(self, model_path):
        try:
            self.model = self._load_model_compat(model_path)
            print(f"Model loaded from {model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
            self.model = None

    def _load_model_compat(self, model_path):
        try:
            import keras
            return keras.models.load_model(model_path, compile=False)
        except Exception as keras_error:
            print(f"Could not load model with keras: {keras_error}")

        import tensorflow as tf
        return tf.keras.models.load_model(model_path, compile=False)

    def is_loaded(self):
        return self.model is not None

    def predict(self, features):
        if self.model is None:
            return "Model not loaded", 0.0

        try:
            predictions = self.model.predict(features, verbose=0)
            predicted_index = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_index])
            all_scores = {
                self.genres[i]: float(predictions[0][i])
                for i in range(len(self.genres))
            }
            genre = self.genres[predicted_index] if predicted_index < len(self.genres) else "Unknown"
            return genre, confidence, all_scores
        except Exception as e:
            print(f"Prediction error: {e}")
            return "Error", 0.0, {}
