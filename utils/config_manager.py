import json
import os


class ConfigManager:
    DEFAULT_CONFIG = {
        "confidence_threshold": 0.5,
        "model_path": "models/model.h5",
    }

    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self.config = self._load()

    def _load(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    return {**self.DEFAULT_CONFIG, **json.load(f)}
            except Exception:
                pass
        return dict(self.DEFAULT_CONFIG)

    def save(self):
        try:
            with open(self.config_path, "w") as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")

    def get(self, key):
        return self.config.get(key, self.DEFAULT_CONFIG.get(key))

    def set(self, key, value):
        self.config[key] = value
        self.save()
