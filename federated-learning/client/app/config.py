import os
import json
from pathlib import Path

# File to store persistent settings
CONFIG_FILE = "config.json"

class SimpleConfig:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SimpleConfig, cls).__new__(cls)
            cls._instance.load_config()
        return cls._instance

    def load_config(self):
        self.config = {}
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    self.config = json.load(f)
            except Exception as e:
                print(f"[CONFIG] Failed to load config.json: {e}")
    
    def get_server_url(self):
        # Priority: Config File > Env Var > Default
        return self.config.get("SERVER_API_URL", os.getenv("SERVER_API_URL", "http://localhost:8081"))

    def get_fl_server_address(self):
        return self.config.get("FL_SERVER_ADDRESS", os.getenv("FL_SERVER_ADDRESS", "localhost:8085"))
    
    def save_settings(self, server_url, fl_address):
        self.config["SERVER_API_URL"] = server_url
        self.config["FL_SERVER_ADDRESS"] = fl_address
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(self.config, f, indent=4)
            print("[CONFIG] Settings saved to config.json")
            return True
        except Exception as e:
            print(f"[CONFIG] Failed to save settings: {e}")
            return False

# Global instance
config = SimpleConfig()
