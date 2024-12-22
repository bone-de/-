import yaml

class Config:
    def __init__(self, config_path="config.yml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
            
        self.github_link = self.config.get("github_link", "")
        self.default_assistants = self.config.get("default_assistants", {})
        self.api_settings = self.config.get("api_settings", {})
