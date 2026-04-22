import os

class Settings:
    PROJECT_NAME: str = "Gabeo RCM Analysis Dashboard"
    PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATABASE_PATH: str = os.path.join(PROJECT_ROOT, "data", "claims.db")
    STATIC_DIR: str = os.path.join(PROJECT_ROOT, "app", "static")
    TEMPLATE_DIR: str = os.path.join(PROJECT_ROOT, "app", "templates")

settings = Settings()
