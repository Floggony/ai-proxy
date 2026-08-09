"""Application configuration."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Local LLM (llama.cpp)
    local_llm_url: str = "http://192.168.0.193:5001/v1"
    local_llm_model: str = "Ternary-Bonsai-27B-Q2_g64"

    # Cloud LLM (MiMo)
    cloud_llm_url: str = "https://token-plan-sgp.xiaomimimo.com/v1"
    cloud_llm_api_key: str = ""
    cloud_llm_model: str = "mimo-v2.5"

    # Server
    host: str = "0.0.0.0"
    port: int = 8080

    # Routing
    classification_prompt: str = """Оцени сложность запроса. Ответь ОДНОМ словом: "local" или "cloud".

Правила:
- local: простые вопросы, факты, перевод, форматирование, короткие ответы
- cloud: код, анализ, дизайн, сравнение, многошаговые задачи, сложные объяснения

Запрос: "{query}"
Ответ:"""

    model_config = {"env_file": ".env"}


settings = Settings()
