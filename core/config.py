# core/config.py

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from functools import lru_cache


class Settings(BaseSettings):
    """
    Configuración central de KodaQuant Terminal.
    Lee variables de entorno desde el archivo .env en la raíz del proyecto.
    """

    # --- Supabase ---
    SUPABASE_URL: str
    SUPABASE_KEY: str
    SUPABASE_SERVICE_ROLE_KEY: str
    SUPABASE_JWT_SECRET: str | None = None

    # --- Stripe ---
    STRIPE_SECRET_KEY: str
    STRIPE_PRICE_PRO: str
    STRIPE_PRICE_ULTRA: str
    STRIPE_WEBHOOK_SECRET: str
    STRIPE_SUCCESS_URL: str
    STRIPE_CANCEL_URL: str

    # --- Seguridad interna (firma de tokens JWT/HMAC) ---
    API_SECRET_KEY: str

    # --- Rate limiting anti fuerza bruta (auth) ---
    RATE_LIMIT_MAX_ATTEMPTS: int = 5
    RATE_LIMIT_WINDOW_SECONDS: int = 60

    # --- Configuración general opcional ---
    APP_NAME: str = "KodaQuant Terminal"
    ENV: str = "development"
    DEBUG: bool = False

    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: str
    SMTP_PASSWORD: str
    SMTP_FROM: str
    FRONTEND_URL: str = "https://kodaquant.web.app"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # Defensa contra espacios accidentales al pegar valores en .env o en el
    # panel de variables de entorno del hosting (ej. "FRONTEND_URL= https://...").
    # Un espacio colado acá rompe el matching exacto de origen en CORS
    # (Starlette compara el header Origin contra el string tal cual) y
    # generaría URLs rotas en los links de los correos de verificación.
    @field_validator("FRONTEND_URL", "SUPABASE_URL", mode="before")
    @classmethod
    def _strip_whitespace(cls, v: str) -> str:
        return v.strip() if isinstance(v, str) else v


@lru_cache
def get_settings() -> Settings:
    """
    Devuelve una instancia cacheada de Settings para evitar
    releer el .env en cada request.
    """
    return Settings()


settings = get_settings()