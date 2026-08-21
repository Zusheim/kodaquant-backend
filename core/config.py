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

    # FIX DEFINITIVO (root cause real, confirmado con traceback): Hugging
    # Face Spaces bloquea a nivel de firewall de red TODO el egress saliente
    # a puertos SMTP (25/465/587) -- ni 587/STARTTLS ni 465/SSL-implícito
    # llegan siquiera a abrir la conexión TCP (el error saltaba en
    # `_create_connection`, antes de cualquier byte de protocolo). Solo 443
    # sale garantizado en este entorno. El envío de correo se migró de SMTP
    # directo a la API HTTPS de Resend (ver core/security.py). Las variables
    # SMTP_* de abajo ya NO se usan para enviar correo -- se dejan opcionales
    # por si en el futuro corrés esto en un hosting que sí permita SMTP
    # saliente. Lo único obligatorio ahora es RESEND_API_KEY.
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 465
    SMTP_USE_SSL: bool = True
    SMTP_USER: str | None = None
    SMTP_PASSWORD: str | None = None
    # Techo duro (segundos) para la llamada HTTPS a Resend. El envío corre
    # como FastAPI BackgroundTask (ver api/auth.py) -- no bloquea la
    # respuesta al cliente -- así que 15s da margen real sin arriesgar
    # tareas colgadas indefinidamente. Ver core/security.py::_send_mail_message.
    SMTP_TIMEOUT_SECONDS: float = 15.0
    # API key de Resend (https://resend.com/api-keys). Obligatoria para que
    # el envío de correo funcione en este hosting -- sin ella, se loguea un
    # warning y el correo simplemente no sale (no rompe el registro/login).
    RESEND_API_KEY: str | None = None
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