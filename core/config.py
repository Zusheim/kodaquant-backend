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

    # FIX DEFINITIVO: dominio kodaquant.site verificado en Resend, sandbox
    # desactivado. Se abandona SMTP directo por completo (bloqueado a nivel
    # de firewall en HF Spaces -- ver historial) a favor del SDK oficial
    # `resend` (ver core/security.py). Las variables SMTP_HOST/PORT/
    # USE_SSL/USER/PASSWORD de las iteraciones anteriores ya no existen acá
    # -- no se usan en ningún lado del código.
    #
    # Techo duro (segundos) para la llamada del SDK de Resend, corrida en
    # thread vía asyncio.to_thread (ver core/security.py::_send_mail_message).
    # El envío corre como FastAPI BackgroundTask -- no bloquea la respuesta
    # al cliente -- así que 15s da margen real sin arriesgar tareas colgadas
    # indefinidamente.
    SMTP_TIMEOUT_SECONDS: float = 15.0
    # API key de Resend (https://resend.com/api-keys). Obligatoria para que
    # el envío de correo funcione -- sin ella, se loguea un warning y el
    # correo simplemente no sale (no rompe el registro/login).
    RESEND_API_KEY: str | None = None
    # Remitente verificado. Dominio kodaquant.site ya validado por DNS en
    # Resend -- puede enviar a cualquier destinatario, no solo al sandbox.
    SMTP_FROM: str = "KodaQuant <no-reply@kodaquant.site>"
    # Si un usuario responde un correo automático, llega acá en vez de a
    # no-reply@kodaquant.site (que nadie lee).
    REPLY_TO_EMAIL: str = "karim.egure@gmail.com"
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