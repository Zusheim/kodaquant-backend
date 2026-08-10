# core/security.py
"""
Autenticación de producción para KodaQuant Terminal.
Valida el JWT de Supabase delegando la verificación criptográfica al propio
backend de Supabase Auth (GoTrue) vía `supabase.auth.get_user()`. No existe
bypass de desarrollo.
"""
import secrets
import aiosmtplib
from email.message import EmailMessage
from core.config import settings
from datetime import datetime, timedelta, timezone
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from core.supabase_client import supabase, supabase_admin

# --- CORS: lista cerrada de orígenes de producción ---
# NUNCA usar "*" aquí: Starlette lo rechaza en runtime cuando
# allow_credentials=True, y aunque no lo hiciera, "*" + credentials abre la
# puerta a CSRF cross-origin contra cualquier endpoint autenticado
# (/api/v1/quanti/consult, /api/v1/auth/*, payments). Solo se listan
# orígenes explícitos y verificados.
PRODUCTION_ORIGINS = {
    "https://kodaquant.web.app",         # Firebase Hosting (dominio primario)
    "https://kodaquant.firebaseapp.com", # Firebase Hosting (dominio secundario, mismo proyecto)
}

# Puertos reales del dev server (Vite), tomados de la config actual del
# proyecto -- NO 3000 (eso es el puerto default de Create React App / Next,
# que este repo no usa). Agregar un origen que nadie sirve no protege nada
# y agregar uno que falta rompe el login en local.
_DEV_ORIGINS = {
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:5174", "http://127.0.0.1:5174",
}

# Métodos y headers reales que usan los endpoints (fetch/EventSource desde
# App.jsx + Authorization Bearer del login Supabase). "*" en methods/headers
# es innecesario aquí y amplía la superficie de ataque sin motivo.
ALLOWED_METHODS = ["GET", "POST", "OPTIONS"]
ALLOWED_HEADERS = ["Authorization", "Content-Type"]


def get_allowed_origins() -> list[str]:
    """
    Orígenes permitidos por CORS.

    - Producción (settings.ENV == "production"): SOLO los dominios de
      Firebase Hosting + settings.FRONTEND_URL. Los orígenes de localhost
      jamás se incluyen en este modo.
    - Cualquier otro valor de settings.ENV (dev/staging): se añaden también
      los orígenes del dev server local.

    IMPORTANTE: `ENV` en core/config.py tiene default "development". Si el
    deploy de producción no exporta explícitamente ENV=production en su
    entorno, este backend seguirá aceptando peticiones con credenciales
    desde localhost. Confirmar esa variable en la config del servidor
    (Railway/Render/Fly/EC2/lo que uses) -- esto NO es opcional para que el
    lockdown de CORS sea real.
    """
    origins = set(PRODUCTION_ORIGINS)
    if getattr(settings, "FRONTEND_URL", None):
        origins.add(settings.FRONTEND_URL.strip())

    if settings.ENV.strip().lower() != "production":
        origins |= _DEV_ORIGINS

    return sorted(origins)


# ---------------------------------------------------------------------------
# FIX CRÍTICO — CORS bypass real en producción vía gradio.Server
# ---------------------------------------------------------------------------
# ROOT CAUSE (confirmado leyendo gradio/route_utils.py::CustomCORSMiddleware
# en gradio==6.22.0 y reproducido en vivo, no es una hipótesis):
#
#   Server.launch() -> Blocks.launch() -> App.create_app() SIEMPRE ejecuta
#       app.add_middleware(CustomCORSMiddleware, strict_cors=strict_cors)
#
#   ...DESPUÉS de que main.py ya agregó nuestro CORSMiddleware. En Starlette,
#   add_middleware() inserta al frente de la pila (`user_middleware.insert(0,
#   ...)`), así que el middleware de Gradio, agregado más tarde (en
#   .launch()), termina SIEMPRE por FUERA del nuestro -- procesa cada
#   request antes que el nuestro y puede reescribir la respuesta después.
#
#   Su lógica interna (is_valid_origin) SOLO restringe origenes cuando el
#   Host de la request es localhost/127.0.0.1/0.0.0.0 (para proteger demos
#   corriendo en la laptop del dev). En cualquier deploy real -- este caso,
#   *.hf.space -- esa condición nunca aplica, así que Gradio termina
#   agregando `Access-Control-Allow-Origin: <origin que sea>` +
#   `Access-Control-Allow-Credentials: true` a CUALQUIER origen, y además
#   contesta TODO el preflight (OPTIONS) él mismo -- nuestro CORSMiddleware
#   (con su ALLOWED_METHODS/ALLOWED_HEADERS restringido) nunca llega a
#   correr para el preflight. `PRODUCTION_ORIGINS` de arriba queda
#   decorativo en producción. Verificado en vivo: una request con
#   `Origin: https://evil-attacker.com` contra un server con Host no-local
#   recibe `Access-Control-Allow-Origin: https://evil-attacker.com` de
#   vuelta.
#
#   No hay manera de "ganarle" la posición a Gradio en la pila de
#   middleware -- SIEMPRE se agrega después, en .launch(), sin importar el
#   orden en main.py. La única defensa real es no depender de qué headers
#   CORS termine poniendo esa capa, y en cambio RECHAZAR nosotros mismos,
#   a nivel aplicación, cualquier request de un Origin no reconocido --
#   esto SÍ es efectivo pase lo que pase después en la respuesta, porque el
#   request nunca llega al router si lo cortamos acá.
#
#   Nota de alcance: como este backend NO usa cookies para auth (el JWT de
#   Supabase viaja en el header Authorization, nunca en una cookie), esto
#   no es un CSRF clásico explotable por un sitio malicioso ajeno -- pero
#   sí anula la protección que el equipo cree tener, y cualquier
#   dependencia futura en cookies o en "Origin como control de acceso"
#   quedaría rota. Corregir de todos modos.
class StrictOriginMiddleware(BaseHTTPMiddleware):
    """
    Rechaza (403) cualquier request con un header Origin presente que NO
    esté en la lista cerrada de `get_allowed_origins()`. Se evalúa en cada
    request (no cachea la lista), así que respeta el mismo ENV en runtime
    que ve `get_allowed_origins()`.

    Requests SIN header Origin (curl, llamadas server-to-server, health
    checks del propio hosting) pasan de largo -- Origin solo lo manda un
    navegador haciendo fetch/XHR cross-origin.
    """

    async def dispatch(self, request: Request, call_next):
        origin = request.headers.get("origin")
        if origin is not None and origin not in set(get_allowed_origins()):
            return JSONResponse(
                {"detail": "Origen no permitido."},
                status_code=status.HTTP_403_FORBIDDEN,
            )
        return await call_next(request)


# auto_error=False es la pieza clave: si el header falta o está mal formado,
# HTTPBearer devuelve None en lugar de lanzar su propio HTTPException(403).
# Así TODO el manejo de errores de auth pasa por get_current_user y responde
# 401 de forma consistente -- nunca más un 403 "silencioso" que evita
# nuestro try/except.
security_scheme = HTTPBearer(auto_error=False)


class AuthenticatedUser(BaseModel):
    """Identidad resuelta a partir del usuario verificado por Supabase Auth."""
    id: str
    email: str | None = None
    role: str | None = None


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security_scheme),
) -> AuthenticatedUser:
    if credentials is None or not credentials.credentials:
        print("[AUTH] Falta el header Authorization o no tiene el esquema 'Bearer <token>'.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido o expirado",
        )

    # credentials.credentials YA es el JWT puro (HTTPBearer separa el esquema
    # "Bearer" del token). Este strip extra es cinturón de seguridad por si
    # el cliente concatena mal el header -- en condiciones normales es un no-op.
    raw_token = credentials.credentials.strip()
    if raw_token.lower().startswith("bearer "):
        raw_token = raw_token[7:].strip()

    try:
        response = supabase.auth.get_user(raw_token)
    except Exception as exc:
        print(f"Error Supabase: {exc}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido o expirado",
        )

    user = getattr(response, "user", None)
    if user is None or getattr(user, "id", None) is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token inválido o expirado",
        )

    return AuthenticatedUser(
        id=user.id,
        email=getattr(user, "email", None),
        role=getattr(user, "role", None),
    )


async def verify_api_credits(
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> AuthenticatedUser:
    result = (
        supabase_admin.table("profiles")
        .select("subscription_tier, api_requests_count")
        .eq("id", current_user.id)
        .single()
        .execute()
    )
    profile = result.data
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Perfil no encontrado",
        )

    tier = profile.get("subscription_tier")
    if tier == "FREE":
        current_count = profile.get("api_requests_count", 0)
        if current_count >= 3:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Límite de solicitudes alcanzado para el plan FREE",
            )
        supabase_admin.table("profiles").update(
            {"api_requests_count": current_count + 1}
        ).eq("id", current_user.id).execute()

    return current_user

# --- Verificación de email ---
def generate_verification_token() -> tuple[str, datetime]:
    """Token de verificación, válido 24h."""
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(hours=24)
    return token, expires_at


async def send_verification_email(email_to: str, token: str) -> None:
    """Envía el correo de verificación vía SMTP (Gmail) sin bloquear el event loop."""
    verify_url = f"{settings.FRONTEND_URL}/verify?token={token}"

    html = f"""\
<html>
  <body style="margin:0;padding:0;background-color:#0a0e14;font-family:'Segoe UI',Arial,sans-serif;">
    <table width="100%" cellpadding="0" cellspacing="0" style="padding:40px 0;background-color:#0a0e14;">
      <tr>
        <td align="center">
          <table width="480" cellpadding="0" cellspacing="0" style="background-color:#10151f;border:1px solid #1f2a37;border-radius:8px;overflow:hidden;">
            <tr>
              <td style="padding:32px;text-align:center;border-bottom:1px solid #1f2a37;">
                <span style="font-size:22px;font-weight:700;color:#00e5c7;letter-spacing:1px;">KODA<span style="color:#e6edf3;">QUANT</span></span>
              </td>
            </tr>
            <tr>
              <td style="padding:32px;color:#c9d1d9;">
                <p style="font-size:15px;line-height:1.6;margin:0 0 20px;">Confirma tu correo electrónico para activar tu cuenta y desbloquear tu Free Tier.</p>
                <table cellpadding="0" cellspacing="0" width="100%">
                  <tr>
                    <td align="center" style="padding:12px 0 24px;">
                      <a href="{verify_url}" style="background-color:#00e5c7;color:#0a0e14;text-decoration:none;font-weight:700;font-size:14px;padding:14px 32px;border-radius:6px;display:inline-block;">
                        VERIFICAR CUENTA
                      </a>
                    </td>
                  </tr>
                </table>
                <p style="font-size:12px;line-height:1.5;color:#6e7681;margin:0;">Si el botón no funciona, copia este enlace:<br>
                  <a href="{verify_url}" style="color:#00e5c7;word-break:break-all;">{verify_url}</a>
                </p>
                <p style="font-size:12px;color:#6e7681;margin:24px 0 0;">Este enlace expira en 24 horas.</p>
              </td>
            </tr>
          </table>
        </td>
      </tr>
    </table>
  </body>
</html>
"""

    message = EmailMessage()
    message["From"] = settings.SMTP_FROM
    message["To"] = email_to
    message["Subject"] = "Verifica tu cuenta — KodaQuant Terminal"
    message.set_content("Verifica tu cuenta visitando: " + verify_url)  # fallback texto plano
    message.add_alternative(html, subtype="html")

    await aiosmtplib.send(
        message,
        hostname=settings.SMTP_HOST,
        port=settings.SMTP_PORT,
        username=settings.SMTP_USER,
        password=settings.SMTP_PASSWORD,
        start_tls=True,
    )


# --- Recuperación de contraseña ---
def generate_password_reset_token() -> tuple[str, datetime]:
    """Token de restablecimiento de contraseña, válido 60 minutos (más corto
    que el de verificación de email porque habilita una acción sensible)."""
    token = secrets.token_urlsafe(32)
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=60)
    return token, expires_at


async def send_password_reset_email(email_to: str, token: str) -> None:
    """Envía el correo de restablecimiento de contraseña vía SMTP, sin bloquear el event loop."""
    reset_url = f"{settings.FRONTEND_URL}/update-password?token={token}"

    html = f"""\
<html>
  <body style="margin:0;padding:0;background-color:#0a0e14;font-family:'Segoe UI',Arial,sans-serif;">
    <table width="100%" cellpadding="0" cellspacing="0" style="padding:40px 0;background-color:#0a0e14;">
      <tr>
        <td align="center">
          <table width="480" cellpadding="0" cellspacing="0" style="background-color:#10151f;border:1px solid #1f2a37;border-radius:8px;overflow:hidden;">
            <tr>
              <td style="padding:32px;text-align:center;border-bottom:1px solid #1f2a37;">
                <span style="font-size:22px;font-weight:700;color:#00e5c7;letter-spacing:1px;">KODA<span style="color:#e6edf3;">QUANT</span></span>
              </td>
            </tr>
            <tr>
              <td style="padding:32px;color:#c9d1d9;">
                <p style="font-size:15px;line-height:1.6;margin:0 0 20px;">Recibimos una solicitud para restablecer la contraseña de tu cuenta. Si no fuiste tú, ignora este correo.</p>
                <table cellpadding="0" cellspacing="0" width="100%">
                  <tr>
                    <td align="center" style="padding:12px 0 24px;">
                      <a href="{reset_url}" style="background-color:#00e5c7;color:#0a0e14;text-decoration:none;font-weight:700;font-size:14px;padding:14px 32px;border-radius:6px;display:inline-block;">
                        RESTABLECER CONTRASEÑA
                      </a>
                    </td>
                  </tr>
                </table>
                <p style="font-size:12px;line-height:1.5;color:#6e7681;margin:0;">Si el botón no funciona, copia este enlace:<br>
                  <a href="{reset_url}" style="color:#00e5c7;word-break:break-all;">{reset_url}</a>
                </p>
                <p style="font-size:12px;color:#6e7681;margin:24px 0 0;">Este enlace expira en 60 minutos.</p>
              </td>
            </tr>
          </table>
        </td>
      </tr>
    </table>
  </body>
</html>
"""

    message = EmailMessage()
    message["From"] = settings.SMTP_FROM
    message["To"] = email_to
    message["Subject"] = "Restablece tu contraseña — KodaQuant Terminal"
    message.set_content("Restablece tu contraseña visitando: " + reset_url)
    message.add_alternative(html, subtype="html")

    await aiosmtplib.send(
        message,
        hostname=settings.SMTP_HOST,
        port=settings.SMTP_PORT,
        username=settings.SMTP_USER,
        password=settings.SMTP_PASSWORD,
        start_tls=True,
    )


# --- Correos transaccionales de suscripción (i18n) ---
# Disparar SIEMPRE desde el backend (webhook de Stripe / endpoint server-side
# que ya conoce el nuevo tier), nunca desde una acción de cliente sin validar.

_SUBSCRIPTION_COPY = {
    "en": {
        "upgrade": {
            "subject": "Welcome to KodaQuant {tier} — Subscription Confirmed",
            "heading": "Your upgrade is live",
            "body": "Your subscription has just been upgraded to <strong>{tier}</strong>. Here's what changed:",
            "cta": "OPEN TERMINAL",
        },
        "downgrade": {
            "subject": "Your KodaQuant plan has changed to {tier}",
            "heading": "Plan updated",
            "body": "Your subscription has been changed to <strong>{tier}</strong>, effective at the end of your current billing cycle. Here's what to expect:",
            "cta": "VIEW PLAN DETAILS",
        },
        "cancel": {
            "subject": "Your KodaQuant subscription has been canceled",
            "heading": "Subscription canceled",
            "body": "Your paid subscription has been canceled. You'll keep premium access until the end of your current billing cycle, then your account will move to <strong>Free</strong>.",
            "cta": "MANAGE ACCOUNT",
        },
        "footer": "No refunds are issued for partial billing periods.",
    },
    "es": {
        "upgrade": {
            "subject": "Bienvenido a KodaQuant {tier} — Suscripción confirmada",
            "heading": "Tu mejora ya está activa",
            "body": "Tu suscripción acaba de mejorar a <strong>{tier}</strong>. Esto es lo que cambió:",
            "cta": "ABRIR TERMINAL",
        },
        "downgrade": {
            "subject": "Tu plan de KodaQuant cambió a {tier}",
            "heading": "Plan actualizado",
            "body": "Tu suscripción cambiará a <strong>{tier}</strong> al finalizar tu ciclo de facturación actual. Esto es lo que puedes esperar:",
            "cta": "VER DETALLES DEL PLAN",
        },
        "cancel": {
            "subject": "Tu suscripción de KodaQuant fue cancelada",
            "heading": "Suscripción cancelada",
            "body": "Tu suscripción paga fue cancelada. Mantendrás el acceso premium hasta el final de tu ciclo de facturación actual, luego tu cuenta pasará a <strong>Free</strong>.",
            "cta": "GESTIONAR CUENTA",
        },
        "footer": "No se emiten reembolsos por períodos de facturación parciales.",
    },
}

_TIER_BENEFITS = {
    "en": {
        "FREE": ["1 active signal at a time", "Standard forecast horizon"],
        "PRO": ["Unlimited signal generation", "Opportunity Scanner access"],
        "ULTRA": ["Priority ONNX inference queue", "Quanti Discovery mode"],
    },
    "es": {
        "FREE": ["1 señal activa a la vez", "Horizonte de pronóstico estándar"],
        "PRO": ["Generación ilimitada de señales", "Acceso a Opportunity Scanner"],
        "ULTRA": ["Cola de inferencia ONNX prioritaria", "Modo Quanti Discovery"],
    },
}


def _render_subscription_email_html(heading: str, body: str, tier: str, language: str, cta_label: str) -> str:
    """Mismo layout visual que verify/reset (fondo #0a0e14, tarjeta #10151f,
    borde #1f2a37, logo KODAQUANT #00e5c7/#e6edf3), con comparativa de
    beneficios del tier destino inyectada como lista."""
    benefits = _TIER_BENEFITS.get(language, _TIER_BENEFITS["en"]).get(tier.upper(), [])
    benefits_html = "".join(f'<li style="padding:4px 0;">{b}</li>' for b in benefits)
    footer = _SUBSCRIPTION_COPY.get(language, _SUBSCRIPTION_COPY["en"])["footer"]
    cta_url = f"{settings.FRONTEND_URL}/dashboard"
    return f"""\
<html>
  <body style="margin:0;padding:0;background-color:#0a0e14;font-family:'Segoe UI',Arial,sans-serif;">
    <table width="100%" cellpadding="0" cellspacing="0" style="padding:40px 0;background-color:#0a0e14;">
      <tr>
        <td align="center">
          <table width="480" cellpadding="0" cellspacing="0" style="background-color:#10151f;border:1px solid #1f2a37;border-radius:8px;overflow:hidden;">
            <tr>
              <td style="padding:32px;text-align:center;border-bottom:1px solid #1f2a37;">
                <span style="font-size:22px;font-weight:700;color:#00e5c7;letter-spacing:1px;">KODA<span style="color:#e6edf3;">QUANT</span></span>
              </td>
            </tr>
            <tr>
              <td style="padding:32px;color:#c9d1d9;">
                <p style="font-size:17px;font-weight:600;color:#e6edf3;margin:0 0 12px;">{heading}</p>
                <p style="font-size:15px;line-height:1.6;margin:0 0 20px;">{body}</p>
                <ul style="font-size:13px;line-height:1.6;color:#c9d1d9;margin:0 0 24px;padding-left:18px;">
                  {benefits_html}
                </ul>
                <table cellpadding="0" cellspacing="0" width="100%">
                  <tr>
                    <td align="center" style="padding:12px 0 24px;">
                      <a href="{cta_url}" style="background-color:#00e5c7;color:#0a0e14;text-decoration:none;font-weight:700;font-size:14px;padding:14px 32px;border-radius:6px;display:inline-block;">
                        {cta_label}
                      </a>
                    </td>
                  </tr>
                </table>
                <p style="font-size:12px;color:#6e7681;margin:0;">{footer}</p>
              </td>
            </tr>
          </table>
        </td>
      </tr>
    </table>
  </body>
</html>
"""


async def send_subscription_upgrade_email(email_to: str, tier: str, language: str = "en") -> None:
    """Correo al mejorar de plan (ej. FREE -> PRO, PRO -> ULTRA)."""
    copy = _SUBSCRIPTION_COPY.get(language, _SUBSCRIPTION_COPY["en"])["upgrade"]
    subject = copy["subject"].format(tier=tier.upper())
    body = copy["body"].format(tier=tier.upper())
    html = _render_subscription_email_html(copy["heading"], body, tier, language, copy["cta"])
    plain = f"{copy['heading']}. {body}".replace("<strong>", "").replace("</strong>", "")

    message = EmailMessage()
    message["From"] = settings.SMTP_FROM
    message["To"] = email_to
    message["Subject"] = subject
    message.set_content(plain)
    message.add_alternative(html, subtype="html")

    await aiosmtplib.send(
        message,
        hostname=settings.SMTP_HOST,
        port=settings.SMTP_PORT,
        username=settings.SMTP_USER,
        password=settings.SMTP_PASSWORD,
        start_tls=True,
    )


async def send_subscription_downgrade_email(email_to: str, tier: str, language: str = "en") -> None:
    """Correo al bajar de plan (ej. ULTRA -> PRO), efectivo al fin del ciclo."""
    copy = _SUBSCRIPTION_COPY.get(language, _SUBSCRIPTION_COPY["en"])["downgrade"]
    subject = copy["subject"].format(tier=tier.upper())
    body = copy["body"].format(tier=tier.upper())
    html = _render_subscription_email_html(copy["heading"], body, tier, language, copy["cta"])
    plain = f"{copy['heading']}. {body}".replace("<strong>", "").replace("</strong>", "")

    message = EmailMessage()
    message["From"] = settings.SMTP_FROM
    message["To"] = email_to
    message["Subject"] = subject
    message.set_content(plain)
    message.add_alternative(html, subtype="html")

    await aiosmtplib.send(
        message,
        hostname=settings.SMTP_HOST,
        port=settings.SMTP_PORT,
        username=settings.SMTP_USER,
        password=settings.SMTP_PASSWORD,
        start_tls=True,
    )


async def send_subscription_cancel_email(email_to: str, language: str = "en") -> None:
    """Correo al cancelar un plan pago; el usuario vuelve a FREE al fin del ciclo."""
    copy = _SUBSCRIPTION_COPY.get(language, _SUBSCRIPTION_COPY["en"])["cancel"]
    html = _render_subscription_email_html(copy["heading"], copy["body"], "FREE", language, copy["cta"])
    plain = f"{copy['heading']}. {copy['body']}".replace("<strong>", "").replace("</strong>", "")

    message = EmailMessage()
    message["From"] = settings.SMTP_FROM
    message["To"] = email_to
    message["Subject"] = copy["subject"]
    message.set_content(plain)
    message.add_alternative(html, subtype="html")

    await aiosmtplib.send(
        message,
        hostname=settings.SMTP_HOST,
        port=settings.SMTP_PORT,
        username=settings.SMTP_USER,
        password=settings.SMTP_PASSWORD,
        start_tls=True,
    )