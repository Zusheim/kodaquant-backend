"""
api/auth.py

Módulo de autenticación para KodaQuant Terminal.
Responsable de exponer los endpoints de registro de usuarios delegando
la persistencia y validación de credenciales a Supabase Auth.

Arquitectura:
    - Cliente Supabase inicializado a partir de credenciales centralizadas en core.config.
    - Esquemas de entrada/salida validados con Pydantic.
    - Manejo explícito de errores de Supabase mapeados a HTTPException.
"""
import hashlib
from datetime import datetime, timezone
from core.security import (
    generate_verification_token,
    send_verification_email,
    generate_password_reset_token,
    send_password_reset_email,
)
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status
from pydantic import BaseModel, EmailStr, Field, field_validator
from core.supabase_client import supabase, supabase_admin
from core.rate_limiter import enforce_rate_limit


def _hash_token(token: str) -> str:
    """Hash de un solo sentido para tokens de un solo uso antes de persistirlos.
    El token crudo viaja SOLO por el link del correo — nunca en texto plano en DB."""
    return hashlib.sha256(token.encode()).hexdigest()

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
router = APIRouter(
    tags=["Authentication"],
)


# ---------------------------------------------------------------------------
# Esquemas Pydantic
# ---------------------------------------------------------------------------
class UserRegisterRequest(BaseModel):
    """Payload de entrada para el registro de un nuevo usuario."""

    email: EmailStr = Field(..., description="Correo electrónico del usuario.")
    full_name: str = Field(..., min_length=2, max_length=120, description="Nombre completo del usuario.")
    password: str = Field(
        ...,
        min_length=8,
        description="Contraseña del usuario. Mínimo 8 caracteres.",
    )
    terms_accepted: bool = Field(
        ..., description="Debe ser true: aceptación explícita de T&C y Aviso de Privacidad."
    )

    @field_validator("terms_accepted")
    @classmethod
    def validate_terms_accepted(cls, value: bool) -> bool:
        if value is not True:
            raise ValueError("Debes aceptar los Términos y Condiciones y el Aviso de Privacidad.")
        return value

    @field_validator("password")
    @classmethod
    def validate_password_strength(cls, value: str) -> str:
        # El hashing de la contraseña NO se hace acá: Supabase Auth (GoTrue)
        # la hashea con bcrypt del lado del servidor dentro de auth.sign_up().
        # Hashear manualmente antes de enviarla duplicaría el trabajo y
        # rompería la verificación posterior en sign_in_with_password.
        """Valida requisitos mínimos de robustez antes de llamar a Supabase."""
        if not any(char.isdigit() for char in value):
            raise ValueError("La contraseña debe contener al menos un número.")
        if not any(char.isupper() for char in value):
            raise ValueError("La contraseña debe contener al menos una letra mayúscula.")
        return value


class UserRegisterResponse(BaseModel):
    """Respuesta devuelta tras un registro exitoso."""

    id: str
    email: EmailStr
    message: str = "Usuario registrado exitosamente. Verifica tu correo electrónico."


# ---------------------------------------------------------------------------
# Endpoint: Registro de usuario
# ---------------------------------------------------------------------------
@router.post(
    "/register",
    response_model=UserRegisterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Registrar un nuevo usuario",
    description="Crea una nueva cuenta de usuario en Supabase Auth para KodaQuant Terminal.",
)
async def register_user(
    payload: UserRegisterRequest, request: Request, background_tasks: BackgroundTasks
) -> UserRegisterResponse:
    """
    Registra un nuevo usuario en Supabase Auth.

    Args:
        payload (UserRegisterRequest): Credenciales del usuario (email, password,
            full_name y terms_accepted).

    Returns:
        UserRegisterResponse: Datos básicos del usuario creado.

    Raises:
        HTTPException 409: Si el correo ya está registrado.
        HTTPException 422: Si la contraseña es rechazada por Supabase (débil).
        HTTPException 429: Si se excede el límite de intentos (fuerza bruta).
        HTTPException 500: Ante cualquier error inesperado del proveedor de auth
            o si falla la inicialización del perfil en `profiles`.
    """
    await enforce_rate_limit(request, "register")

    try:
        # FIX: `full_name` va anidado bajo "options.data", no como key de nivel
        # superior. supabase-py solo escribe en auth.users.raw_user_meta_data
        # cuando el metadata llega dentro de "options" -- con la forma anterior
        # ({"data": {...}} al mismo nivel que "email"/"password"), el SDK la
        # ignora silenciosamente y raw_user_meta_data queda vacío, sin error.
        result = supabase.auth.sign_up(
            {
                "email": payload.email,
                "password": payload.password,
                "options": {"data": {"full_name": payload.full_name}},
            }
        )
    except Exception as exc:
        error_message = str(exc).lower()

        if "already registered" in error_message or "already exists" in error_message:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="El correo electrónico ya se encuentra registrado.",
            ) from exc

        if "password" in error_message and (
            "weak" in error_message or "short" in error_message or "should be" in error_message
        ):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="La contraseña no cumple con los requisitos mínimos de seguridad.",
            ) from exc

        print(f"[ERROR] Fallo inesperado en registro (user={payload.email}): {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No se pudo completar el registro. Intenta nuevamente más tarde.",
        ) from exc

    user = getattr(result, "user", None)
    if user is None or getattr(user, "id", None) is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Supabase no devolvió información válida del usuario creado.",
        )

    # --- FIX: detección de correo duplicado silencioso ---
    # Cuando la confirmación por email está activada en Supabase, sign_up()
    # con un correo YA existente no lanza excepción: devuelve un "user" con
    # identities=[] vacío (medida anti-enumeración de Supabase). Sin este
    # chequeo, el backend responde 201 como si hubiera creado una cuenta nueva.
    identities = getattr(user, "identities", None)
    if identities is not None and len(identities) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Este correo ya está registrado.",
        )

    token, expires_at = generate_verification_token()

    # FIX (root cause del 500 / PGRST204): requiere que exista
    # public.profiles.full_name (ver migración SQL adjunta). Además, esta
    # escritura ahora está aislada en su propio try/except: para este punto
    # el usuario YA existe en Auth (sign_up ya se ejecutó con éxito), así que
    # un fallo acá NO debe propagarse como un AttributeError/KeyError crudo --
    # se loguea con detalle y se responde un 500 explícito y accionable, en
    # vez de dejar a la cuenta en un estado a medio inicializar sin avisar.
    try:
        supabase_admin.table("profiles").update({
            "full_name": payload.full_name,
            "is_verified": False,
            "verification_token": _hash_token(token),
            "verification_token_expires_at": expires_at.isoformat(),
        }).eq("id", user.id).execute()
    except Exception as exc:
        print(f"[ERROR] No se pudo inicializar el perfil en 'profiles' (user_id={user.id}): {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Tu cuenta se creó pero hubo un problema al configurar tu perfil. "
                "Contacta a soporte antes de intentar registrarte de nuevo."
            ),
        ) from exc

    # FIX (root cause de "el registro tarda demasiado" / SMTP timeout en
    # línea): antes esto era `await send_verification_email(...)` dentro de
    # un try/except -- el cliente HTTP esperaba a que el SMTP conectara (o
    # tronara) ANTES de recibir el 201. Con el puerto bloqueado y sin timeout
    # explícito eso eran hasta 60s de espera real por request. Ahora se
    # agenda como BackgroundTask: la respuesta 201 sale apenas termina lo de
    # arriba, y el envío corre después, con su propio techo de 3s
    # (core/security.py::_send_mail_message) que nunca vuelve a tocar este
    # request.
    background_tasks.add_task(send_verification_email, payload.email, token)

    return UserRegisterResponse(
        id=user.id,
        email=payload.email,
    )

class VerifyEmailRequest(BaseModel):
    token: str

@router.post("/verify-email")
async def verify_email(payload: VerifyEmailRequest):
    result = (
        supabase_admin.table("profiles")
        .select("id, is_verified, verification_token_expires_at")
        .eq("verification_token", _hash_token(payload.token))
        .maybe_single()
        .execute()
    )
    profile = result.data
    if not profile:
        raise HTTPException(status_code=400, detail="Token de verificación inválido.")
    if profile.get("is_verified"):
        return {"message": "El correo ya estaba verificado."}

    expires_at = profile.get("verification_token_expires_at")
    if expires_at and datetime.fromisoformat(expires_at) < datetime.now(timezone.utc):
        raise HTTPException(status_code=400, detail="El token ha expirado.")

    supabase_admin.table("profiles").update({
        "is_verified": True,
        "verification_token": None,
        "verification_token_expires_at": None,
    }).eq("id", profile["id"]).execute()

    return {"message": "Correo verificado exitosamente."}

class LoginRequest(BaseModel):
    email: str
    password: str

@router.post("/login")
async def login(payload: LoginRequest, request: Request):
    await enforce_rate_limit(request, "login")

    try:
        response = supabase.auth.sign_in_with_password({
            "email": payload.email,
            "password": payload.password
        })
    except Exception as exc:
        print(f"[ERROR] Fallo de login (user={payload.email}): {exc}")
        raise HTTPException(status_code=401, detail="Credenciales inválidas.")

    if not response or not response.session:
        raise HTTPException(status_code=401, detail="No se pudo autenticar")

    profile = (
        supabase_admin.table("profiles")
        .select("is_verified")
        .eq("id", response.user.id)
        .single()
        .execute()
    ).data
    if not profile or not profile.get("is_verified"):
        raise HTTPException(status_code=401, detail="Debes verificar tu correo antes de iniciar sesión.")

    return {
        "access_token": response.session.access_token,
        "token_type": "bearer",
        "user": response.user
    }


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


@router.post("/forgot-password")
async def forgot_password(payload: ForgotPasswordRequest, request: Request, background_tasks: BackgroundTasks):
    """
    Inicia el flujo de recuperación de contraseña.

    Respuesta SIEMPRE genérica (anti user-enumeration): nunca confirmamos
    ni negamos si el correo existe en la base, sin importar el resultado.
    """
    await enforce_rate_limit(request, "forgot-password")

    generic_response = {
        "message": "Si existe una cuenta con ese correo, te enviamos un enlace para restablecer tu contraseña."
    }

    # ⚠️ Asume que la tabla `profiles` tiene columna `email` (patrón típico de
    # trigger on-signup en Supabase). Si tu esquema no la tiene, ajusta esta
    # búsqueda a como resuelvas hoy user-por-email.
    profile_result = (
        supabase_admin.table("profiles")
        .select("id")
        .eq("email", payload.email)
        .maybe_single()
        .execute()
    )
    profile = profile_result.data
    if not profile:
        return generic_response

    token, expires_at = generate_password_reset_token()
    supabase_admin.table("profiles").update({
        "password_reset_token": _hash_token(token),
        "password_reset_token_expires_at": expires_at.isoformat(),
    }).eq("id", profile["id"]).execute()

    # Mismo fix que en /register: agendado como BackgroundTask para que la
    # respuesta genérica (anti-enumeración) salga de inmediato, sin esperar
    # al SMTP. Ver core/security.py::_send_mail_message para el techo de 3s.
    background_tasks.add_task(send_password_reset_email, payload.email, token)

    return generic_response


class ResetPasswordRequest(BaseModel):
    token: str
    password: str = Field(..., min_length=8)

    @field_validator("password")
    @classmethod
    def validate_password_strength(cls, value: str) -> str:
        if not any(char.isdigit() for char in value):
            raise ValueError("La contraseña debe contener al menos un número.")
        if not any(char.isupper() for char in value):
            raise ValueError("La contraseña debe contener al menos una letra mayúscula.")
        return value


@router.post("/reset-password")
async def reset_password(payload: ResetPasswordRequest, request: Request):
    await enforce_rate_limit(request, "reset-password")

    result = (
        supabase_admin.table("profiles")
        .select("id, password_reset_token_expires_at")
        .eq("password_reset_token", _hash_token(payload.token))
        .maybe_single()
        .execute()
    )
    profile = result.data
    if not profile:
        raise HTTPException(status_code=400, detail="Enlace de restablecimiento inválido.")

    expires_at = profile.get("password_reset_token_expires_at")
    if not expires_at or datetime.fromisoformat(expires_at) < datetime.now(timezone.utc):
        raise HTTPException(status_code=400, detail="El enlace ha expirado. Solicita uno nuevo.")

    try:
        supabase_admin.auth.admin.update_user_by_id(
            profile["id"], {"password": payload.password}
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"No se pudo actualizar la contraseña: {str(exc)}",
        ) from exc

    supabase_admin.table("profiles").update({
        "password_reset_token": None,
        "password_reset_token_expires_at": None,
    }).eq("id", profile["id"]).execute()

    return {"message": "Contraseña actualizada exitosamente."}