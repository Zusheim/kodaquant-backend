# core/quota_manager.py
"""
Aduana de cuotas para KodaQuant Terminal.
Verifica y consume cuota de servicios (predicciones / chat Quanti) según
el tier de suscripción, delegando el estado persistente a Supabase.

Ciclos:
    - FREE predicciones: reinicio diario (cron externo ya existente,
      fuera del alcance de este módulo).
    - FREE / PRO chat y PRO predicciones: reinicio cada 14 días vía
      `current_period_end` (pg_cron, ver migrations/quota_migration.sql).
    - ULTRA: sin límites, bypass total (no se leen ni consumen contadores).
"""
from datetime import datetime, timedelta, timezone
from fastapi import Depends, HTTPException, status
from postgrest.exceptions import APIError
from core.security import get_current_user, AuthenticatedUser
from core.supabase_client import supabase_admin

LIMITS = {
    "prediction": {"FREE": 3, "PRO": 50},
    "chat": {"FREE": 1, "PRO": 100},
}

DEFAULT_PROFILE_FIELDS = {
    "subscription_tier": "FREE",
    "api_requests_count": 0,
    "period_predictions_used": 0,
    "quanti_chat_used": 0,
}

# Cada (service_type, tier) apunta a la columna de Supabase que lleva su contador.
COUNTER_COLUMN = {
    "prediction": {"FREE": "api_requests_count", "PRO": "period_predictions_used"},
    "chat": {"FREE": "quanti_chat_used", "PRO": "quanti_chat_used"},
}

PROFILE_SELECT_FIELDS = (
    "subscription_tier, api_requests_count, period_predictions_used, "
    "quanti_chat_used, current_period_end"
)

# Ventana rodante de 14 días. Aplica a period_predictions_used y
# quanti_chat_used (PRO predicciones + FREE/PRO chat). NO aplica a
# api_requests_count (FREE predicciones = reinicio diario, cron externo,
# fuera del alcance de este módulo).
QUOTA_PERIOD = timedelta(days=14)


def _parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _format_reset_date(current_period_end: str | None) -> str:
    if not current_period_end:
        return "próximamente"
    try:
        dt = datetime.fromisoformat(current_period_end)
        return dt.strftime("%d/%m/%Y %H:%M UTC")
    except ValueError:
        return "próximamente"


def verify_quota(service_type: str):
    """
    Factory de dependencia FastAPI.

    Uso:
        @router.post("/predict", dependencies=[Depends(verify_quota("prediction"))])
        async def predict(...): ...
    """
    if service_type not in LIMITS:
        raise ValueError(f"service_type inválido: {service_type!r}")

    async def _dependency(
        current_user: AuthenticatedUser = Depends(get_current_user),
    ) -> AuthenticatedUser:
        try:
            result = (
                supabase_admin.table("profiles")
                .select(PROFILE_SELECT_FIELDS)
                .eq("id", current_user.id)
                .maybe_single()
                .execute()
            )
            profile = result.data if result else None
        except APIError:
            profile = None

        if not profile:
            # Sin insert/upsert: evita el 42501 del cliente anon/user-scoped.
            # Se asume perfil FREE con 0 usos; el UPDATE posterior sobre una
            # fila inexistente es un no-op silencioso (no rompe el flujo).
            profile = {"id": current_user.id, **DEFAULT_PROFILE_FIELDS}

        tier = (profile.get("subscription_tier") or "FREE").upper()

        # ULTRA: bypass total. Ni siquiera se toca la tabla.
        if tier == "ULTRA":
            return current_user

        if tier not in LIMITS[service_type]:
            tier = "FREE"  # Tier desconocido -> tratar como FREE por seguridad.

        limit = LIMITS[service_type][tier]
        column = COUNTER_COLUMN[service_type][tier]
        current_count = profile.get(column) or 0

        # Ventana rodante de 14 días, forzada en Python (no depende solo del
        # pg_cron externo). Si current_period_end no existe o ya venció,
        # se resetea el contador YA, antes de evaluar el límite.
        if column != "api_requests_count":
            period_end = _parse_utc(profile.get("current_period_end"))
            now_utc = datetime.now(timezone.utc)
            if period_end is None or now_utc >= period_end:
                new_period_end = now_utc + QUOTA_PERIOD
                supabase_admin.table("profiles").update(
                    {column: 0, "current_period_end": new_period_end.isoformat()}
                ).eq("id", current_user.id).execute()
                current_count = 0
                profile["current_period_end"] = new_period_end.isoformat()

        if current_count >= limit:
            reset_label = _format_reset_date(profile.get("current_period_end"))
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Límite alcanzado, reinicio en {reset_label}",
            )

        supabase_admin.table("profiles").update(
            {column: current_count + 1}
        ).eq("id", current_user.id).execute()

        return current_user

    return _dependency