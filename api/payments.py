# RUTA: api/payments.py
import json
import logging
from datetime import datetime, timezone
from typing import Any

import stripe
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel

from core.config import settings
from core.security import AuthenticatedUser, get_current_user
from core.supabase_client import supabase_admin
from core.security import (
    send_subscription_upgrade_email,
    send_subscription_downgrade_email,
    send_subscription_cancel_email,
)

logger = logging.getLogger(__name__)

stripe.api_key = settings.STRIPE_SECRET_KEY

router = APIRouter(prefix="/api/v1/payments", tags=["payments"])

PRICE_MAP = {
    "pro": settings.STRIPE_PRICE_PRO,
    "ultra": settings.STRIPE_PRICE_ULTRA,
}
TIER_BY_PRICE = {v: k.upper() for k, v in PRICE_MAP.items() if v}
TIER_RANK = {"FREE": 0, "PRO": 1, "ULTRA": 2}


def _ts_to_iso(ts: int | None) -> str | None:
    if not ts:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def _stripe_object_to_dict(obj: Any) -> dict:
    """Convierte recursivamente un Event/StripeObject del SDK a dict/list
    nativos de Python. En SDK v12+, ningún StripeObject anidado garantiza
    .get() en accesos encadenados -- normalizar UNA vez acá (aquí, en
    change-plan, en el webhook) evita repetir ese riesgo en cada callsite."""
    if hasattr(obj, "to_dict_recursive"):
        return obj.to_dict_recursive()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    try:
        return json.loads(json.dumps(obj, default=str))
    except (TypeError, ValueError) as exc:
        logger.error("No se pudo normalizar objeto Stripe: %s", exc)
        return {}


def _extract_period_end(sub_dict: dict) -> int | None:
    """Extrae el timestamp Unix de current_period_end de forma segura.

    Root cause real del 23502: a partir de la versión de API 'Basil'
    (2025-03-31+), Stripe removió current_period_start/current_period_end
    del objeto Subscription de nivel superior. Ahora viven por-item, en
    items.data[N].current_period_end (ver changelog de Stripe: "Adds
    subscription item-level billing periods and removes subscription-level
    periods"). Cualquier cuenta/SDK apuntando a esa versión de API (o
    posterior) devuelve `None` en sub_dict.get("current_period_end"),
    que es exactamente el síntoma reportado.

    Estrategia: leer primero desde el primer item de la suscripción (fuente
    de verdad en Basil+), con fallback al campo legacy de nivel superior
    para cuentas todavía en una versión de API pre-Basil.
    """
    items = (sub_dict.get("items") or {}).get("data") or []
    if items:
        item_period_end = items[0].get("current_period_end")
        if item_period_end is not None:
            return int(item_period_end)

    legacy_period_end = sub_dict.get("current_period_end")
    if legacy_period_end is not None:
        return int(legacy_period_end)

    return None


def _require_period_end_iso(sub_dict: dict, subscription_id: str) -> str:
    """Fail-fast: valida ANTES de tocar la base de datos. Si Stripe no
    devuelve current_period_end en ninguna de las dos ubicaciones posibles,
    aborta con un error claro en vez de dejar pasar un None hacia el
    .update() de Supabase (que dispara el 23502 sobre la constraint
    NOT NULL de profiles.current_period_end)."""
    period_end_ts = _extract_period_end(sub_dict)
    if period_end_ts is None:
        logger.error(
            "current_period_end ausente en la suscripción %s (ni en items[0] ni en el nivel superior)",
            subscription_id,
        )
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "No se pudo determinar la fecha de fin de período de la suscripción en Stripe.",
        )
    period_end_iso = _ts_to_iso(period_end_ts)
    if period_end_iso is None:
        logger.error("Fallo al convertir current_period_end=%s a ISO-8601 (sub=%s)", period_end_ts, subscription_id)
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "No se pudo convertir la fecha de fin de período de la suscripción.",
        )
    return period_end_iso


def _release_schedule_if_attached(sub_dict: dict, subscription_id: str) -> None:
    """Rompe el vínculo Subscription<->SubscriptionSchedule si existe uno activo.

    Root cause de los 400 en change-plan/cancel: una vez que una Subscription
    queda atada a un SubscriptionSchedule (ej. por un downgrade programado
    previo vía _schedule_downgrade), Stripe bloquea CUALQUIER
    Subscription.modify() directo -- tanto cambio de item/price como
    cancel_at_period_end -- exigiendo que la modificación pase por el
    schedule ("You cannot migrate a subscription that is already attached
    to a schedule" / "updating any cancelation behavior directly is not
    allowed"). `SubscriptionSchedule.release()` desata el schedule de la
    Subscription sin cancelarla, devolviendo el control directo sobre
    Subscription.modify(). Se ejecuta ANTES de cualquier modify() o de
    crear un schedule nuevo (_schedule_downgrade también fallaría con 400
    si ya hay uno attached)."""
    schedule_id = sub_dict.get("schedule")
    if not schedule_id:
        return
    try:
        stripe.SubscriptionSchedule.release(schedule_id)
        logger.info(
            "Schedule %s liberado para desbloquear modificación directa de sub %s",
            schedule_id, subscription_id,
        )
    except stripe.error.StripeError as exc:
        logger.error(
            "No se pudo liberar el schedule %s de la suscripción %s: %s",
            schedule_id, subscription_id, exc,
        )
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "La suscripción está atada a un schedule de Stripe que no pudo liberarse. Contactá a soporte.",
        )


def _get_user_email_and_language(user_id: str) -> tuple[str | None, str]:
    email = None
    try:
        admin_user = supabase_admin.auth.admin.get_user_by_id(user_id)
        email = getattr(admin_user.user, "email", None)
    except Exception as exc:
        logger.warning("No se pudo resolver email de %s: %s", user_id, exc)

    profile = (
        supabase_admin.table("profiles")
        .select("preferred_language")
        .eq("id", user_id)
        .maybe_single()
        .execute()
    )
    language = (profile.data or {}).get("preferred_language") or "en"
    return email, language


async def _notify_change(user_id: str, tier: str, direction: str):
    email, language = _get_user_email_and_language(user_id)
    if not email:
        return
    if direction == "upgrade":
        await send_subscription_upgrade_email(email, tier, language)
    else:
        await send_subscription_downgrade_email(email, tier, language)


async def _notify_cancel_scheduled(user_id: str):
    email, language = _get_user_email_and_language(user_id)
    if email:
        await send_subscription_downgrade_email(email, "FREE", language)


async def _notify_cancel_final(user_id: str):
    email, language = _get_user_email_and_language(user_id)
    if email:
        await send_subscription_cancel_email(email, language)


def _schedule_downgrade(subscription_id: str, new_price_id: str):
    """Downgrade a un plan pago inferior (ej. ULTRA -> PRO): SubscriptionSchedule
    con 2 fases -- la actual (sin tocar) y una nueva que arranca justo cuando
    termina el ciclo vigente. Sin prorrateo, sin reembolso parcial.

    Root cause real del 400 "Received unknown parameter: phases[iterations]":
    Stripe removió `iterations` de phases en subscription_schedules a partir
    de la versión de API 'Clover' (2025-09-30), reemplazándolo por `duration`
    (ver changelog: "Removes iterations parameter for subscription schedules").
    Cualquier cuenta en esa versión de API o posterior rechaza `iterations`
    de plano. `duration` se ancla al intervalo de facturación real del price
    destino (no se asume 'month' a ciegas -- rompería planes anuales)."""
    schedule = stripe.SubscriptionSchedule.create(from_subscription=subscription_id)
    schedule_dict: dict = _stripe_object_to_dict(schedule)
    current_phase = schedule_dict["phases"][0]

    new_price_dict: dict = _stripe_object_to_dict(stripe.Price.retrieve(new_price_id))
    recurring = new_price_dict.get("recurring") or {}
    duration_interval = recurring.get("interval") or "month"
    duration_interval_count = recurring.get("interval_count") or 1

    stripe.SubscriptionSchedule.modify(
        schedule_dict["id"],
        # Explícito (no queda a un default implícito de Stripe): al terminar
        # la fase 2, el schedule se libera y la suscripción sigue renovando
        # normalmente con el price nuevo -- consistente con "downgrade
        # permanente", no una promo temporal de 1 ciclo.
        end_behavior="release",
        phases=[
            {
                "items": [
                    {"price": item["price"], "quantity": item.get("quantity", 1)}
                    for item in current_phase["items"]
                ],
                "start_date": current_phase["start_date"],
                "end_date": current_phase["end_date"],
            },
            {
                "items": [{"price": new_price_id, "quantity": 1}],
                "duration": {
                    "interval": duration_interval,
                    "interval_count": duration_interval_count,
                },
            },
        ],
    )


@router.post("/checkout")
async def create_checkout_session(
    plan: str = Query(..., description="Escribe 'pro' o 'ultra'"),
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    plan_key = plan.lower()
    price_id = PRICE_MAP.get(plan_key)
    if not price_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Plan inválido. Usa 'pro' o 'ultra'.")

    # Checkout SOLO para alta nueva (usuario sin suscripción activa). Si ya
    # tiene una, upgrade/downgrade/cancelación van por /change-plan -- nunca
    # por acá, o se duplicaría la suscripción en Stripe.
    existing = (
        supabase_admin.table("profiles")
        .select("stripe_subscription_id")
        .eq("id", current_user.id)
        .maybe_single()
        .execute()
    )
    if existing.data and existing.data.get("stripe_subscription_id"):
        raise HTTPException(status.HTTP_409_CONFLICT, "Ya tenés una suscripción activa. Usa /change-plan.")

    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            line_items=[{"price": price_id, "quantity": 1}],
            client_reference_id=current_user.id,
            customer_email=current_user.email,
            metadata={"user_id": current_user.id, "target_tier": plan_key.upper()},
            # subscription_data.metadata es la ÚNICA forma de que el metadata
            # sobreviva al objeto Subscription (el de Session no se copia solo).
            subscription_data={"metadata": {"user_id": current_user.id, "target_tier": plan_key.upper()}},
            success_url=settings.STRIPE_SUCCESS_URL,
            cancel_url=settings.STRIPE_CANCEL_URL,
        )
    except stripe.error.StripeError as e:
        logger.warning("StripeError en checkout (user=%s, plan=%s): %s", current_user.id, plan_key, e)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    except Exception as exc:
        logger.error("Fallo inesperado en checkout (user=%s, plan=%s): %s", current_user.id, plan_key, exc, exc_info=True)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Error interno al crear la sesión de pago.")

    return {"checkout_url": session.url}


class ChangePlanRequest(BaseModel):
    target_tier: str  # "free" | "pro" | "ultra"


class ChangePlanResponse(BaseModel):
    """Respuesta fuertemente tipada para que el frontend SPA sepa exactamente
    qué pasó y, si corresponde, a dónde debe empujar el router -- en vez de
    inferirlo de un mensaje genérico de texto."""

    status: str  # "scheduled" | "processing"
    message: str
    pending_tier: str | None = None
    current_period_end: str | None = None
    redirect_url: str | None = None  # ej. "/cancel" -- None cuando no aplica


@router.post("/change-plan", response_model=ChangePlanResponse)
async def change_plan(
    # No existe un endpoint /cancel-plan separado: la cancelación es
    # target_tier="free" acá mismo (rama cancel_at_period_end más abajo).
    # El redirect_url="/cancel" se setea únicamente en esa rama.
    payload: ChangePlanRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    target = payload.target_tier.lower()
    if target not in ("free", "pro", "ultra"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "target_tier debe ser 'free', 'pro' o 'ultra'.")

    profile = (
        supabase_admin.table("profiles")
        .select("subscription_tier, stripe_subscription_id")
        .eq("id", current_user.id)
        .maybe_single()
        .execute()
    )
    if not profile.data or not profile.data.get("stripe_subscription_id"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No hay suscripción activa para modificar. Usa /checkout.")

    subscription_id = profile.data["stripe_subscription_id"]
    current_tier = profile.data.get("subscription_tier", "FREE")
    target_upper = target.upper()

    if target_upper == current_tier:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Ya estás en ese plan.")

    try:
        subscription = stripe.Subscription.retrieve(subscription_id)
        # Normalización única a dict -- misma causa raíz que el webhook:
        # StripeObject en SDK v12+ no garantiza .get() en accesos anidados.
        sub_dict: dict = _stripe_object_to_dict(subscription)

        # The Breaker: si la sub está attached a un schedule (de un downgrade
        # previo, ej. Ultra->Pro programado), liberarlo ANTES de tocar items
        # o cancel_at_period_end -- si no, tanto el modify() de abajo como
        # _schedule_downgrade() (que crea un schedule nuevo) rechazan con 400.
        _release_schedule_if_attached(sub_dict, subscription_id)

        items = sub_dict.get("items", {}).get("data", [])
        if not items:
            logger.error("Suscripción %s sin items al ejecutar change-plan", subscription_id)
            raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Suscripción inconsistente en Stripe.")
        item_id = items[0]["id"]

        # Fail-fast: extrae y valida current_period_end ANTES de escribir en
        # Supabase (ver _extract_period_end para la causa raíz real: Stripe
        # API 'Basil' 2025-03-31+ movió este campo a items.data[0]).
        period_end_iso = _require_period_end_iso(sub_dict, subscription_id)

        if target == "free":
            # Cancelación total: sigue con el plan actual hasta fin de ciclo.
            # subscription_tier real lo pisa el webhook (.deleted), esto solo
            # deja constancia visible para el frontend.
            stripe.Subscription.modify(subscription_id, cancel_at_period_end=True)
            supabase_admin.table("profiles").update({
                "pending_tier": "FREE",
                "cancel_at_period_end": True,
                "current_period_end": period_end_iso,
            }).eq("id", current_user.id).execute()
            return ChangePlanResponse(
                status="scheduled",
                message=f"Tu suscripción se cancelará el {period_end_iso}. Hasta entonces seguís con acceso.",
                pending_tier="free",
                current_period_end=period_end_iso,
                redirect_url="/cancel",  # instrucción explícita de ruteo para el SPA
            )

        if TIER_RANK[target_upper] > TIER_RANK[current_tier]:
            # Upgrade in-place (Pro -> Ultra): inmediato, con prorrateo.
            stripe.Subscription.modify(
                subscription_id,
                items=[{"id": item_id, "price": PRICE_MAP[target]}],
                proration_behavior="always_invoice",
                cancel_at_period_end=False,
            )
            # subscription_tier lo confirma el webhook; no hay ruteo especial acá.
            return ChangePlanResponse(
                status="processing",
                message=f"Tu plan se está actualizando a {target_upper}.",
            )

        # Downgrade a un plan pago inferior (Ultra -> Pro): programado.
        _schedule_downgrade(subscription_id, PRICE_MAP[target])
        supabase_admin.table("profiles").update({
            "pending_tier": target_upper,
            "current_period_end": period_end_iso,
        }).eq("id", current_user.id).execute()
        return ChangePlanResponse(
            status="scheduled",
            message=f"Tu plan bajará a {target_upper} el {period_end_iso}.",
            pending_tier=target,
            current_period_end=period_end_iso,
        )

    except stripe.error.StripeError as e:
        logger.warning("StripeError en change-plan (user=%s, sub=%s): %s", current_user.id, subscription_id, e)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Fallo inesperado en change-plan (user=%s, sub=%s): %s",
            current_user.id, subscription_id, exc, exc_info=True,
        )
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Error interno al modificar la suscripción.")


@router.get("/subscription")
async def get_subscription_status(current_user: AuthenticatedUser = Depends(get_current_user)):
    profile = (
        supabase_admin.table("profiles")
        .select("subscription_tier, pending_tier, cancel_at_period_end, current_period_end")
        .eq("id", current_user.id)
        .maybe_single()
        .execute()
    )
    data = profile.data or {}
    pending = data.get("pending_tier")
    return {
        "subscription_tier": (data.get("subscription_tier") or "FREE").lower(),
        "pending_tier": pending.lower() if pending else None,
        "cancel_at_period_end": bool(data.get("cancel_at_period_end")),
        "current_period_end": data.get("current_period_end"),
    }


@router.post("/webhook")
async def stripe_webhook(request: Request, background_tasks: BackgroundTasks):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    endpoint_secret = settings.STRIPE_WEBHOOK_SECRET

    try:
        event = stripe.Webhook.construct_event(payload, sig_header, endpoint_secret)
    except (ValueError, stripe.error.SignatureVerificationError) as exc:
        logger.warning("Webhook rechazado (firma/payload inválido): %s", exc)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Webhook inválido")

    # Normalización única a dict nativo -- de acá en más TODO acceso usa
    # .get()/[] de dict estándar.
    event_dict: dict = _stripe_object_to_dict(event)
    event_type: str = event_dict.get("type", "")
    event_id: str = event_dict.get("id", "unknown")
    data: dict = event_dict.get("data") or {}
    obj: dict = data.get("object") or {}
    previous: dict = data.get("previous_attributes") or {}

    try:
        if event_type == "checkout.session.completed":
            user_id = obj.get("metadata", {}).get("user_id")
            target_tier = obj.get("metadata", {}).get("target_tier", "PRO")
            if user_id:
                supabase_admin.table("profiles").update({
                    "subscription_tier": target_tier,
                    "stripe_customer_id": obj.get("customer"),
                    "stripe_subscription_id": obj.get("subscription"),
                    "cancel_at_period_end": False,
                    "pending_tier": None,
                }).eq("id", user_id).execute()
                background_tasks.add_task(_notify_change, user_id, target_tier, "upgrade")

        elif event_type == "customer.subscription.updated":
            subscription_id = obj.get("id")
            profile = (
                supabase_admin.table("profiles")
                .select("id, subscription_tier")
                .eq("stripe_subscription_id", subscription_id)
                .maybe_single()
                .execute()
            )
            if profile.data:
                user_id = profile.data["id"]
                current_tier_db = profile.data.get("subscription_tier", "FREE")
                cancel_at_period_end = obj.get("cancel_at_period_end", False)
                items = obj.get("items", {}).get("data", [])
                price_id = items[0]["price"]["id"] if items else None
                new_tier = TIER_BY_PRICE.get(price_id)

                if "cancel_at_period_end" in previous and cancel_at_period_end:
                    # Cancelación total recién programada (paid -> FREE a fin
                    # de ciclo). NO tocar subscription_tier -- eso lo hace
                    # .deleted.
                    # Mismo fix que change-plan: current_period_end puede
                    # vivir en items.data[0] (Stripe API 'Basil' 2025-03-31+)
                    # en vez del nivel superior de `obj`.
                    update_payload: dict = {
                        "cancel_at_period_end": True,
                        "pending_tier": "FREE",
                    }
                    period_end_ts = _extract_period_end(obj)
                    if period_end_ts is not None:
                        update_payload["current_period_end"] = _ts_to_iso(period_end_ts)
                    else:
                        # No se crashea el webhook por esto (Stripe reintentaría
                        # indefinidamente si el campo nunca llega): se omite el
                        # campo en vez de mandar null y violar la constraint
                        # NOT NULL, y se deja rastro en logs para investigar.
                        logger.error(
                            "customer.subscription.updated (sub=%s): current_period_end no disponible "
                            "ni en items[0] ni en el nivel superior; se omite el campo en el update.",
                            subscription_id,
                        )
                    supabase_admin.table("profiles").update(update_payload).eq("id", user_id).execute()
                    background_tasks.add_task(_notify_cancel_scheduled, user_id)

                elif new_tier and new_tier != current_tier_db and not cancel_at_period_end:
                    # Precio ya efectivo: upgrade inmediato (Pro->Ultra) o
                    # release automático de un SubscriptionSchedule (fase 2 de
                    # un downgrade Ultra->Pro activándose al fin del ciclo
                    # anterior).
                    direction = "upgrade" if TIER_RANK[new_tier] > TIER_RANK[current_tier_db] else "downgrade"
                    supabase_admin.table("profiles").update({
                        "subscription_tier": new_tier,
                        "pending_tier": None,
                        "cancel_at_period_end": False,
                    }).eq("id", user_id).execute()
                    background_tasks.add_task(_notify_change, user_id, new_tier, direction)

        elif event_type == "customer.subscription.deleted":
            subscription_id = obj.get("id")
            profile = (
                supabase_admin.table("profiles")
                .select("id")
                .eq("stripe_subscription_id", subscription_id)
                .maybe_single()
                .execute()
            )
            if profile.data:
                user_id = profile.data["id"]
                supabase_admin.table("profiles").update({
                    "subscription_tier": "FREE",
                    "pending_tier": None,
                    "cancel_at_period_end": False,
                    "stripe_subscription_id": None,
                }).eq("id", user_id).execute()
                background_tasks.add_task(_notify_cancel_final, user_id)

        else:
            logger.info("Evento Stripe no manejado: %s (id=%s)", event_type, event_id)

    except Exception as exc:
        # Cualquier fallo real de procesamiento (Supabase, KeyError, etc.)
        # devuelve 500 para que Stripe reintente -- nunca un AttributeError
        # crudo ni un 200 silencioso que enmascare el fallo.
        logger.error(
            "Error procesando webhook %s (id=%s): %s", event_type, event_id, exc, exc_info=True
        )
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Error interno procesando el evento")

    return {"status": "success"}