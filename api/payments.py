# RUTA: api/payments.py
from datetime import datetime, timezone

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


def _get_user_email_and_language(user_id: str) -> tuple[str | None, str]:
    email = None
    try:
        admin_user = supabase_admin.auth.admin.get_user_by_id(user_id)
        email = getattr(admin_user.user, "email", None)
    except Exception as exc:
        print(f"[WEBHOOK] No se pudo resolver email de {user_id}: {exc}")

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
    termina el ciclo vigente. Sin prorrateo, sin reembolso parcial."""
    schedule = stripe.SubscriptionSchedule.create(from_subscription=subscription_id)
    current_phase = schedule["phases"][0]
    stripe.SubscriptionSchedule.modify(
        schedule["id"],
        phases=[
            {
                "items": [
                    {"price": item["price"], "quantity": item.get("quantity", 1)}
                    for item in current_phase["items"]
                ],
                "start_date": current_phase["start_date"],
                "end_date": current_phase["end_date"],
            },
            {"items": [{"price": new_price_id, "quantity": 1}], "iterations": 1},
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
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))

    return {"checkout_url": session.url}


class ChangePlanRequest(BaseModel):
    target_tier: str  # "free" | "pro" | "ultra"


@router.post("/change-plan")
async def change_plan(
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
        item_id = subscription["items"]["data"][0]["id"]
        period_end_iso = _ts_to_iso(subscription.get("current_period_end"))

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
            return {"status": "scheduled", "pending_tier": "free", "current_period_end": period_end_iso}

        if TIER_RANK[target_upper] > TIER_RANK[current_tier]:
            # Upgrade in-place (Pro -> Ultra): inmediato, con prorrateo.
            stripe.Subscription.modify(
                subscription_id,
                items=[{"id": item_id, "price": PRICE_MAP[target]}],
                proration_behavior="always_invoice",
                cancel_at_period_end=False,
            )
            return {"status": "processing"}  # subscription_tier lo confirma el webhook

        # Downgrade a un plan pago inferior (Ultra -> Pro): programado.
        _schedule_downgrade(subscription_id, PRICE_MAP[target])
        supabase_admin.table("profiles").update({
            "pending_tier": target_upper,
            "current_period_end": period_end_iso,
        }).eq("id", current_user.id).execute()
        return {"status": "scheduled", "pending_tier": target, "current_period_end": period_end_iso}

    except stripe.error.StripeError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))


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
    except (ValueError, stripe.error.SignatureVerificationError):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Webhook inválido")

    event_type = event["type"]
    obj = event["data"]["object"]
    previous = event["data"].get("previous_attributes", {})

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
        subscription_id = obj["id"]
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
            price_id = obj["items"]["data"][0]["price"]["id"]
            new_tier = TIER_BY_PRICE.get(price_id)

            if "cancel_at_period_end" in previous and cancel_at_period_end:
                # Cancelación total recién programada (paid -> FREE a fin de
                # ciclo). NO tocar subscription_tier -- eso lo hace .deleted.
                supabase_admin.table("profiles").update({
                    "cancel_at_period_end": True,
                    "pending_tier": "FREE",
                    "current_period_end": _ts_to_iso(obj.get("current_period_end")),
                }).eq("id", user_id).execute()
                background_tasks.add_task(_notify_cancel_scheduled, user_id)

            elif new_tier and new_tier != current_tier_db and not cancel_at_period_end:
                # Precio ya efectivo: upgrade inmediato (Pro->Ultra) o release
                # automático de un SubscriptionSchedule (fase 2 de un downgrade
                # Ultra->Pro activándose al fin del ciclo anterior).
                direction = "upgrade" if TIER_RANK[new_tier] > TIER_RANK[current_tier_db] else "downgrade"
                supabase_admin.table("profiles").update({
                    "subscription_tier": new_tier,
                    "pending_tier": None,
                    "cancel_at_period_end": False,
                }).eq("id", user_id).execute()
                background_tasks.add_task(_notify_change, user_id, new_tier, direction)

    elif event_type == "customer.subscription.deleted":
        subscription_id = obj["id"]
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

    return {"status": "success"}