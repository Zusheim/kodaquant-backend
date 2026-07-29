# api/support.py
"""
Quanti Support — Nivel 1 (chat IA Tier-1) + Nivel 2 (escalación a humano).
Router aislado; se monta en main.py vía app.include_router(support_router).
"""

import ipaddress
import time
import uuid
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from core.supabase_client import supabase_admin as _supabase
from services.quanti_engine import generate_quanti_support_reply, is_support_escalation

router = APIRouter(prefix="/api/support", tags=["support"])

# ---------------------------------------------------------------------------
# Anti-spam — rate limit en memoria (1 ticket por IP cada 30 minutos). Vive
# en el proceso: se resetea en cada deploy/restart y NO se comparte entre
# workers si corres > 1 proceso uvicorn/gunicorn. Migrar a Redis si se
# escala horizontalmente.
# ---------------------------------------------------------------------------
_TICKET_COOLDOWN_SECONDS = 30 * 60
_last_ticket_by_ip: dict[str, float] = {}


def _get_client_ip(request: Request) -> str:
    """
    request.client.host es la IP de la conexión TCP directa — el cliente NO
    puede falsearla. X-Forwarded-For sí lo controla el cliente en su primer
    salto (spoofing trivial: mandar un header distinto en cada request rota
    el cooldown de 30 min). Por eso ahora es la fuente PRIMARIA, y XFF queda
    como fallback validado (solo si viene un formato de IP válido).
    """
    if request.client and request.client.host:
        return request.client.host

    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        candidate = forwarded.split(",")[0].strip()
        try:
            ipaddress.ip_address(candidate)
            return candidate
        except ValueError:
            pass

    return "unknown"


def _is_rate_limited(ip: str) -> bool:
    last = _last_ticket_by_ip.get(ip)
    return last is not None and (time.time() - last) < _TICKET_COOLDOWN_SECONDS


def _mark_ticket_sent(ip: str) -> None:
    _last_ticket_by_ip[ip] = time.time()


def _insert_support_ticket(user_ip: str, chat_history: list[dict], user_email: Optional[str]) -> str:
    ticket_id = str(uuid.uuid4())
    _supabase.table("koda_support_tickets").insert({
        "id": ticket_id,
        "user_ip": user_ip,
        "user_email": user_email,
        "chat_history": chat_history,
        "status": "open",
    }).execute()
    return ticket_id


def _send_escalation_email(ticket_id: str, user_email: Optional[str], chat_history: list[dict]) -> None:
    """Placeholder de notificación — listo para smtplib real. Un fallo de
    correo nunca debe tumbar el request; el ticket ya quedó en Supabase."""
    try:
        # import smtplib
        # from email.message import EmailMessage
        # msg = EmailMessage()
        # msg["Subject"] = f"[KodaQuant] Nuevo ticket de soporte #{ticket_id}"
        # msg["From"] = "no-reply@kodaquant.com"
        # msg["To"] = "karim.egure@gmail.com"
        # msg.set_content(f"Ticket {ticket_id}\nEmail usuario: {user_email}\n\n{chat_history}")
        # with smtplib.SMTP(os.getenv("SMTP_HOST"), 587) as server:
        #     server.starttls()
        #     server.login(os.getenv("SMTP_USER"), os.getenv("SMTP_PASSWORD"))
        #     server.send_message(msg)
        print(f"📧 [SIMULADO] Ticket #{ticket_id} → karim.egure@gmail.com | usuario: {user_email}")
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️ No se pudo enviar el correo de escalación (ticket #{ticket_id} ya está en BD): {exc}")


_ESCALATION_CONFIRMATION = {
    "en": "Thank you. I've escalated this to Karim Estrada (Administrator) — he'll personally reach out to {email} shortly.",
    "es": "Gracias. Escalé esto con Karim Estrada (Administrador) — se pondrá en contacto contigo a {email} en breve.",
}

_RATE_LIMIT_MESSAGE = {
    "en": "You already have an open ticket. Karim Estrada will reach out shortly — no need to send another request yet.",
    "es": "Ya tienes un ticket abierto. Karim Estrada se pondrá en contacto en breve — no hace falta enviar otra solicitud todavía.",
}


class SupportChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class SupportChatRequest(BaseModel):
    messages: list[SupportChatMessage] = Field(..., min_length=1)
    language: Optional[str] = "en"
    email: Optional[str] = None


@router.post("/chat")
async def support_chat(payload: SupportChatRequest, request: Request):
    lang = (payload.language or "en").strip().lower()
    lang = lang if lang in ("en", "es") else "en"

    history = [m.model_dump() for m in payload.messages]
    reply = await generate_quanti_support_reply(history, language=lang)

    if not is_support_escalation(reply):
        return {"status": "ok", "message": reply}

    if not payload.email:
        return {"status": "needs_email"}

    client_ip = _get_client_ip(request)
    if _is_rate_limited(client_ip):
        return {"status": "rate_limited", "message": _RATE_LIMIT_MESSAGE[lang]}

    try:
        ticket_id = _insert_support_ticket(client_ip, history, payload.email)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error al crear el ticket de soporte: {exc}")

    _mark_ticket_sent(client_ip)
    _send_escalation_email(ticket_id, payload.email, history)

    return {
        "status": "escalated",
        "ticket_id": ticket_id,
        "message": _ESCALATION_CONFIRMATION[lang].format(email=payload.email),
    }