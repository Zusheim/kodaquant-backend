# api/quanti_chat.py
import io
import json

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.security import AuthenticatedUser, get_current_user
from core.quota_manager import verify_quota
from core.supabase_client import supabase_admin as supabase
from services.quanti_engine import (
    QUANTI_CHAT_FALLBACK,
    describe_llm_failure,
    is_llm_connection_failure,
    sanitize_llm_output,
    stream_quanti_chat_completion,
)

router = APIRouter(prefix="/api/v1/quanti/chat", tags=["quanti-chat"])

MAX_FILE_BYTES = 5 * 1024 * 1024  # 5 MB
HISTORY_TURNS = 6
TITLE_MAX_LEN = 48

CHAT_SYSTEM_PROMPT = """You are Quanti, the AI engine embedded in the KodaQuant Terminal — the same analytical brain behind the platform's Alpha Seeker forecasting engine. You are not a general-purpose chatbot: you are a precision quantitative instrument, and in this chat channel specifically, you also act as the concierge who routes clients to the exact instrument that solves their request.

Tone: technical, direct, helpful, highly sophisticated — the register of a quantitative hedge fund's concierge desk, never a hedging generic assistant. Zero filler, zero unnecessary courtesies. Zero Markdown (no asterisks, no # headers).

HARD BAN — before anything else: you may NEVER open or fill a reply with "I'm sorry", "As an AI/language model", "I can't provide financial advice", "consult a licensed advisor", "diversify your portfolio" as a stock non-answer, or any equivalent generic financial-disclaimer boilerplate. That reflex produces a useless answer and actively damages the user's trust in the terminal. If a request sits in category 2 below, the correct move is never a refusal — it's a routing handoff to the tool that actually computes the number.

Classify every user message into ONE of these 4 categories and respond accordingly:

1) GREETING OR QUESTION ABOUT YOURSELF ("hi", "good morning", "what are you?", "what can you do?", thanks, farewells):
Respond in 1-2 lines, natural but corporate: introduce yourself as Quanti, KodaQuant's analysis engine, and offer to continue with a concrete financial query. No treatise — this is a welcome, not an analysis.

2) CAPITAL ALLOCATION / RECOMMENDATION REQUEST (the user states an amount of capital and asks what to do with it, asks what to invest in, asks for a pick, a portfolio, or "what would you do with my money" in any phrasing):
This chat channel has no live wire into the Alpha Seeker forecasting engine — you cannot compute a real, current allocation here, and improvising one would be a fabricated number, not an analysis. So you never refuse and you never invent a figure: you hand the client off to the instrument built for exactly this, in your own words, elegantly, hitting these beats in order:
   - Open by framing this as getting them a mathematically precise answer instead of a verbal guess — that's the reason for the handoff, not an excuse.
   - Direct them to the "Strategy Parameters" panel on the terminal's side panel.
   - Tell them to enter their exact capital in their desired currency.
   - Tell them to activate "Quanti's Choice" mode, and explain concretely what happens next: your Alpha Seeker algorithm scans the market in real time, evaluates directional accuracy (DirAcc) and volatility per asset, and automatically structures a tactical portfolio split between Risk and Reserve capital.
   - Close with confident, concierge-grade phrasing that frames this as superior service, not a deflection.
Vary your exact wording turn to turn — never recite a fixed script verbatim — but always hit every beat above. Never say you "can't" or "aren't allowed to" help; you ARE helping, by routing them to the precise tool for the job.

3) GENERAL FINANCIAL / MARKET QUERY (explaining an indicator, discussing market conditions or a specific asset, quantitative/statistical concepts, or analyzing an attached file's data) that does NOT ask you to allocate the user's capital or name a recommendation:
This is your core territory. Respond with the maximum analytical and technical depth the available data allows. If the user attached a file, the FILE CONTEXT block is your ONLY numeric source about that file — never invent figures that aren't there.

4) OUT OF SCOPE (general knowledge, jokes, programming unrelated to KodaQuant, everyday topics, anything unrelated to finance):
Decline firmly but not curtly — one sentence explaining why, not a flat door-slam. Tone example, generate variants, never repeat it verbatim every time: "As an AI specialized exclusively in finance and quantitative analysis, I don't process requests outside that scope. Let's keep our interaction to market data or strategy evaluation."

The LANGUAGE RULE and IDENTITY RULE for this session are prepended above this prompt by the engine, under the same system role — treat them as highest priority and non-negotiable."""


# ---------------------------------------------------------------------------
# Utilidades de archivo (pandas)
# ---------------------------------------------------------------------------

def _read_tabular_file(filename: str, raw_bytes: bytes) -> pd.DataFrame:
    buf = io.BytesIO(raw_bytes)
    lower = (filename or "").lower()
    if lower.endswith(".csv"):
        return pd.read_csv(buf)
    if lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(buf)
    raise ValueError("Formato no soportado. Usa .csv o .xlsx.")


def _summarize_dataframe(df: pd.DataFrame, max_cols: int = 8) -> str:
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        return "Archivo sin columnas numéricas detectables."

    lines = [f"Filas: {len(df)} | Columnas numéricas: {len(numeric_df.columns)}"]
    for col in list(numeric_df.columns)[:max_cols]:
        series = numeric_df[col].dropna()
        if series.empty:
            continue
        first, last = float(series.iloc[0]), float(series.iloc[-1])
        pct_change = ((last - first) / first * 100) if first != 0 else 0.0
        trend = "alcista" if last > first else ("bajista" if last < first else "plano")
        lines.append(
            f"{col}: media={series.mean():.4f} | desv.std={series.std(ddof=0):.4f} | "
            f"tendencia={trend} ({pct_change:+.2f}%) | min={series.min():.4f} | max={series.max():.4f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ownership guard
# ---------------------------------------------------------------------------

def _verify_thread_ownership(thread_id: str, user_id: str) -> None:
    """
    `maybe_single().execute()` puede devolver `None` DIRECTAMENTE (no un
    objeto con `.data = None`) cuando la query no matchea ninguna fila —
    comportamiento conocido de ciertas versiones de supabase-py/postgrest-py.
    Evaluar `.data` sobre eso revienta con AttributeError antes de llegar al
    chequeo de negocio. Se blinda en 2 capas: try/except por si el cliente
    lanza en vez de devolver None, y `getattr` en vez de acceso directo.

    NOTA: `supabase` acá es el cliente SERVICE ROLE (ver import arriba) —
    bypassea RLS por completo, así que este filtro por user_id es la ÚNICA
    barrera de autorización real. No se puede quitar ni relajar.
    """
    try:
        owned = (
            supabase.table("chat_threads")
            .select("id")
            .eq("id", thread_id)
            .eq("user_id", user_id)
            .maybe_single()
            .execute()
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Acceso denegado o hilo no encontrado.",
        ) from exc

    if owned is None or not getattr(owned, "data", None):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Acceso denegado o hilo no encontrado.",
        )

class ThreadPinUpdate(BaseModel):
    pinned: bool

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/threads")
async def list_threads(current_user: AuthenticatedUser = Depends(get_current_user)):
    result = (
        supabase.table("chat_threads")
        .select("id, title, pinned, created_at")
        .eq("user_id", current_user.id)
        .is_("deleted_at", "null")
        .order("created_at", desc=True)
        .execute()
    )
    return {"threads": result.data or []}


@router.get("/threads/{thread_id}/messages")
async def get_thread_messages(
    thread_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    _verify_thread_ownership(thread_id, current_user.id)

    result = (
        supabase.table("chat_messages")
        .select("id, role, content, created_at")
        .eq("thread_id", thread_id)
        .order("created_at", desc=False)
        .execute()
    )
    return {"messages": result.data or []}

# ── PATCH /api/v1/quanti/chat/threads/{thread_id} ───────────────────────────
@router.patch("/threads/{thread_id}")
async def update_thread_pin(
    thread_id: str,
    payload: ThreadPinUpdate,
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    _verify_thread_ownership(thread_id, current_user.id)

    try:
        result = (
            supabase.table("chat_threads")
            .update({"pinned": payload.pinned})
            .eq("id", thread_id)
            .eq("user_id", current_user.id)
            .execute()
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No se pudo actualizar el hilo.",
        ) from exc

    if not result.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Hilo no encontrado.")

    return {"thread": result.data[0]}


# ── DELETE /api/v1/quanti/chat/threads/{thread_id} (hard delete) ───────────
@router.delete("/threads/{thread_id}")
async def delete_thread(
    thread_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
):
    _verify_thread_ownership(thread_id, current_user.id)

    try:
        # Cascade manual: si la FK chat_messages.thread_id no tiene
        # ON DELETE CASCADE en el schema de Supabase, este borrado
        # explícito es obligatorio para no dejar mensajes huérfanos.
        supabase.table("chat_messages").delete().eq("thread_id", thread_id).execute()
        result = (
            supabase.table("chat_threads")
            .delete()
            .eq("id", thread_id)
            .eq("user_id", current_user.id)
            .execute()
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No se pudo eliminar el hilo.",
        ) from exc

    if not result.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Hilo no encontrado.")

    return {"deleted": True}

def _persist_chat_turn(thread_id: str, prompt: str, reply: str) -> None:
    supabase.table("chat_messages").insert(
        [
            {"thread_id": thread_id, "role": "user", "content": prompt},
            {"thread_id": thread_id, "role": "assistant", "content": reply},
        ]
    ).execute()


async def _chat_sse_generator(thread_id: str, prompt: str, llm_messages: list[dict]):
    """
    Generador SSE del turno de chat. Contrato:
        data: {"type":"thread","thread_id":"..."}\\n\\n   (1 vez, al inicio)
        data: {"type":"token","text":"..."}\\n\\n          (N veces, deltas del LLM)
        data: {"type":"done"}\\n\\n                        (cierre, siempre se emite)

    Circuit Breaker: el motor (stream_quanti_chat_completion) corre con
    read timeout=None, así que una generación larga NUNCA dispara esta
    rama por sí sola — solo un fallo real de conexión al servidor GGUF
    (is_llm_connection_failure) o un error inesperado del transporte cae
    acá. En cualquier caso, la respuesta HTTP ya salió en 200 (SSE), así
    que degradamos con un evento 'token' de fallback + 'done' en vez de
    intentar levantar un status code a esta altura.
    """
    yield f"data: {json.dumps({'type': 'thread', 'thread_id': thread_id})}\n\n"

    full_text = ""
    try:
        async for delta in stream_quanti_chat_completion(llm_messages):
            full_text += delta
            yield f"data: {json.dumps({'type': 'token', 'text': delta})}\n\n"
    except Exception as exc:  # noqa: BLE001
        if is_llm_connection_failure(exc):
            print(f"⚠️ Circuit Breaker activado — servidor GGUF inalcanzable (Quanti Chat stream): {describe_llm_failure(exc)}")
        else:
            print(f"⚠️ Fallo no-conexión en Quanti Chat stream (no se enmascara como servidor caído): {describe_llm_failure(exc)}")
        full_text = QUANTI_CHAT_FALLBACK
        yield f"data: {json.dumps({'type': 'token', 'text': full_text})}\n\n"

    sanitized = sanitize_llm_output(full_text)
    reply = sanitized[0] if sanitized else QUANTI_CHAT_FALLBACK

    try:
        _persist_chat_turn(thread_id, prompt, reply)
    except Exception as exc:  # noqa: BLE001
        # El stream ya se le entregó al usuario completo — un fallo de
        # persistencia acá no debe convertirse en un error HTTP a esta
        # altura, pero sí queda logueado para no perderlo en silencio.
        print(f"⚠️ No se pudo persistir el turno de chat (thread_id={thread_id}): {exc}")

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


@router.post("")
async def send_message(
    prompt: str = Form(...),
    thread_id: str | None = Form(None),
    file: UploadFile | None = File(None),
    # verify_quota("chat") ya envuelve get_current_user: valida cuota Y
    # devuelve el AuthenticatedUser en una sola dependencia.
    current_user: AuthenticatedUser = Depends(verify_quota("chat")),
):
    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="El mensaje no puede estar vacío.")

    # --- Resolver / crear hilo, validando ownership ---
    if thread_id:
        _verify_thread_ownership(thread_id, current_user.id)
    else:
        title = prompt[:TITLE_MAX_LEN] + ("…" if len(prompt) > TITLE_MAX_LEN else "")
        created = (
            supabase.table("chat_threads")
            .insert({"user_id": current_user.id, "title": title})
            .execute()
        )
        thread_id = created.data[0]["id"]

    # --- Contexto de archivo adjunto (opcional) ---
    file_context = ""
    if file is not None:
        raw_bytes = await file.read()
        if len(raw_bytes) > MAX_FILE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Archivo demasiado grande (máx 5MB).",
            )
        try:
            df = _read_tabular_file(file.filename or "", raw_bytes)
            file_context = f"\n\nCONTEXTO DE ARCHIVO ({file.filename}):\n{_summarize_dataframe(df)}"
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"No se pudo procesar el archivo: {exc}",
            )

    # --- Historial reciente (últimos HISTORY_TURNS mensajes del hilo) ---
    history_result = (
        supabase.table("chat_messages")
        .select("role, content")
        .eq("thread_id", thread_id)
        .order("created_at", desc=True)
        .limit(HISTORY_TURNS)
        .execute()
    )
    recent_history = list(reversed(history_result.data or []))

    llm_messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
    llm_messages.extend({"role": m["role"], "content": m["content"]} for m in recent_history)
    llm_messages.append({"role": "user", "content": prompt + file_context})

    # Todo lo síncrono (validación, ownership, hilo, archivo, historial) ya
    # corrió arriba y puede seguir devolviendo 400/403 normales. De acá en
    # más la respuesta es 200 (SSE) sin importar qué pase con el LLM local
    # — el Circuit Breaker vive DENTRO del generador, no acá.
    return StreamingResponse(
        _chat_sse_generator(thread_id, prompt, llm_messages),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # evita que nginx bufferee el stream
            "Connection": "keep-alive",
        },
    )