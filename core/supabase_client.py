from supabase import create_client, Client
from core.config import settings

# --- Cliente ANON (respeta RLS) ---
# Para todo lo que ya funciona hoy: auth.get_user() y cualquier lectura o
# escritura donde las policies de RLS deben seguir aplicando (ver
# core/security.py: get_current_user, verify_api_credits). NO se toca.
url: str = settings.SUPABASE_URL
key: str = settings.SUPABASE_KEY
supabase: Client = create_client(url, key)

# --- Cliente SERVICE ROLE (bypassea RLS por completo) ---
# Uso EXCLUSIVO para operaciones de servidor de confianza donde el backend
# ya verificó la identidad del usuario por su cuenta (ej. el INSERT en
# chat_threads, hecho después de pasar por get_current_user). Este cliente
# no tiene NINGÚN control de acceso propio -- es admin total sobre la DB.
# Nunca lo uses para resolver datos a partir de un id que no hayas validado
# vos mismo antes, y nunca lo expongas fuera del backend.
service_key: str = settings.SUPABASE_SERVICE_ROLE_KEY
supabase_admin: Client = create_client(url, service_key)