import logging

# Configuración del logger interno
logger = logging.getLogger(__name__)

# Matriz Propietaria de KodaQuant (100% Offline y Autónoma)
# Aquí tú tienes el control absoluto de las tasas del mercado.
KODAQUANT_EXCHANGE_RATES = {
    "USD": 1.0,      # Dólar Estadounidense (Base)
    "MXN": 18.50,    # Peso Mexicano
    "EUR": 0.92,     # Euro
    "GBP": 0.78,     # Libra Esterlina
    "CAD": 1.36      # Dólar Canadiense
}

async def convert_to_usd(amount: float, currency: str) -> dict:
    """
    Motor cuantitativo de conversión interno.
    Cero dependencias de APIs externas. Latencia cero.
    """
    currency_code = currency.upper().strip()
    
    # 1. Validación estricta en la matriz local
    if currency_code not in KODAQUANT_EXCHANGE_RATES:
        logger.error(f"Fallo de motor: Divisa {currency_code} fuera del radar autónomo.")
        raise ValueError(f"La divisa {currency_code} no está soportada en el ecosistema KodaQuant.")

    # 2. Si ya son dólares, retorna el monto puro
    if currency_code == "USD":
        return {
            "original_amount": amount,
            "original_currency": currency_code,
            "usd_amount": float(amount),
            "internal_rate": 1.0,
            "status": "autonomous_success"
        }

    # 3. Cálculo matemático instantáneo
    internal_rate = KODAQUANT_EXCHANGE_RATES[currency_code]
    budget_usd = round(amount / internal_rate, 2)
    
    logger.info(f"Conversión ejecutada localmente: {amount} {currency_code} -> {budget_usd} USD")

    # 4. Retorno del reporte estructurado
    return {
        "original_amount": amount,
        "original_currency": currency_code,
        "usd_amount": budget_usd,
        "internal_rate": internal_rate,
        "status": "autonomous_success"
    }