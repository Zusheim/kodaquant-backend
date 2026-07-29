import json
from typing import Union, Dict, Any


class RadarDataError(Exception):
    """Excepción lanzada cuando radar_data no puede ser interpretado correctamente."""
    pass


async def calculate_portfolio(budget_usd: float, radar_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    # Escudo de Datos Estricto
    if isinstance(radar_data, str):
        try:
            radar_data = json.loads(radar_data)
        except json.JSONDecodeError as e:
            raise RadarDataError(
                f"radar_data no contiene un JSON válido: {e}"
            ) from e

    if not isinstance(radar_data, dict):
        raise RadarDataError(
            "radar_data debe ser un diccionario o un string JSON serializable a diccionario."
        )

    signal = radar_data.get("signal", "NEUTRAL")
    if not isinstance(signal, str):
        signal = "NEUTRAL"
    signal = signal.strip().upper()

    if budget_usd < 0:
        raise ValueError("budget_usd no puede ser negativo.")

    # Lógica de Capital según la señal
    if signal == "COMPRA":
        riesgo_pct = 0.75
        reserva_pct = 0.25
        mensaje = (
            "Señal de COMPRA detectada. Se prioriza el despliegue de capital "
            "hacia posiciones de riesgo para capturar la tendencia alcista, "
            "manteniendo una reserva mínima de liquidez táctica."
        )
    elif signal == "VENTA":
        riesgo_pct = 0.20
        reserva_pct = 0.80
        mensaje = (
            "Señal de VENTA detectada. Se privilegia el resguardo de capital, "
            "reduciendo la exposición a riesgo y fortaleciendo la posición de "
            "liquidez ante una posible corrección o tendencia bajista."
        )
    elif signal == "NEUTRAL":
        riesgo_pct = 0.50
        reserva_pct = 0.50
        mensaje = (
            "Señal NEUTRAL. Se mantiene una distribución equilibrada entre "
            "riesgo y liquidez a la espera de una confirmación direccional "
            "más clara del mercado."
        )
    else:
        riesgo_pct = 0.35
        reserva_pct = 0.65
        mensaje = (
            f"Señal no reconocida ('{signal}'). Se aplica una postura "
            "conservadora por defecto, favoreciendo la reserva de capital "
            "hasta obtener una señal clasificable."
        )

    capital_riesgo = round(budget_usd * riesgo_pct, 2)
    capital_reserva = round(budget_usd * reserva_pct, 2)

    return {
        "capital_inicial_usd": round(budget_usd, 2),
        "signal_detectada": signal,
        "asignacion": {
            "riesgo_usd": capital_riesgo,
            "reserva_usd": capital_reserva,
            "riesgo_pct": riesgo_pct,
            "reserva_pct": reserva_pct,
        },
        "mensaje_estrategico": mensaje,
    }