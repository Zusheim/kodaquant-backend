"""
purify_scalers.py
==================
Ejecutar UNA VEZ, desde la raíz del proyecto (mismo nivel que main.py):

    python purify_scalers.py

Objetivo único: eliminar el `InconsistentVersionWarning` de scikit-learn
    "Trying to unpickle estimator StandardScaler from version 1.6.1 when
    using version 1.4.2"
    "Trying to unpickle estimator MinMaxScaler from version 1.6.1 when
    using version 1.4.2"

re-serializando `services/scalers.pkl` con la firma binaria NATIVA de la
versión de scikit-learn instalada en este venv (1.4.2, LTS Lock). Es un
load + dump inmediato — joblib inyecta el `__getstate__`/metadata de
versión ACTUALES al guardar, así que la próxima carga no dispara ningún
warning de compatibilidad. Cero reentrenamiento, cero alteración de
valores: los `StandardScaler`/`MinMaxScaler` (mean_, scale_, data_min_,
data_max_, etc.) quedan bit-a-bit idénticos, solo cambia el metadato de
versión embebido por pickle.
"""

import os
import warnings

# Debe importarse y suprimirse ANTES de `joblib.load()`, porque el warning
# se dispara en `BaseEstimator.__setstate__` durante la deserialización, no
# al importar el módulo.
from sklearn.exceptions import InconsistentVersionWarning

import joblib

# Ruta ANCLADA a la ubicación de este script (se ejecuta desde la raíz del
# proyecto; `services/` vive un nivel abajo, tal cual el árbol de directorio
# provisto) — nunca al cwd del proceso que invoca el script.
_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SCALERS_PATH = os.path.join(_ROOT_DIR, "services", "scalers.pkl")

# Claves mínimas que `quanti_engine.py` espera encontrar en el bundle
# (`feature_scalers`, `target_scalers`, `asset_to_id`) — se valida ANTES de
# sobreescribir el archivo, para no hornear un dump corrupto o vacío sobre
# el único artefacto de scalers del proyecto.
_REQUIRED_KEYS = {"feature_scalers", "target_scalers", "asset_to_id", "lookback", "macro_tickers"}


def purify_scalers(path: str = SCALERS_PATH) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el bundle de scalers en: {path}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=InconsistentVersionWarning)
        bundle = joblib.load(path)

    if not isinstance(bundle, dict) or not _REQUIRED_KEYS.issubset(bundle.keys()):
        missing = _REQUIRED_KEYS - (bundle.keys() if isinstance(bundle, dict) else set())
        raise ValueError(
            "El objeto cargado no tiene la forma esperada del bundle de "
            f"KodaQuant — faltan claves: {sorted(missing)}. Se aborta el dump "
            "para no sobreescribir services/scalers.pkl con una estructura desconocida."
        )

    # Backup defensivo antes de sobreescribir — un solo archivo, cero riesgo.
    backup_path = path + ".bak"
    if not os.path.exists(backup_path):
        with open(path, "rb") as src, open(backup_path, "wb") as dst:
            dst.write(src.read())

    joblib.dump(bundle, path)

    n_features = len(bundle["feature_scalers"])
    n_targets = len(bundle["target_scalers"])
    print(
        f"✅ scalers.pkl purificado y re-firmado con scikit-learn instalado.\n"
        f"   Ruta: {path}\n"
        f"   Backup preservado en: {backup_path}\n"
        f"   feature_scalers: {n_features} tickers | target_scalers: {n_targets} tickers\n"
        f"   Vuelve a levantar Uvicorn — el InconsistentVersionWarning no debería reaparecer."
    )


if __name__ == "__main__":
    purify_scalers()