
"""
avg_from_agent_csv.py
---------------------
Lee los CSVs de 'agent_data' de una corrida y crea un CSV con el error medio
entre agentes por instante de tiempo. Guarda: <run_dir>/avg_error.csv

Asume que cada archivo *_state_data.csv tiene una columna de tiempo ("Time")
y alguna columna de error. Detecta automáticamente entre:
    - 'Error', 'error', 'Tracking Error', 'tracking_error'
o, si no existen, intentará construir el módulo a partir de columnas
'Error X','Error Y','Error Z' (o 'Position Error X', etc.).

Uso (ejemplos):

    python avg_from_agent_csv.py /path/a/la/corrida
    python avg_from_agent_csv.py /path/a/la/corrida --error-col "Tracking Error"

"""
import argparse
import sys
from pathlib import Path
import re
import pandas as pd
import numpy as np

def _find_error_column(df: pd.DataFrame, prefer: str|None=None) -> str:
    if prefer and prefer in df.columns:
        return prefer
    # candidatos directos
    candidates = [c for c in df.columns if c.lower() in {"error","tracking error","tracking_error"}]
    if candidates:
        return candidates[0]
    # buscar por substring
    for c in df.columns:
        if "error" in c.lower():
            return c
    # vector por componentes
    # p.ej., 'Error X','Error Y','Error Z' o 'Position Error X', etc.
    def find_comp(prefixes):
        comps = []
        for p in prefixes:
            xs = [c for c in df.columns if re.fullmatch(fr"{re.escape(p)}\s*X", c, flags=re.I)]
            ys = [c for c in df.columns if re.fullmatch(fr"{re.escape(p)}\s*Y", c, flags=re.I)]
            zs = [c for c in df.columns if re.fullmatch(fr"{re.escape(p)}\s*Z", c, flags=re.I)]
            if xs and ys and zs:
                return xs[0], ys[0], zs[0]
        return None
    comps = find_comp(["Error", "Position Error", "Err", "E"])
    if comps:
        # Creamos una columna virtual de norma y la devolvemos
        ex, ey, ez = comps
        df["_temp_vector_error_norm_"] = np.sqrt(df[ex]**2 + df[ey]**2 + df[ez]**2)
        return "_temp_vector_error_norm_"
    raise ValueError("No se detectó ninguna columna de error en el CSV. Pasa --error-col explícitamente.")

def compute_avg(run_dir: Path, error_col: str|None=None) -> Path:
    agent_dir = run_dir / "simulation_data" / "agent_data"
    if not agent_dir.exists():
        raise FileNotFoundError(f"No existe {agent_dir}. ¿Apunta al directorio de la corrida correcta?")
    csvs = sorted(agent_dir.glob("*_state_data.csv"))
    if not csvs:
        raise FileNotFoundError(f"No se encontraron CSVs en {agent_dir}")
    # Leer y alinear por 'Time'
    dfs = []
    detected = None
    for p in csvs:
        df = pd.read_csv(p)
        if "Time" not in df.columns:
            # tolerar 'time' o 't'
            tcol = "Time" if "Time" in df.columns else ("time" if "time" in df.columns else None)
            if not tcol:
                raise ValueError(f"El archivo {p.name} no tiene columna de tiempo reconocible ('Time').")
        err_col_name = _find_error_column(df, prefer=error_col)
        detected = detected or err_col_name
        dfs.append(df[["Time", err_col_name]].rename(columns={err_col_name: p.stem}))
    # Merge por 'Time'
    merged = dfs[0]
    for d in dfs[1:]:
        merged = pd.merge_asof(merged.sort_values("Time"), d.sort_values("Time"), on="Time")
    # Media por fila (ignorando NaNs)
    value_cols = [c for c in merged.columns if c != "Time"]
    merged["avg_error"] = merged[value_cols].mean(axis=1, skipna=True)
    out = merged[["Time","avg_error"]].copy()
    out_path = run_dir / "avg_error.csv"
    out.to_csv(out_path, index=False)
    print(f"[OK] Guardado promedio en {out_path} (columna error detectada: {detected})")
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--error-col", type=str, default=None, help="Nombre exacto de la columna de error si quieres forzarlo")
    args = ap.parse_args()
    compute_avg(args.run_dir, args.error_col)

if __name__ == "__main__":
    main()
