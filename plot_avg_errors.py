
"""
plot_avg_errors.py
------------------
Lee varios avg_error.csv (uno por valor de k) y dibuja todas las curvas en un mismo gráfico.
También permite, si no existen, calcular los promedios llamando al script de arriba.

Estructuras admitidas:
- Pasar una lista de directorios que contienen 'avg_error.csv'
- O pasar un directorio padre que contenga subcarpetas por corrida (p.ej., k_0.1, k_1, k_5)

Uso:
    python plot_avg_errors.py --runs runs/k_0.1 runs/k_1 runs/k_5
    python plot_avg_errors.py --parent runs

Guarda:
    - plot_avg_errors.png (figura)
    - all_avg_errors.csv (tabla ancha con columnas por k)
"""
import argparse
from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt

def _label_from_dir(d: Path) -> str:
    # intenta extraer 'k=...' del nombre
    m = re.search(r"k[_=]([0-9\.eE+-]+)", d.name)
    return m.group(1) if m else d.name

def _load_or_warn(run: Path) -> tuple[str, pd.DataFrame]:
    p = run / "avg_error.csv"
    if not p.exists():
        # intentar autogenerar
        try:
            from avg_from_agent_csv import compute_avg
            p = compute_avg(run)
        except Exception as e:
            raise RuntimeError(f"No se pudo generar avg_error.csv en {run}: {e}")
    df = pd.read_csv(p)
    if "Time" not in df.columns or "avg_error" not in df.columns:
        raise ValueError(f"{p} no tiene columnas esperadas ('Time','avg_error').")
    return _label_from_dir(run), df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", type=Path, help="Lista de carpetas de corrida")
    ap.add_argument("--parent", type=Path, help="Carpeta padre que contiene subcarpetas por corrida")
    args = ap.parse_args()

    runs = args.runs or []
    if args.parent:
        runs.extend([d for d in args.parent.iterdir() if d.is_dir()])
    if not runs:
        raise SystemExit("Debes dar --runs ... o --parent CARPETA")

    series = []
    wide = None
    for r in sorted(runs, key=lambda p: p.name):
        label, df = _load_or_warn(r)
        series.append((label, df))
        if wide is None:
            wide = df.rename(columns={"avg_error": f"k={label}"})
        else:
            wide = wide.merge(df.rename(columns={"avg_error": f"k={label}"}), on="Time", how="outer")

    # Plot
    plt.figure(figsize=(8,5))
    for label, df in series:
        plt.plot(df["Time"], df["avg_error"], label=f"k={label}")
    # plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Time (s)")
    plt.ylabel("Average Tracking Error")
    # plt.title("Average agent error vs time (per k)")
    plt.legend()
    plt.tight_layout()
    out_fig = Path("plot_avg_errors.png")
    plt.savefig(out_fig, dpi=200)
    print(f"[OK] Guardada figura en {out_fig}")

    out_csv = Path("all_avg_errors.csv")
    if wide is not None:
        wide.sort_values("Time").to_csv(out_csv, index=False)
        print(f"[OK] Guardada tabla en {out_csv}")

if __name__ == "__main__":
    main()
