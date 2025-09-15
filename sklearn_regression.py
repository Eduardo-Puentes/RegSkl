import argparse
import csv
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score


def parse_args():
    p = argparse.ArgumentParser(description="Regresión con scikit-learn (CLI)")
    p.add_argument("--data", required=True, help="Ruta al CSV")
    p.add_argument("--target-index", type=int, default=-1, help="Índice de la columna objetivo (default: -1, última).")
    p.add_argument("--header", action="store_true", help="Indica si el CSV tiene encabezado (default: False).")
    p.add_argument("--test-size", type=float, default=0.2, help="Proporción del conjunto de prueba (default: 0.2).")
    p.add_argument("--seed", type=int, default=42, help="Semilla para el split (default: 42).")
    p.add_argument("--standardize", action="store_true", help="Estandarizar X con StandardScaler (default: False).")
    p.add_argument("--model", choices=["linear", "ridge", "lasso"], default="linear",
                   help="Modelo a usar (default: linear).")
    p.add_argument("--alpha", type=float, default=1.0, help="Alpha para Ridge/Lasso (default: 1.0).")
    p.add_argument("--predict", type=str, default=None, help='Vector de features para predecir, ej: "120,3,2".')
    return p.parse_args()


def read_csv(path: str, header: bool, target_idx: int):
    with open(path, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError("CSV vacío.")
    start = 1 if header else 0
    if header:
        ncols = len(rows[0])
    else:
        ncols = len(rows[0])
    if target_idx < 0:
        target_idx = ncols + target_idx

    data = []
    for r in rows[start:]:
        if len(r) != ncols:
            raise ValueError("Filas con distinto número de columnas.")
        data.append(r)
    data = np.array(data, dtype=float)
    y = data[:, target_idx]
    X = np.delete(data, target_idx, axis=1)
    return X, y


def build_pipeline(model_name: str, standardize: bool, alpha: float):
    if model_name == "linear":
        model = LinearRegression()
    elif model_name == "ridge":
        model = Ridge(alpha=alpha, random_state=None)
    else:
        model = Lasso(alpha=alpha, random_state=None, max_iter=10000)

    steps = []
    if standardize:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def main():
    args = parse_args()

    X, y = read_csv(args.data, header=args.header, target_idx=args.target_index)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )

    pipe = build_pipeline(args.model, args.standardize, args.alpha)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Modelo: {args.model} (standardize={args.standardize}, alpha={args.alpha})")
    print(f"MSE (test): {mse:.6f}")
    print(f"R^2 (test): {r2:.6f}")

    if args.predict:
        vec = np.array([float(v.strip()) for v in args.predict.split(",")], dtype=float).reshape(1, -1)
        y_one = pipe.predict(vec)[0]
        print(f"Predicción para [{args.predict}] -> {y_one:.6f}")


if __name__ == "__main__":
    main()
