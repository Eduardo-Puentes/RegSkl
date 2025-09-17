# report_plots.py
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score


def build_pipeline(model_name: str, alpha: float = 1.0, standardize: bool = False):
    if model_name == "linear":
        model = LinearRegression()
    elif model_name == "ridge":
        model = Ridge(alpha=alpha)
    elif model_name == "lasso":
        model = Lasso(alpha=alpha, max_iter=10000)
    else:
        raise ValueError("model_name must be one of: linear, ridge, lasso")

    steps = []
    if standardize:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def split_train_val_test(X, y, test_size=0.2, val_size=0.2, seed=42):
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    val_rel = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_rel, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def learning_curve_MSE(pipe, X_train, y_train, X_val, y_val, train_fracs):
    mse_train = []
    mse_val = []
    n = X_train.shape[0]
    for frac in train_fracs:
        m = max(10, int(n * frac))
        X_sub = X_train[:m]
        y_sub = y_train[:m]
        pipe.fit(X_sub, y_sub)
        y_hat_tr = pipe.predict(X_sub)
        y_hat_va = pipe.predict(X_val)
        mse_train.append(mean_squared_error(y_sub, y_hat_tr))
        mse_val.append(mean_squared_error(y_val, y_hat_va))
    return np.array(mse_train), np.array(mse_val)


def plot_learning_curve(train_fracs, mse_train, mse_val, out_path):
    plt.figure()
    plt.plot(train_fracs, mse_train, marker="o", label="Train MSE")
    plt.plot(train_fracs, mse_val, marker="s", label="Validation MSE")
    plt.xlabel("Fracción de datos de entrenamiento")
    plt.ylabel("MSE")
    plt.title("Curvas de aprendizaje (MSE)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_ytrue_vs_ypred(y_true, y_pred, out_path, title="y_real vs y_pred (Test)"):
    plt.figure()
    plt.scatter(y_true, y_pred, alpha=0.5)
    minv = min(np.min(y_true), np.min(y_pred))
    maxv = max(np.max(y_true), np.max(y_pred))
    plt.plot([minv, maxv], [minv, maxv], linewidth=2)
    plt.xlabel("y real")
    plt.ylabel("y predicho")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_model_compare_r2(model_names, r2_vals, out_path):
    plt.figure()
    x = np.arange(len(model_names))
    plt.bar(x, r2_vals)
    plt.xticks(x, model_names)
    plt.ylabel("R² (validation)")
    plt.title("Comparativa de R² por modelo (Validation)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_alpha_vs_r2(alphas, r2_vals, out_path, title="R² (validation) vs alpha (Ridge)"):
    plt.figure()
    plt.plot(alphas, r2_vals, marker="o")
    plt.xscale("log")
    plt.xlabel("alpha (log)")
    plt.ylabel("R² (validation)")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Generar gráficas para el reporte (California Housing)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--val-size", type=float, default=0.2)
    ap.add_argument("--std", action="store_true", help="Estandarizar para Ridge/Lasso")
    ap.add_argument("--ridge-alphas", type=str, default="0.1,1.0,10.0")
    ap.add_argument("--lasso-alphas", type=str, default="0.1,0.5,1.0")
    ap.add_argument("--outdir", type=str, default="img")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    data = fetch_california_housing()
    X, y = data.data, data.target

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        X, y, test_size=args.test_size, val_size=args.val_size, seed=args.seed
    )

    pipe_linear = build_pipeline("linear", standardize=False)
    rng = np.random.default_rng(args.seed)
    idx = np.arange(X_train.shape[0])
    rng.shuffle(idx)
    X_train = X_train[idx]
    y_train = y_train[idx]
    train_fracs = np.linspace(0.1, 1.0, 10)
    mse_tr, mse_va = learning_curve_MSE(pipe_linear, X_train, y_train, X_val, y_val, train_fracs)
    plot_learning_curve(train_fracs, mse_tr, mse_va, os.path.join(args.outdir, "learning_curve.png"))

    models = [("Linear", "linear", None),
              ("Ridge (α=1.0)", "ridge", 1.0),
              ("Lasso (α=0.5)", "lasso", 0.5)]
    r2_val_list = []
    for label, mname, alpha in models:
        pipe = build_pipeline(mname, alpha=alpha if alpha is not None else 1.0,
                              standardize=(args.std if mname in ("ridge", "lasso") else False))
        pipe.fit(X_train, y_train)
        r2_v = r2_score(y_val, pipe.predict(X_val))
        r2_val_list.append(r2_v)
    plot_model_compare_r2([m[0] for m in models], r2_val_list, os.path.join(args.outdir, "model_compare_r2.png"))

    ridge_alphas = [float(a.strip()) for a in args.ridge_alphas.split(",") if a.strip()]
    r2_ridge = []
    for a in ridge_alphas:
        pipe = build_pipeline("ridge", alpha=a, standardize=args.std)
        pipe.fit(X_train, y_train)
        r2_ridge.append(r2_score(y_val, pipe.predict(X_val)))
    plot_alpha_vs_r2(ridge_alphas, r2_ridge, os.path.join(args.outdir, "alpha_vs_r2_ridge.png"))

    best_idx = int(np.argmax(r2_val_list))
    best_label, best_name, best_alpha = models[best_idx]
    best_pipe = build_pipeline(best_name, alpha=best_alpha if best_alpha is not None else 1.0,
                               standardize=(args.std if best_name in ("ridge", "lasso") else False))
    best_pipe.fit(X_train, y_train)
    y_pred_test = best_pipe.predict(X_test)
    plot_ytrue_vs_ypred(y_test, y_pred_test, os.path.join(args.outdir, "y_true_vs_pred_test.png"),
                        title=f"y_real vs y_pred (Test) – {best_label}")

    print("=== Resumen (Validation R²) ===")
    for (label, _, _), r2v in zip(models, r2_val_list):
        print(f"{label}: R²(val) = {r2v:.4f}")
    print("Mejor en validation:", models[best_idx][0])
    print("PNG guardados en:", os.path.abspath(args.outdir))


if __name__ == "__main__":
    main()
