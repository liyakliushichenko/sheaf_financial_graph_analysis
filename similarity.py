from __future__ import annotations
import os
import numpy as np
import pandas as pd

try:
    BASE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE = os.getcwd()
PROC = os.path.join(BASE, "processed")


def load_returns() -> pd.DataFrame:
    return pd.read_csv(os.path.join(PROC, "returns.csv"), index_col=0, parse_dates=True)


def pearson(R: pd.DataFrame) -> pd.DataFrame:
    return R.corr(method="pearson")


def spearman(R: pd.DataFrame) -> pd.DataFrame:
    return R.corr(method="spearman")


def cosine(R: pd.DataFrame) -> pd.DataFrame:
    X = R.to_numpy(dtype=float)
    norm = np.linalg.norm(X, axis=0)
    norm[norm == 0] = 1.0
    Xn = X / norm
    S = Xn.T @ Xn
    np.fill_diagonal(S, 1.0)
    return pd.DataFrame(S, index=R.columns, columns=R.columns)


def distance_correlation(R: pd.DataFrame) -> pd.DataFrame:
    X = R.to_numpy(dtype=float)
    T, p = X.shape
    # для каждой переменной — двойно-центрированная матрица |x_i - x_j|
    A = []
    for k in range(p):
        x = X[:, k]
        D = np.abs(x[:, None] - x[None, :])            # (T, T)
        Dr = D.mean(axis=0, keepdims=True)
        Dc = D.mean(axis=1, keepdims=True)
        Ak = (D - Dr - Dc + D.mean()).astype(np.float32)
        A.append(Ak)
    dcov2 = np.zeros((p, p))
    for k in range(p):
        for l in range(k, p):
            v = float(np.mean(A[k].astype(np.float64) * A[l].astype(np.float64)))
            dcov2[k, l] = dcov2[l, k] = v
    dvar2 = np.diag(dcov2).copy()
    dcor = np.zeros((p, p))
    for k in range(p):
        for l in range(p):
            denom = np.sqrt(dvar2[k] * dvar2[l])
            if denom > 0:
                dcor[k, l] = np.sqrt(max(dcov2[k, l], 0.0) / denom)
    np.fill_diagonal(dcor, 1.0)
    return pd.DataFrame(dcor, index=R.columns, columns=R.columns)


def partial_correlation(R: pd.DataFrame) -> pd.DataFrame:
    """Частная корреляция через матрицу точности Θ = C^{-1} """
    C = R.corr(method="pearson").to_numpy(float)
    p = C.shape[0]
    ridge = 1e-6
    try:
        Theta = np.linalg.inv(C + ridge * np.eye(p))
    except np.linalg.LinAlgError:
        Theta = np.linalg.pinv(C)
    dinv = 1.0 / np.sqrt(np.clip(np.diag(Theta), 1e-12, None))
    P = -Theta * dinv[:, None] * dinv[None, :]
    np.fill_diagonal(P, 1.0)
    return pd.DataFrame(P, index=R.columns, columns=R.columns)


MEASURES = {
    "pearson": pearson,
    "spearman": spearman,
    "cosine": cosine,
    "dcor": distance_correlation,
    "partial": partial_correlation,
}


def cross_market_summary(S: pd.DataFrame, meta: pd.DataFrame):
    ru = [t for t in S.columns if meta.loc[t, "country"] == "RU"]
    us = [t for t in S.columns if meta.loc[t, "country"] == "US"]

    def block(a, b):
        sub = S.loc[a, b].to_numpy()
        if a is b:
            iu = np.triu_indices(len(a), k=1)
            return float(sub[iu].mean())
        return float(sub.mean())

    return block(ru, ru), block(us, us), block(ru, us)


def main():
    R = load_returns()
    meta = pd.read_csv(os.path.join(PROC, "meta.csv"), index_col=0)
    print(f"Доходностей: {R.shape[0]} дней × {R.shape[1]} активов\n")

    print(f"{'measure':10}{'range':>14}{'RU-RU':>9}{'US-US':>9}{'RU-US':>9}")
    for name, fn in MEASURES.items():
        S = fn(R).reindex(index=R.columns, columns=R.columns)
        S.to_csv(os.path.join(PROC, f"sim_{name}.csv"), encoding="utf-8")
        off = S.to_numpy()[~np.eye(len(S), dtype=bool)]
        rr, uu, ru = cross_market_summary(S, meta)
        print(f"{name:10}[{off.min():+.2f}, {off.max():+.2f}]".ljust(24)
              + f"{rr:+9.3f}{uu:+9.3f}{ru:+9.3f}")

    print(f"\nСохранено в {PROC}/: sim_pearson.csv, sim_spearman.csv, "
          f"sim_cosine.csv, sim_dcor.csv, sim_partial.csv")


if __name__ == "__main__":
    main()
