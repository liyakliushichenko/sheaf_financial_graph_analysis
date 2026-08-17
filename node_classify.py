from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from build_graph import load_inputs, build_graph_knn, PROC
from build_sheaf import coboundary


# ----------------------------- признаки (метод A) -----------------------------
def node_features(R: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Статистики распределения доходностей по каждому активу (граф НЕ используется)."""
    mkt = R.mean(axis=1)                       # равновзвешенный «рынок»
    mkt_var = float(mkt.var())
    rows = {}
    for t in tickers:
        x = R[t].dropna()
        mu, sd = float(x.mean()), float(x.std())
        dn = x[x < 0]
        beta = float(np.cov(x, mkt.loc[x.index])[0, 1] / mkt_var) if mkt_var > 0 else 0.0
        rows[t] = dict(
            vol=sd * np.sqrt(252),
            mean=mu,
            skew=float(x.skew()),
            kurt=float(x.kurt()),
            downside=float(dn.std()) if len(dn) > 1 else 0.0,
            autocorr=float(x.autocorr(lag=1)) if len(x) > 2 else 0.0,
            beta=beta,
            sharpe=mu / sd * np.sqrt(252) if sd > 0 else 0.0,
        )
    return pd.DataFrame(rows).T.reindex(tickers)


def cls_features(X: pd.DataFrame, y: np.ndarray) -> tuple[float, float, np.ndarray]:
    loo = LeaveOneOut()
    pred = np.empty_like(y)
    Xv = X.to_numpy(float)
    for tr, te in loo.split(Xv):
        sc = StandardScaler().fit(Xv[tr])
        clf = LogisticRegression(max_iter=2000, C=0.5)
        clf.fit(sc.transform(Xv[tr]), y[tr])
        pred[te] = clf.predict(sc.transform(Xv[te]))
    return accuracy_score(y, pred), f1_score(y, pred, average="macro"), pred


# ------------------------- граф: матрица смежности ---------------------------
def adjacency(G, tickers: list[str]) -> np.ndarray:
    idx = {t: i for i, t in enumerate(tickers)}
    A = np.zeros((len(tickers), len(tickers)))
    for a, b, d in G.edges(data=True):
        w = float(d["weight"])                 # |ρ|
        A[idx[a], idx[b]] = A[idx[b], idx[a]] = w
    return A


# --------------------- метод B: label spreading (Zhou) -----------------------
def cls_label_spreading(A: np.ndarray, y: np.ndarray, alpha: float = 0.99):
    classes = np.unique(y)
    cidx = {c: i for i, c in enumerate(classes)}
    n, C = len(y), len(classes)
    deg = A.sum(1)
    dinv = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    S = (A * dinv[:, None]) * dinv[None, :]     # D^{-1/2} A D^{-1/2}
    M = np.linalg.inv(np.eye(n) - alpha * S)    # (I − αS)^{-1}
    pred = np.empty_like(y)
    for i in range(n):                          # leave-one-out
        Y = np.zeros((n, C))
        for j in range(n):
            if j != i:
                Y[j, cidx[y[j]]] = 1.0
        F = M @ Y
        pred[i] = classes[int(np.argmax(F[i]))]
    return accuracy_score(y, pred), f1_score(y, pred, average="macro"), pred


# ------------------- метод C: гармоническое продолжение по пучку --------------
def cls_sheaf(L0: np.ndarray, y: np.ndarray):
    """L0 — sheaf-лапласиан на подграфе (порядок узлов = порядок y).
    Для held-out узла i решаем L0_UU f_U = −L0_UL Y_L при U={i}."""
    classes = np.unique(y)
    cidx = {c: i for i, c in enumerate(classes)}
    n, C = len(y), len(classes)
    pred = np.empty_like(y)
    for i in range(n):
        L = [j for j in range(n) if j != i]
        Y = np.zeros((len(L), C))
        for r, j in enumerate(L):
            Y[r, cidx[y[j]]] = 1.0
        diag = L0[i, i]
        if diag <= 0:                           # изолированный узел -> приор
            pred[i] = classes[int(np.argmax(Y.sum(0)))] if len(Y) else classes[0]
            continue
        f = (-(L0[i, L]) @ Y) / diag            # f_i^c
        pred[i] = classes[int(np.argmax(f))]
    return accuracy_score(y, pred), f1_score(y, pred, average="macro"), pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", default="pearson")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--min-count", type=int, default=3,
                    help="минимум членов сектора, чтобы включить его как класс")
    args, _ = ap.parse_known_args()

    sim, meta, signed = load_inputs(args.measure)
    R = pd.read_csv(os.path.join(PROC, "returns.csv"), index_col=0, parse_dates=True)

    # --- классы: сектора с достаточным числом членов ---
    counts = meta["sector"].value_counts()
    classes = sorted(counts[counts >= args.min_count].index)
    kept = [t for t in sim.columns if t in meta.index and meta.loc[t, "sector"] in classes]
    y = np.array([meta.loc[t, "sector"] for t in kept])

    print(f"Задача: предсказать сектор акции (мера={args.measure}, kNN union k={args.k})")
    print(f"Классы (≥{args.min_count} членов): " +
          ", ".join(f"{c}={int(counts[c])}" for c in classes))
    print(f"Узлов в выборке: {len(kept)}  |  Baseline (мажоритарный): "
          f"{counts[classes].max() / len(kept):.3f}\n")

    # --- структура: kNN-union среди отобранных узлов ---
    sub = sim.loc[kept, kept]
    G = build_graph_knn(sub, meta, k=args.k, signed=signed, mutual=False)
    A = adjacency(G, kept)
    D, nodes, _ = coboundary(G, signed=True)
    order = {t: i for i, t in enumerate(kept)}
    P = np.zeros((len(kept), len(nodes)))       # перестановка nodes -> порядок kept
    for j, nn in enumerate(nodes):
        P[order[nn], j] = 1.0
    L0 = P @ (D.T @ D) @ P.T

    # --- признаки (метод A) ---
    X = node_features(R, kept)

    results = {}
    results["A_features"] = cls_features(X, y)
    results["B_label_spreading"] = cls_label_spreading(A, y)
    results["C_sheaf"] = cls_sheaf(L0, y)

    print(f"{'метод':22}{'accuracy':>10}{'macro-F1':>10}")
    labels = {"A_features": "A. признаки (без графа)",
              "B_label_spreading": "B. label spreading",
              "C_sheaf": "C. sheaf-диффузия"}
    for k, (acc, f1, _) in results.items():
        print(f"{labels[k]:22}{acc:10.3f}{f1:10.3f}")

    # матрица ошибок лучшего графового метода
    best = max(("B_label_spreading", "C_sheaf"), key=lambda k: results[k][0])
    acc, f1, pred = results[best]
    cm = confusion_matrix(y, pred, labels=classes)
    print(f"\nМатрица ошибок ({labels[best]}, строки=истина, столбцы=предсказание):")
    print("        " + "".join(f"{c[:5]:>7}" for c in classes))
    for i, c in enumerate(classes):
        print(f"{c[:7]:8}" + "".join(f"{cm[i, j]:7}" for j in range(len(classes))))

    # сохранить пофакторный прогноз
    out = pd.DataFrame({"ticker": kept, "country": [meta.loc[t, "country"] for t in kept],
                        "true": y})
    for k, (_, _, pred) in results.items():
        out[k] = pred
    suf = f"{args.measure}_knn{args.k}"
    out.to_csv(os.path.join(PROC, f"node_classify_{suf}.csv"), index=False, encoding="utf-8")
    print(f"\nСохранено: processed/node_classify_{suf}.csv "
          f"(истина + прогнозы всех методов по узлам)")


if __name__ == "__main__":
    main()
