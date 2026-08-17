from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_graph import load_inputs, build_graph_knn, PROC
from build_sheaf import coboundary, TOL
from node_classify import (node_features, cls_features, adjacency,
                           cls_label_spreading, cls_sheaf)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# --------------------------- шаг 1: локальный PCA ----------------------------
def tangent_frames(R: pd.DataFrame, G, tickers: list[str], d: int) -> dict:
    """O_i ∈ R^{T×d}: ортонормированный касательный базис из соседей по графу."""
    frames = {}
    X = {t: R[t].to_numpy(float) for t in tickers}   # x_i ∈ R^T
    for i in tickers:
        nbrs = list(G.neighbors(i))
        if len(nbrs) < d:                            # union-kNN гарантирует deg≥k
            nbrs = nbrs + [i]                        # страховка (почти не срабатывает)
        B = np.stack([X[j] - X[i] for j in nbrs])    # (deg, T)
        # top-d правых сингулярных векторов B (направления в R^T)
        _, _, Vt = np.linalg.svd(B, full_matrices=False)
        O = Vt[:d].T                                 # (T, d), столбцы ортонормированы
        frames[i] = O
    return frames


# ----------------------- шаг 2: транспорт -------------------------
def procrustes(Oi: np.ndarray, Oj: np.ndarray) -> np.ndarray:
    """O_ij ∈ O(d): ортогональное выравнивание касательных базисов через SVD."""
    M = Oi.T @ Oj                                    # (d, d)
    U, _, Vt = np.linalg.svd(M)
    return U @ Vt


# ------------------- шаг 3: сборка Connection Laplacian ----------------------
def connection_laplacian(G, tickers: list[str], frames: dict, d: int):
    """Блочный connection Laplacian L (Nd×Nd) и степени deg (по узлам)."""
    n = len(tickers)
    idx = {t: i for i, t in enumerate(tickers)}
    L = np.zeros((n * d, n * d))
    deg = np.zeros(n)

    def rng(a):
        return slice(a * d, (a + 1) * d)

    for a, b, data in G.edges(data=True):
        i, j = idx[a], idx[b]
        w = float(data["weight"])                    # |ρ|
        Oij = procrustes(frames[a], frames[b])       # O(d)
        L[rng(i), rng(j)] += -w * Oij
        L[rng(j), rng(i)] += -w * Oij.T
        deg[i] += w
        deg[j] += w
    for i in range(n):
        L[rng(i), rng(i)] = deg[i] * np.eye(d)
    return L, deg


def cls_connection_harmonic(L: np.ndarray, y: np.ndarray, d: int):
    """Классификация гармоническим продолжением по connection Laplacian.

    Каждый класс c поднимается в стебель как единичный вектор u=1_d/√d. Для held-out
    узла i решаем L_UU F_U = −L_UL Y_L (U={i}); голоса соседей класса c приходят
    ПОВЁРНУТЫМИ транспортом O_ij и складываются в блоке F[:,c]. Предсказание —
    argmax_c ‖F[:,c]‖ (класс с наиболее СОГЛАСОВАННЫМ транспортированным голосом)."""
    classes = np.unique(y)
    cidx = {c: i for i, c in enumerate(classes)}
    n, C = len(y), len(classes)
    u = np.ones(d) / np.sqrt(d)
    pred = np.empty_like(y)
    for i in range(n):
        others = [j for j in range(n) if j != i]
        ri = slice(i * d, (i + 1) * d)
        Luu = L[ri, ri]                              # d×d = deg_i·I
        Lul = np.concatenate([L[ri, j * d:(j + 1) * d] for j in others], axis=1)
        Yl = np.zeros((d * len(others), C))
        for r, j in enumerate(others):
            Yl[r * d:(r + 1) * d, cidx[y[j]]] = u
        F = -np.linalg.solve(Luu, Lul @ Yl)          # d×C
        pred[i] = classes[int(np.argmax(np.linalg.norm(F, axis=0)))]
    return accuracy_score(y, pred), f1_score(y, pred, average="macro"), pred


def spectrum_summary(evals: np.ndarray, d: int, beta0: int, label: str):
    evals = np.clip(evals, 0, None)
    ker = int(np.sum(evals < TOL))
    nz = evals[evals >= TOL]
    lam2 = float(nz.min()) if len(nz) else 0.0
    print(f"  {label:26} dim ker={ker:3}  (тривиальная голономия ⇒ d·β₀={d*beta0})"
          f"  λ₂={lam2:.4f}  λ_max={evals.max():.4f}")
    return ker, lam2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", default="pearson")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--dim", type=int, default=2, help="размерность стебля d")
    ap.add_argument("--min-count", type=int, default=3)
    args, _ = ap.parse_known_args()
    d = args.dim

    import networkx as nx
    sim, meta, signed = load_inputs(args.measure)
    R = pd.read_csv(os.path.join(PROC, "returns.csv"), index_col=0, parse_dates=True)

    # ---------- спектр на ПОЛНОМ рынке (24 узла) ----------
    allt = [t for t in sim.columns if t in meta.index]
    Gf = build_graph_knn(sim.loc[allt, allt], meta, k=args.k, signed=signed, mutual=False)
    beta0 = nx.number_connected_components(Gf)
    frames_f = tangent_frames(R, Gf, allt, d)
    Lf, _ = connection_laplacian(Gf, allt, frames_f, d)
    ev_conn = np.linalg.eigvalsh(Lf)
    # скалярный пучок на том же графе — для сравнения
    Ds, _, _ = coboundary(Gf, signed=True)
    ev_scalar = np.linalg.eigvalsh(Ds.T @ Ds)

    print(f"Векторный пучок / Connection Laplacian (мера={args.measure}, "
          f"kNN union k={args.k}, d={d})")
    print(f"Граф: |V|={Gf.number_of_nodes()}, |E|={Gf.number_of_edges()}, β₀={beta0}\n")
    print("Спектр:")
    spectrum_summary(ev_scalar, 1, beta0, "скалярный пучок (d=1)")
    ker_c, _ = spectrum_summary(ev_conn, d, beta0, f"connection (d={d})")
    frustration = d * beta0 - ker_c
    if frustration > 0:
        print(f"  ⇒ голономия НЕТРИВИАЛЬНА: ядро на {frustration} меньше d·β₀ "
              f"(транспорт O(d) фрустрирован по циклам).")
    else:
        print(f"  ⇒ голономия тривиальна: ядро = d·β₀ (транспорт согласован).")

    # ---------- классификация на подвыборке (сектора ≥ min_count) ----------
    counts = meta["sector"].value_counts()
    classes = sorted(counts[counts >= args.min_count].index)
    kept = [t for t in allt if meta.loc[t, "sector"] in classes]
    y = np.array([meta.loc[t, "sector"] for t in kept])

    Gk = build_graph_knn(sim.loc[kept, kept], meta, k=args.k, signed=signed, mutual=False)
    frames_k = tangent_frames(R, Gk, kept, d)
    Lk, _ = connection_laplacian(Gk, kept, frames_k, d)

    # скалярные методы (те же, что node_classify) на том же подграфе
    A_scalar = adjacency(Gk, kept)
    Dk, nodes, _ = coboundary(Gk, signed=True)
    order = {t: i for i, t in enumerate(kept)}
    P = np.zeros((len(kept), len(nodes)))
    for j, nn in enumerate(nodes):
        P[order[nn], j] = 1.0
    L0k = P @ (Dk.T @ Dk) @ P.T
    Xf = node_features(R, kept)

    res = {
        "A. признаки (без графа)": cls_features(Xf, y),
        "B. label spreading (|ρ|)": cls_label_spreading(A_scalar, y),
        "C. скалярный пучок": cls_sheaf(L0k, y),
        f"D. connection harmonic (d={d})": cls_connection_harmonic(Lk, y, d),
    }
    maj = counts[classes].max() / len(kept)
    print(f"\nКлассификация сектора (узлов={len(kept)}, классы={len(classes)}, "
          f"мажоритарный baseline={maj:.3f}):")
    print(f"{'метод':34}{'accuracy':>10}{'macro-F1':>10}")
    for name, (acc, f1, _) in res.items():
        print(f"{name:34}{acc:10.3f}{f1:10.3f}")

    # матрица ошибок connection-harmonic
    _, _, pred = res[f"D. connection harmonic (d={d})"]
    cm = confusion_matrix(y, pred, labels=classes)
    print(f"\nМатрица ошибок (connection harmonic, строки=истина):")
    print("        " + "".join(f"{c[:5]:>7}" for c in classes))
    for i, c in enumerate(classes):
        print(f"{c[:7]:8}" + "".join(f"{cm[i, j]:7}" for j in range(len(classes))))

    # ---------- сохранение: спектр + график ----------
    suf = f"{args.measure}_knn{args.k}_d{d}"
    pd.DataFrame({"connection": np.sort(ev_conn)}).to_csv(
        os.path.join(PROC, f"connection_spectrum_{suf}.csv"), index=False, encoding="utf-8")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(len(ev_conn)), np.sort(ev_conn), "o-", ms=3, color="#1f77b4",
            label=f"connection d={d}")
    ax.plot(np.linspace(0, len(ev_conn) - 1, len(ev_scalar)), np.sort(ev_scalar),
            "s--", ms=3, color="#7f7f7f", alpha=0.7, label="скалярный d=1")
    ax.axhline(TOL, color="#d62728", ls=":", lw=1, label="уровень нуля")
    ax.set_xlabel("индекс собственного значения")
    ax.set_ylabel("λ")
    ax.set_title(f"Спектр Connection Laplacian ({args.measure}, kNN k={args.k}, d={d})\n"
                 f"dim ker={ker_c} (d·β₀={d*beta0}, фрустрация={frustration})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PROC, f"connection_spectrum_{suf}.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)

    out = pd.DataFrame({"ticker": kept, "true": y})
    for name, (_, _, pred) in res.items():
        out[name.split(".")[0]] = pred
    out.to_csv(os.path.join(PROC, f"connection_classify_{suf}.csv"),
               index=False, encoding="utf-8")
    print(f"\nСохранено: connection_spectrum_{suf}.csv/.png, connection_classify_{suf}.csv")


if __name__ == "__main__":
    main()
