from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from build_graph import load_inputs, build_graph, PROC

TOL = 1e-9  


def coboundary(G, signed: bool = True):
    """Матрица кограницы δ (|E|×|V|) для весового скалярного пучка."""
    nodes = list(G.nodes())
    idx = {v: i for i, v in enumerate(nodes)}
    edges = list(G.edges(data=True))
    D = np.zeros((len(edges), len(nodes)))
    for k, (u, v, data) in enumerate(edges):
        rho = data["rho"]
        w = np.sqrt(abs(rho))
        s = np.sign(rho) if signed else 1.0
        if s == 0:
            s = 1.0
        D[k, idx[u]] = -w          # −F_{u⊴e}
        D[k, idx[v]] = s * w       #  F_{v⊴e}
    return D, nodes, edges


def cohomology(D):
    """Возвращает L0, спектр L0, dim H0, dim H1, ранг δ."""
    L0 = D.T @ D
    evals = np.linalg.eigvalsh(L0)
    evals = np.clip(evals, 0, None)
    dim_h0 = int(np.sum(evals < TOL))
    rank = np.linalg.matrix_rank(D, tol=1e-8)
    dim_h1 = D.shape[0] - rank          # |E| − rank δ
    return L0, evals, dim_h0, dim_h1, rank


def harmonic_sections(L0, nodes, k=3):
    """Ядро L0 (глобальные сечения) + вектор Фидлера (первый ненулевой мод)."""
    w, V = np.linalg.eigh(L0)
    w = np.clip(w, 0, None)
    kernel_cols = [i for i in range(len(w)) if w[i] < TOL]
    nonzero = [i for i in range(len(w)) if w[i] >= TOL]
    kernel = pd.DataFrame(V[:, kernel_cols],
                          index=nodes,
                          columns=[f"H0_{j}" for j in range(len(kernel_cols))])
    fiedler = None
    if nonzero:
        f_idx = nonzero[0]
        fiedler = pd.Series(V[:, f_idx], index=nodes, name="fiedler")
    return kernel, fiedler, w


def per_component_connectivity(G, evals_signed_fn):
    """Алгебраическая связность (λ2) весового пучка внутри каждой компоненты."""
    import networkx as nx
    rows = []
    for comp in nx.connected_components(G):
        sub = G.subgraph(comp)
        D, nodes, edges = coboundary(sub, signed=True)
        L0 = D.T @ D
        ev = np.clip(np.linalg.eigvalsh(L0), 0, None)
        nz = ev[ev >= TOL]
        lam2 = float(nz.min()) if len(nz) else 0.0
        rows.append(dict(size=len(comp), edges=sub.number_of_edges(),
                         lambda2=round(lam2, 4),
                         members=",".join(sorted(comp))))
    return pd.DataFrame(rows).sort_values("size", ascending=False).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", default="pearson")
    ap.add_argument("--threshold", type=float, default=0.30)
    args, _ = ap.parse_known_args()

    sim, meta, signed_measure = load_inputs(args.measure)
    G = build_graph(sim, meta, args.threshold, signed_measure)
    V, E = G.number_of_nodes(), G.number_of_edges()
    neg = sum(1 for _, _, d in G.edges(data=True) if d["rho"] < 0)

    # --- весовой скалярный пучок ---
    D, nodes, edges = coboundary(G, signed=True)
    L0, evals, h0, h1, rank = cohomology(D)
    kernel, fiedler, spec = harmonic_sections(L0, nodes)

    # --- постоянный пучок (для сравнения): restriction=1 => граф-лапласиан ---
    Dc, _, _ = coboundary(G, signed=False)
    # зануляем веса -> ±1 
    Dc = np.sign(Dc)
    _, evc, b0, b1, rankc = cohomology(Dc)

    print(f"Граф: |V|={V}, |E|={E}, отрицательных рёбер={neg} "
          f"(мера={args.measure}, порог={args.threshold})\n")
    print("Весовой скалярный пучок (restriction=√|ρ|·sign):")
    print(f"  dim H⁰ (глоб. сечения)   = {h0}")
    print(f"  dim H¹ (циклы/препятствия)= {h1}")
    print(f"  rank δ                    = {rank}")
    print(f"  Эйлерова χ = |V|−|E|      = {V - E}   (= dimH⁰ − dimH¹ = {h0 - h1})")
    nz = spec[spec >= TOL]
    print(f"  спектр L₀: λ_min≈0 (×{h0}), λ₂={nz.min():.4f}, λ_max={spec.max():.4f}")
    print(f"\nПостоянный пучок (чистая топология): β₀={b0}, β₁={b1} "
          f"(проверка β₁=|E|−|V|+β₀={E - V + b0})")

    conn = per_component_connectivity(G, None)
    print("\nАлгебраическая связность λ₂ по компонентам (весовой пучок):")
    print(conn.to_string(index=False))

    if fiedler is not None:
        print("\nВектор Фидлера (ось наименьшей связности), топ по модулю:")
        top = fiedler.reindex(fiedler.abs().sort_values(ascending=False).index).head(8)
        for t, val in top.items():
            print(f"  {t:6} {val:+.3f}  [{meta.loc[t,'country']}/{meta.loc[t,'sector']}]")

    # --- сохранение ---
    os.makedirs(PROC, exist_ok=True)
    suf = f"{args.measure}_{args.threshold}"
    pd.Series(evals, name="eigenvalue").to_csv(
        os.path.join(PROC, f"sheaf_spectrum_{suf}.csv"), index=False, encoding="utf-8")
    kernel.to_csv(os.path.join(PROC, f"sheaf_H0_{suf}.csv"), encoding="utf-8")
    if fiedler is not None:
        fiedler.to_csv(os.path.join(PROC, f"sheaf_fiedler_{suf}.csv"), encoding="utf-8")
    conn.to_csv(os.path.join(PROC, f"sheaf_components_{suf}.csv"), index=False, encoding="utf-8")

    # спектр: весовой vs постоянный
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(range(len(evals)), evals, "o-", color="#1f77b4", ms=4, label="весовой √|ρ|")
    ax.plot(range(len(evc)), np.sort(evc), "s--", color="#7f7f7f", ms=3,
            alpha=0.7, label="постоянный (±1)")
    ax.axhline(TOL, color="#d62728", ls=":", lw=1, label="уровень нуля")
    ax.set_xlabel("индекс собственного значения")
    ax.set_ylabel("λ (L₀)")
    ax.set_title(f"Спектр sheaf-лапласиана ({args.measure}, |ρ|≥{args.threshold})\n"
                 f"dim H⁰={h0}, dim H¹={h1}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PROC, f"sheaf_spectrum_{suf}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nСохранено в {PROC}/ (суффикс _{suf}):")
    print(f"  sheaf_spectrum_{suf}.csv/.png  — спектр L₀")
    print(f"  sheaf_H0_{suf}.csv             — базис глобальных сечений (ядро)")
    print(f"  sheaf_fiedler_{suf}.csv        — вектор Фидлера")
    print(f"  sheaf_components_{suf}.csv     — λ₂ по компонентам")


if __name__ == "__main__":
    main()
