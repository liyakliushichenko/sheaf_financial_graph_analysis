from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    BASE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE = os.getcwd()
PROC = os.path.join(BASE, "processed")

SECTOR_COLORS = {
    "Energy": "#8c564b", "Materials": "#7f7f7f", "Financials": "#1f77b4",
    "Consumer": "#ff7f0e", "Telecom": "#9467bd", "Tech": "#2ca02c",
    "Healthcare": "#e377c2", "Utilities": "#17becf", "Industrials": "#bcbd22",
}



MEASURE_FILES = {
    "pearson": ("sim_pearson.csv", True),
    "spearman": ("sim_spearman.csv", True),
    "cosine": ("sim_cosine.csv", True),
    "dcor": ("sim_dcor.csv", False),
    "partial": ("sim_partial.csv", True),   
}


def load_inputs(measure: str = "pearson"):
    fname, signed = MEASURE_FILES.get(measure, ("sim_pearson.csv", True))
    path = os.path.join(PROC, fname)
    if not os.path.exists(path) and measure == "pearson":
        path = os.path.join(PROC, "corr.csv") 
    sim = pd.read_csv(path, index_col=0)
    meta = pd.read_csv(os.path.join(PROC, "meta.csv"), index_col=0)
    return sim, meta, signed


def build_graph(sim: pd.DataFrame, meta: pd.DataFrame, threshold: float,
                signed: bool = True) -> nx.Graph:
    G = nx.Graph()
    for t in sim.columns:
        country, sector = (meta.loc[t, "country"], meta.loc[t, "sector"]) \
            if t in meta.index else ("?", "?")
        G.add_node(t, country=country, sector=sector)

    tickers = list(sim.columns)
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            a, b = tickers[i], tickers[j]
            rho = float(sim.loc[a, b])
            if abs(rho) >= threshold:
                d = float(np.sqrt(2 * (1 - rho))) if signed else float(np.sqrt(max(1 - rho ** 2, 0)))
                G.add_edge(a, b, rho=rho, weight=abs(rho), distance=d,
                           cross=(G.nodes[a]["country"] != G.nodes[b]["country"]))
    return G


def build_graph_knn(sim: pd.DataFrame, meta: pd.DataFrame, k: int = 5,
                    signed: bool = True, mutual: bool = False) -> nx.Graph:
    G = nx.Graph()
    for t in sim.columns:
        country, sector = (meta.loc[t, "country"], meta.loc[t, "sector"]) \
            if t in meta.index else ("?", "?")
        G.add_node(t, country=country, sector=sector)

    tickers = list(sim.columns)
    topk = {}
    for a in tickers:
        s = sim.loc[a].drop(labels=[a], errors="ignore").astype(float)
        order = s.abs().sort_values(ascending=False)
        topk[a] = list(order.index[:k])

    for a in tickers:
        for b in topk[a]:
            if mutual and a not in topk[b]:
                continue
            if G.has_edge(a, b):
                continue
            rho = float(sim.loc[a, b])
            d = float(np.sqrt(2 * (1 - rho))) if signed else float(np.sqrt(max(1 - rho ** 2, 0)))
            G.add_edge(a, b, rho=rho, weight=abs(rho), distance=d,
                       cross=(G.nodes[a]["country"] != G.nodes[b]["country"]))
    return G


def graph_metrics(G: nx.Graph) -> dict:
    n, m = G.number_of_nodes(), G.number_of_edges()
    ncomp = nx.number_connected_components(G)
    largest = max((len(c) for c in nx.connected_components(G)), default=0)
    density = nx.density(G)
    cross = sum(1 for _, _, d in G.edges(data=True) if d.get("cross"))
    # модулярность по разбиению на страны и по секторам
    try:
        from networkx.algorithms.community import greedy_modularity_communities, modularity
        comms = list(greedy_modularity_communities(G, weight="weight"))
        Q = modularity(G, comms, weight="weight") if m else float("nan")
    except Exception:
        comms, Q = [], float("nan")
    return dict(nodes=n, edges=m, components=ncomp, largest_cc=largest,
                density=density, cross_edges=cross, communities=len(comms),
                modularity=Q)


def node_table(G: nx.Graph) -> pd.DataFrame:
    deg = dict(G.degree())
    wdeg = dict(G.degree(weight="weight"))
    btw = nx.betweenness_centrality(G, weight="distance") if G.number_of_edges() else {n: 0 for n in G}
    # eigenvector centrality не определена на несвязном графе — считаем по компонентам
    eig = {n: 0.0 for n in G}
    for comp in nx.connected_components(G):
        sub = G.subgraph(comp)
        if sub.number_of_edges() == 0:
            continue
        try:
            eig.update(nx.eigenvector_centrality_numpy(sub, weight="weight"))
        except Exception:
            eig.update(nx.eigenvector_centrality(sub, weight="weight", max_iter=1000))
    rows = []
    for t in G.nodes():
        rows.append(dict(ticker=t, country=G.nodes[t]["country"],
                         sector=G.nodes[t]["sector"], degree=deg[t],
                         wdegree=round(wdeg[t], 3), betweenness=round(btw[t], 4),
                         eigen=round(eig[t], 4)))
    return pd.DataFrame(rows).sort_values("degree", ascending=False).reset_index(drop=True)


def draw(G: nx.Graph, path: str, threshold: float, title: str = ""):
    fig, ax = plt.subplots(figsize=(12, 9))
    pos = nx.spring_layout(G, weight="weight", seed=42, k=0.9, iterations=200)
    node_colors = [SECTOR_COLORS.get(G.nodes[n]["sector"], "#333333") for n in G.nodes()]
    node_shapes = {"RU": "o", "US": "s"}
    for country, shape in node_shapes.items():
        nl = [n for n in G.nodes() if G.nodes[n]["country"] == country]
        nx.draw_networkx_nodes(G, pos, nodelist=nl, node_shape=shape,
                               node_color=[SECTOR_COLORS.get(G.nodes[n]["sector"], "#333") for n in nl],
                               node_size=900, edgecolors="black", linewidths=1.2, ax=ax)
    # рёбра: сплошные — положительная корр., пунктир — отрицательная; красные — межрыночные
    for a, b, d in G.edges(data=True):
        style = "-" if d["rho"] >= 0 else "--"
        color = "#d62728" if d["cross"] else "#999999"
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]],
                style, color=color, lw=0.4 + 2.5 * d["weight"], alpha=0.6, zorder=0)
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", ax=ax)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend = [Patch(facecolor=c, edgecolor="black", label=s) for s, c in SECTOR_COLORS.items()]
    legend += [Line2D([0], [0], marker="o", color="w", markerfacecolor="#ccc",
                      markeredgecolor="black", markersize=11, label="RU (круг)"),
               Line2D([0], [0], marker="s", color="w", markerfacecolor="#ccc",
                      markeredgecolor="black", markersize=11, label="US (квадрат)"),
               Line2D([0], [0], color="#d62728", lw=2, label="межрыночное ребро")]
    ax.legend(handles=legend, loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
    ax.set_title(title or f"Граф рынка (|ρ| ≥ {threshold})", fontsize=13)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def sweep(sim, meta, signed):
    print(f"{'thr':>5}{'edges':>7}{'dens':>7}{'comp':>6}{'largCC':>8}{'cross':>7}{'Q':>7}")
    for thr in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]:
        G = build_graph(sim, meta, thr, signed)
        mt = graph_metrics(G)
        print(f"{thr:5.2f}{mt['edges']:7}{mt['density']:7.3f}{mt['components']:6}"
              f"{mt['largest_cc']:8}{mt['cross_edges']:7}{mt['modularity']:7.3f}")


def critical_r(T: int, n_pairs: int, alpha: float = 0.05) -> float:
    if T is None or T <= 3:
        return float("nan")
    a = alpha / max(n_pairs, 1)
    t_crit = stats.t.ppf(1 - a / 2, df=T - 2)
    return float(t_crit / np.sqrt(T - 2 + t_crit ** 2))


def auto_threshold(sim, meta, signed, measure, T=None, plot=True):
    grid = np.round(np.arange(0.05, 0.85 + 1e-9, 0.01), 2)
    rows = []
    for thr in grid:
        G = build_graph(sim, meta, thr, signed)
        mt = graph_metrics(G)
        n = mt["nodes"]
        rows.append(dict(thr=thr, edges=mt["edges"], density=mt["density"],
                         components=mt["components"], largest_cc=mt["largest_cc"],
                         modularity=mt["modularity"], cross=mt["cross_edges"],
                         avg_degree=2 * mt["edges"] / n if n else 0.0))
    df = pd.DataFrame(rows)

    rec = {}
    n = len(sim)
    # 1) перколяция: наибольший порог, при котором граф ещё связен (1 компонента)
    conn = df[df.components == 1]
    rec["percolation"] = float(conn.thr.max()) if len(conn) else None
    # 2) максимум модулярности — только в неразрушенном режиме (крупнейшая
    #    компонента >= половины узлов), иначе Q фиктивно высок на пустом графе
    dense = df[df.largest_cc >= n / 2]
    src = dense if len(dense) else df
    rec["modularity"] = float(src.loc[src.modularity.idxmax(), "thr"])
    # 3) статистическая значимость (только для знаковых корреляционных мер)
    n_pairs = len(sim) * (len(sim) - 1) // 2
    rec["significance"] = round(critical_r(T, n_pairs), 3) if (signed and T) else None
    # 4) «колено»: порог с максимальным приростом числа компонент (резкая фрагментация)
    d_comp = df.components.diff().fillna(0)
    rec["fragmentation_knee"] = float(df.loc[d_comp.idxmax(), "thr"]) if d_comp.max() > 0 else None
    # 5) целевая средняя степень ~ 2*ln(n) (разумная разреженность, связный режим)
    target = 2 * np.log(len(sim))
    rec["target_degree"] = float(df.iloc[(df.avg_degree - target).abs().idxmin()].thr)

    print(f"Автоподбор порога (мера={measure}, узлов={len(sim)}, T={T}):\n")
    print("Рекомендованные пороги по критериям:")
    labels = {
        "percolation": "перколяция (макс. порог со связностью)",
        "modularity": "максимум модулярности Q",
        "significance": "стат. значимость (Bonferroni, α=0.05)",
        "fragmentation_knee": "колено фрагментации (резкий распад)",
        "target_degree": f"целевая ср. степень ≈ 2·ln(n) = {target:.1f}",
    }
    for k, lab in labels.items():
        v = rec[k]
        print(f"  {k:20} {('%.2f' % v) if isinstance(v, float) else '—':>6}  — {lab}")

    if plot:
        _plot_sweep(df, rec, measure)
    return df, rec


def _plot_sweep(df, rec, measure):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    l1, = ax1.plot(df.thr, df.components, color="#d62728", marker=".", label="компоненты")
    l2, = ax1.plot(df.thr, df.avg_degree, color="#1f77b4", marker=".", label="ср. степень")
    ax1.set_xlabel("порог |ρ|")
    ax1.set_ylabel("компоненты / ср. степень")
    ax2 = ax1.twinx()
    l3, = ax2.plot(df.thr, df.modularity, color="#2ca02c", marker=".", label="модулярность Q")
    ax2.set_ylabel("модулярность Q", color="#2ca02c")
    colors = {"percolation": "#9467bd", "modularity": "#2ca02c",
              "significance": "#ff7f0e", "fragmentation_knee": "#d62728",
              "target_degree": "#1f77b4"}
    for k, c in colors.items():
        v = rec.get(k)
        if isinstance(v, float):
            ax1.axvline(v, color=c, ls="--", alpha=0.7, lw=1.2)
            ax1.text(v, ax1.get_ylim()[1] * 0.95, k[:4], rotation=90,
                     color=c, fontsize=8, va="top", ha="right")
    lines = [l1, l2, l3]
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper right", fontsize=9)
    ax1.set_title(f"Метрики графа vs порог ({measure})")
    fig.tight_layout()
    path = os.path.join(PROC, f"threshold_sweep_{measure}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    df.to_csv(os.path.join(PROC, f"threshold_sweep_{measure}.csv"), index=False, encoding="utf-8")
    print(f"\nСохранено: threshold_sweep_{measure}.png, threshold_sweep_{measure}.csv")


def compare(meta, threshold):
    """Сравнение графов по всем мерам сходства при одном пороге."""
    print(f"Сравнение мер сходства при пороге |ρ| ≥ {threshold}:\n")
    print(f"{'measure':10}{'edges':>7}{'comp':>6}{'largCC':>8}{'cross':>7}{'dens':>7}{'Q':>7}")
    for measure, (fname, signed) in MEASURE_FILES.items():
        path = os.path.join(PROC, fname)
        if not os.path.exists(path):
            print(f"{measure:10} нет {fname} — сначала similarity.py")
            continue
        sim = pd.read_csv(path, index_col=0)
        G = build_graph(sim, meta, threshold, signed)
        mt = graph_metrics(G)
        print(f"{measure:10}{mt['edges']:7}{mt['components']:6}{mt['largest_cc']:8}"
              f"{mt['cross_edges']:7}{mt['density']:7.3f}{mt['modularity']:7.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.30)
    ap.add_argument("--measure", choices=list(MEASURE_FILES), default="pearson",
                    help="мера сходства для рёбер графа")
    ap.add_argument("--knn", type=int, default=None, metavar="K",
                    help="kNN-граф: каждый узел с K самыми похожими (напр. --knn 5)")
    ap.add_argument("--mutual", action="store_true",
                    help="только взаимные kNN-рёбра (по умолчанию union)")
    ap.add_argument("--sweep", action="store_true", help="таблица метрик по сетке порогов")
    ap.add_argument("--compare", action="store_true",
                    help="сравнить все меры сходства при одном пороге")
    ap.add_argument("--auto", action="store_true",
                    help="автоподбор порога по объективным критериям + график")
    args, _ = ap.parse_known_args()

    meta = pd.read_csv(os.path.join(PROC, "meta.csv"), index_col=0)

    if args.compare:
        compare(meta, args.threshold)
        return

    sim, meta, signed = load_inputs(args.measure)

    if args.auto:
        T = None
        rpath = os.path.join(PROC, "returns.csv")
        if os.path.exists(rpath):
            T = pd.read_csv(rpath, index_col=0).shape[0]
        auto_threshold(sim, meta, signed, args.measure, T=T)
        return

    if args.sweep:
        print(f"Мера: {args.measure}")
        sweep(sim, meta, signed)
        return

    thr = args.threshold
    if args.knn:
        G = build_graph_knn(sim, meta, k=args.knn, signed=signed, mutual=args.mutual)
        mode = "mutual" if args.mutual else "union"
        suf = f"{args.measure}_knn{args.knn}" + ("_mut" if args.mutual else "")
        title = f"kNN-граф рынка: {args.measure}, k={args.knn} ({mode})"
        header = f"Метрики kNN-графа (мера={args.measure}, k={args.knn}, {mode}):"
    else:
        G = build_graph(sim, meta, thr, signed)
        suf = args.measure
        title = f"Граф рынка: {suf}, |ρ| ≥ {thr}"
        header = f"Метрики графа (мера={args.measure}, порог |ρ| ≥ {thr:.2f}):"

    mt = graph_metrics(G)
    print(header)
    for k, v in mt.items():
        print(f"  {k:12}: {v:.3f}" if isinstance(v, float) else f"  {k:12}: {v}")

    tbl = node_table(G)
    print("\nТоп-узлы по степени:")
    print(tbl.head(10).to_string(index=False))

    os.makedirs(PROC, exist_ok=True)
    nx.write_graphml(G, os.path.join(PROC, f"graph_{suf}.graphml"))
    edges = pd.DataFrame([dict(source=a, target=b, rho=round(d["rho"], 4),
                               distance=round(d["distance"], 4), cross=d["cross"])
                          for a, b, d in G.edges(data=True)])
    edges.to_csv(os.path.join(PROC, f"edges_{suf}.csv"), index=False, encoding="utf-8")
    tbl.to_csv(os.path.join(PROC, f"node_metrics_{suf}.csv"), index=False, encoding="utf-8")
    draw(G, os.path.join(PROC, f"graph_{suf}.png"), thr, title=title)

    print(f"\nСохранено в {PROC}/ (суффикс _{suf}):")
    print(f"  graph_{suf}.graphml     — граф")
    print(f"  edges_{suf}.csv         — рёбра (source,target,rho,distance,cross)")
    print(f"  node_metrics_{suf}.csv  — центральности узлов")
    print(f"  graph_{suf}.png         — визуализация")


if __name__ == "__main__":
    main()
