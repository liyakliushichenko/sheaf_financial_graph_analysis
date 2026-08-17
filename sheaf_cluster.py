from __future__ import annotations
import argparse
import glob
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms.community import modularity as nx_modularity

from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)
from scipy.stats import spearmanr

from build_graph import PROC, build_graph_knn, node_table
from connection_sheaf import tangent_frames, connection_laplacian
from node_classify import node_features

TOL = 1e-8
RNG = 0



#  Признаки-стебли на вершинах
def _z(a: np.ndarray) -> np.ndarray:
    """z-нормировка по столбцам (каждый актив к нулю/единице)."""
    mu = a.mean(axis=0, keepdims=True)
    sd = a.std(axis=0, keepdims=True)
    return (a - mu) / np.clip(sd, 1e-9, None)


def stalk_returns(R: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Стебель = z-нормированный ряд доходностей актива (F = число дней)."""
    X = _z(R[tickers].to_numpy(float))               # (T, n)
    return pd.DataFrame(X, columns=tickers)


def stalk_prices(P: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Стебель = z-нормированный ряд ЦЕН (нестационарен; для контраста)."""
    X = _z(np.log(P[tickers].to_numpy(float)))
    return pd.DataFrame(X, columns=tickers)


def stalk_augmented(R: pd.DataFrame, tickers: list[str],
                    volumes: pd.DataFrame | None, vol_win: int = 20,
                    use_vol=True, use_volume=True) -> pd.DataFrame:
    """Стебель = [z(доходности) ; z(rolling-волатильность) ; z(log-объём)]."""
    blocks = [_z(R[tickers].to_numpy(float))]
    if use_vol:
        rv = R[tickers].rolling(vol_win, min_periods=5).std().bfill().to_numpy(float)
        blocks.append(_z(rv))
    if use_volume and volumes is not None:
        v = volumes.reindex(R.index).reindex(columns=tickers).ffill().bfill()
        # если у части тикеров объёмы недоступны — подставляем медиану по строке
        v = v.apply(lambda row: row.fillna(row.median()), axis=1)
        blocks.append(_z(np.log(v.to_numpy(float) + 1.0)))
    X = np.vstack(blocks)                             # (F, n), F = сумма длин блоков
    return pd.DataFrame(X, columns=tickers)


def load_volumes(tickers: list[str], index: pd.DatetimeIndex) -> pd.DataFrame | None:
    try:
        from preprocess import load_ticker
    except Exception:
        return None
    try:
        from preprocess import load_ticker_en, EN_FILES
    except Exception:
        load_ticker_en, EN_FILES = None, {}
    base = os.path.dirname(PROC)
    cols = {}

    for f in glob.glob(os.path.join(base, "Прошлые данные - *.csv")):
        try:
            d = load_ticker(f)
            tic = d["ticker"].iloc[0]
            if tic in tickers:
                cols[tic] = pd.Series(d["volume"].values, index=pd.to_datetime(d["date"]))
        except Exception:
            continue

    if load_ticker_en is not None:
        for fname, tic in EN_FILES.items():
            if tic in cols or tic not in tickers:
                continue
            p = os.path.join(base, fname)
            if not os.path.exists(p):
                continue
            try:
                d = load_ticker_en(p, tic)
                cols[tic] = pd.Series(d["volume"].values, index=pd.to_datetime(d["date"]))
            except Exception:
                continue
    if not cols:
        return None
    V = pd.DataFrame(cols).reindex(index)
    return V



#  Спектральные вложения: пучок vs граф
def sheaf_embed(L: np.ndarray, deg: np.ndarray, n: int, d: int, m: int):
    """Нормированный connection Laplacian D^{-1/2} L D^{-1/2}; m младших собств.
    Вложение узла = его d×m блок, развёрнутый в R^{d·m} (row-normalized)."""
    dvec = np.repeat(deg, d)
    dinv = 1.0 / np.sqrt(np.clip(dvec, 1e-12, None))
    Lsym = (L * dinv[None, :]) * dinv[:, None]
    Lsym = 0.5 * (Lsym + Lsym.T)
    w, V = np.linalg.eigh(Lsym)
    U = V[:, :m]                                      # (n·d, m)
    emb = U.reshape(n, d * m)                         # узел i -> строки [i·d:(i+1)·d]
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    return emb, w


def graph_embed(A: np.ndarray, m: int):
    """Нормированный графовый лапласиан I − D^{-1/2} A D^{-1/2}; m младших."""
    deg = A.sum(1)
    dinv = 1.0 / np.sqrt(np.clip(deg, 1e-12, None))
    Lsym = np.eye(len(A)) - (A * dinv[None, :]) * dinv[:, None]
    Lsym = 0.5 * (Lsym + Lsym.T)
    w, V = np.linalg.eigh(Lsym)
    U = V[:, :m]
    U = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)
    return U, w


def kmeans_labels(emb: np.ndarray, k: int) -> np.ndarray:
    return KMeans(n_clusters=k, n_init=10, random_state=RNG).fit_predict(emb)


def adjacency(G, tickers):
    idx = {t: i for i, t in enumerate(tickers)}
    A = np.zeros((len(tickers), len(tickers)))
    for a, b, dd in G.edges(data=True):
        w = float(dd["weight"])
        A[idx[a], idx[b]] = A[idx[b], idx[a]] = w
    return A


def eval_partition(pred, emb, y_sec, y_cty, G, tickers):
    parts = {}
    for t, l in zip(tickers, pred):
        parts.setdefault(l, set()).add(t)
    try:
        Q = nx_modularity(G, list(parts.values()), weight="weight")
    except Exception:
        Q = float("nan")
    sil = silhouette_score(emb, pred) if len(set(pred)) > 1 else float("nan")
    return dict(
        ARI_sector=adjusted_rand_score(y_sec, pred),
        NMI_sector=normalized_mutual_info_score(y_sec, pred),
        ARI_country=adjusted_rand_score(y_cty, pred),
        silhouette=sil,
        modularity=Q,
    )


def run_both(feat: pd.DataFrame, R_win: pd.DataFrame, tickers, meta,
             sim: pd.DataFrame, d: int, k_graph: int, n_clusters: int,
             m_sheaf: int):
    """Строит kNN-граф по корреляциям, затем пучковую и графовую кластеризацию."""
    G = build_graph_knn(sim.loc[tickers, tickers], meta, k=k_graph,
                        signed=True, mutual=False)
    n = len(tickers)
    y_sec = np.array([meta.loc[t, "sector"] for t in tickers])
    y_cty = np.array([meta.loc[t, "country"] for t in tickers])

    # пучок
    frames = tangent_frames(feat, G, tickers, d)
    L, deg = connection_laplacian(G, tickers, frames, d)
    emb_s, ev_s = sheaf_embed(L, deg, n, d, m_sheaf)
    pred_s = kmeans_labels(emb_s, n_clusters)

    # граф
    A = adjacency(G, tickers)
    emb_g, ev_g = graph_embed(A, n_clusters)
    pred_g = kmeans_labels(emb_g, n_clusters)

    res_s = eval_partition(pred_s, emb_s, y_sec, y_cty, G, tickers)
    res_g = eval_partition(pred_g, emb_g, y_sec, y_cty, G, tickers)
    return dict(G=G, L=L, deg=deg, emb_s=emb_s, emb_g=emb_g,
                pred_s=pred_s, pred_g=pred_g, res_s=res_s, res_g=res_g,
                ev_s=ev_s)



#  H2: ядро/периферия по Фидлеру пучка
def sheaf_fiedler_core(L, deg, n, d):
    dvec = np.repeat(deg, d)
    dinv = 1.0 / np.sqrt(np.clip(dvec, 1e-12, None))
    Lsym = (L * dinv[None, :]) * dinv[:, None]
    Lsym = 0.5 * (Lsym + Lsym.T)
    w, V = np.linalg.eigh(Lsym)
    nz = np.where(w > TOL)[0]
    fied = V[:, nz[0]] if len(nz) else V[:, 0]
    blocks = fied.reshape(n, d)
    return np.linalg.norm(blocks, axis=1)            # core-score по узлам


def graph_fiedler(A):
    deg = A.sum(1)
    dinv = 1.0 / np.sqrt(np.clip(deg, 1e-12, None))
    Lsym = np.eye(len(A)) - (A * dinv[None, :]) * dinv[:, None]
    w, V = np.linalg.eigh(Lsym)
    return np.abs(V[:, 1])                            # |вектор Фидлера| по узлам


def _fmt(res: dict) -> str:
    return (f"{res['ARI_sector']:>10.3f}{res['NMI_sector']:>10.3f}"
            f"{res['ARI_country']:>11.3f}{res['silhouette']:>11.3f}"
            f"{res['modularity']:>12.3f}")


HDR = f"{'метод':22}{'ARI_sec':>10}{'NMI_sec':>10}{'ARI_cty':>11}{'silhouette':>11}{'modular':>12}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=2, help="размерность стебля d")
    ap.add_argument("--k-graph", type=int, default=5, help="kNN для связности")
    ap.add_argument("--stalk", choices=["returns", "prices"], default="returns")
    ap.add_argument("--m-sheaf", type=int, default=0,
                    help="число младших собств. векторов пучка (0 = как #кластеров)")
    args, _ = ap.parse_known_args()
    d = args.dim

    R = pd.read_csv(os.path.join(PROC, "returns.csv"), index_col=0, parse_dates=True)
    P = pd.read_csv(os.path.join(PROC, "prices.csv"), index_col=0, parse_dates=True)
    meta = pd.read_csv(os.path.join(PROC, "meta.csv"), index_col=0)
    tickers = [t for t in R.columns if t in meta.index]
    R, P = R[tickers], P[tickers]

    n_sectors = meta.loc[tickers, "sector"].nunique()
    k_clusters = n_sectors
    m_sheaf = args.m_sheaf if args.m_sheaf > 0 else k_clusters
    volumes = load_volumes(tickers, R.index)

    print("=" * 78)
    print(f"КЛАСТЕРИЗАЦИЯ: пучковый лапласиан vs графовый  (Barbero et al. 2022)")
    print(f"Активов={len(tickers)}, стебель d={d}, kNN k={args.k_graph}, "
          f"#кластеров={k_clusters} (=числу секторов), стебель-тип={args.stalk}")
    print(f"Объёмы для H3: {'загружены' if volumes is not None else 'НЕТ'}")
    print("=" * 78)

    def make_feat(Rw, Pw, tk, kind="returns"):
        if kind == "prices":
            return stalk_prices(Pw, tk)
        return stalk_returns(Rw, tk)

    # ---------------------------------------------------------------- H1 ----
    print("\n### Гипотеза 1: пучок vs граф, полное окно и под-периоды\n")
    periods = {
        "весь период": (R.index.min(), R.index.max()),
        "кризис 2022": (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")),
        "спокойный 2024": (pd.Timestamp("2023-07-01"), pd.Timestamp("2024-06-30")),
    }
    h1_rows = []
    for pname, (a, b) in periods.items():
        Rw = R.loc[a:b]
        if len(Rw) < 30:
            print(f"  [{pname}] пропуск: мало наблюдений ({len(Rw)})")
            continue
        sim = Rw.corr(method="pearson")
        feat = make_feat(Rw, P.loc[a:b], tickers, args.stalk)
        out = run_both(feat, Rw, tickers, meta, sim, d, args.k_graph,
                       k_clusters, m_sheaf)
        vol = float(Rw.mean(axis=1).std() * np.sqrt(252))
        print(f"[{pname}]  n_дней={len(Rw)}, годовая волат. рынка={vol:.2f}")
        print("  " + HDR)
        print(f"  {'пучок (sheaf)':22}" + _fmt(out["res_s"]))
        print(f"  {'граф (baseline)':22}" + _fmt(out["res_g"]))
        dari = out["res_s"]["ARI_sector"] - out["res_g"]["ARI_sector"]
        dq = out["res_s"]["modularity"] - out["res_g"]["modularity"]
        print(f"  Δ(пучок−граф): ARI_sec={dari:+.3f}, modularity={dq:+.3f}\n")
        for meth, r in [("sheaf", out["res_s"]), ("graph", out["res_g"])]:
            h1_rows.append(dict(period=pname, method=meth, vol=round(vol, 3), **r))
    pd.DataFrame(h1_rows).to_csv(os.path.join(PROC, "cluster_H1.csv"),
                                 index=False, encoding="utf-8")

    # ---------------------------------------------------------------- H2 ----
    print("### Гипотеза 2: Фидлер пучка → ядро/периферия и системный риск\n")
    sim_full = R.corr(method="pearson")
    feat_full = make_feat(R, P, tickers, args.stalk)
    G = build_graph_knn(sim_full.loc[tickers, tickers], meta, k=args.k_graph,
                        signed=True, mutual=False)
    frames = tangent_frames(feat_full, G, tickers, d)
    L, deg = connection_laplacian(G, tickers, frames, d)
    core_sheaf = sheaf_fiedler_core(L, deg, len(tickers), d)
    A = adjacency(G, tickers)
    core_graph = graph_fiedler(A)

    # системная чувствительность: beta к равновзв. рынку + центральности графа
    feats = node_features(R, tickers)
    nt = node_table(G).set_index("ticker").reindex(tickers)
    sys_beta = feats["beta"].to_numpy(float)
    sys_wdeg = nt["wdegree"].to_numpy(float)
    sys_eig = nt["eigen"].to_numpy(float)
    meanabscorr = (sim_full.loc[tickers, tickers].abs().sum(1).to_numpy() - 1) / (len(tickers) - 1)

    h2 = pd.DataFrame({
        "ticker": tickers,
        "country": [meta.loc[t, "country"] for t in tickers],
        "sector": [meta.loc[t, "sector"] for t in tickers],
        "core_sheaf": core_sheaf,
        "core_graph": core_graph,
        "beta_mkt": sys_beta,
        "wdegree": sys_wdeg,
        "eigen_cent": sys_eig,
        "mean|corr|": meanabscorr,
    }).sort_values("core_sheaf", ascending=False)
    h2.to_csv(os.path.join(PROC, "cluster_H2_core.csv"), index=False, encoding="utf-8")

    print("Корреляция Спирмена core-score ↔ системная чувствительность:")
    print(f"{'мера риска':16}{'Фидлер-пучок':>16}{'Фидлер-граф':>16}")
    for nm, arr in [("beta_mkt", sys_beta), ("wdegree", sys_wdeg),
                    ("eigen_cent", sys_eig), ("mean|corr|", meanabscorr)]:
        rs, ps = spearmanr(core_sheaf, arr)
        rg, pg = spearmanr(core_graph, arr)
        print(f"{nm:16}{rs:>10.3f}(p={ps:.2f}){rg:>8.3f}(p={pg:.2f})")
    print("\nТоп-5 ЯДРО (макс core_sheaf) и ПЕРИФЕРИЯ (мин):")
    print("  ядро     :", ", ".join(h2.head(5)["ticker"]))
    print("  периферия:", ", ".join(h2.tail(5)["ticker"]))

    # ---------------------------------------------------------------- H3 ----
    print("\n### Гипотеза 3: обогащение стебля признаками (волат./объём)\n")
    h3_variants = {
        "доходности": stalk_returns(R, tickers),
        "+волатильность": stalk_augmented(R, tickers, volumes, use_vol=True, use_volume=False),
        "+волат.+объём": stalk_augmented(R, tickers, volumes, use_vol=True, use_volume=True),
    }
    print(HDR)
    h3_rows = []
    for name, feat in h3_variants.items():
        out = run_both(feat, R, tickers, meta, sim_full, d, args.k_graph,
                       k_clusters, m_sheaf)
        print(f"{name:22}" + _fmt(out["res_s"]))
        h3_rows.append(dict(features=name, **out["res_s"]))
    # графовый baseline не зависит от признаков 
    base = run_both(stalk_returns(R, tickers), R, tickers, meta, sim_full,
                    d, args.k_graph, k_clusters, m_sheaf)
    print(f"{'граф (baseline)':22}" + _fmt(base["res_g"]))
    h3_rows.append(dict(features="граф (baseline)", **base["res_g"]))
    pd.DataFrame(h3_rows).to_csv(os.path.join(PROC, "cluster_H3.csv"),
                                 index=False, encoding="utf-8")

    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    hp = pd.DataFrame(h1_rows)
    piv = hp.pivot(index="period", columns="method", values="ARI_sector")
    piv.plot(kind="bar", ax=axes[0], color={"sheaf": "#1f77b4", "graph": "#7f7f7f"})
    axes[0].set_title("H1: ARI(сектор) по периодам")
    axes[0].set_ylabel("ARI"); axes[0].tick_params(axis="x", rotation=20)
    axes[1].scatter(core_sheaf, sys_beta, c=["#d62728" if c == "RU" else "#1f77b4"
                    for c in h2.sort_index().index.map(lambda i: meta.loc[tickers[i], "country"])]
                    if False else ["#1f77b4"] * len(tickers))
    axes[1].set_xlabel("core-score (Фидлер пучка)")
    axes[1].set_ylabel("beta к рынку")
    axes[1].set_title("H2: ядро/периферия ↔ системный риск")
    fig.tight_layout()
    fig.savefig(os.path.join(PROC, "cluster_summary.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("\nСохранено: cluster_H1.csv, cluster_H2_core.csv, cluster_H3.csv, "
          "cluster_summary.png")


if __name__ == "__main__":
    main()
