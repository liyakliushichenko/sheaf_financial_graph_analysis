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
from networkx.algorithms.community import louvain_communities

from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)
from scipy.stats import spearmanr

from build_graph import PROC, build_graph_knn, node_table
from connection_sheaf import tangent_frames, connection_laplacian, procrustes
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
                    use_vol=True, use_volume=True,
                    w_returns: float = 1.0, w_vol: float = 0.5,
                    w_volume: float = 0.5) -> pd.DataFrame:
    """Стебель = взвешенная конкатенация [доходности ; волатильность ; объём]."""
    def _wblock(mat: np.ndarray, weight: float) -> np.ndarray:
        z = _z(mat)
        return z / np.sqrt(z.shape[0]) * float(weight)  # равный вклад по размерности
    blocks = [_wblock(R[tickers].to_numpy(float), w_returns)]
    if use_vol:
        rv = R[tickers].rolling(vol_win, min_periods=5).std().bfill().to_numpy(float)
        blocks.append(_wblock(rv, w_vol))
    if use_volume and volumes is not None:
        v = volumes.reindex(R.index).reindex(columns=tickers).ffill().bfill()
        # если у части тикеров объёмы недоступны — подставляем медиану по строке
        v = v.apply(lambda row: row.fillna(row.median()), axis=1)
        blocks.append(_wblock(np.log(v.to_numpy(float) + 1.0), w_volume))
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



#  Корреляционная матрица рынка (Эксперимент А: робастная альтернатива Пирсону)
def market_corr(Rw: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    if method == "spearman":
        return Rw.corr(method="spearman")
    if method == "robust":
        lo = Rw.quantile(0.025)
        hi = Rw.quantile(0.975)
        Rc = Rw.clip(lower=lo, upper=hi, axis=1)
        return Rc.corr(method="pearson")
    return Rw.corr(method="pearson")



#  Эксперимент В: матричные стебли на многообразии SPD (Peng et al.)
def regional_returns(R: pd.DataFrame, tickers: list[str], meta) -> pd.DataFrame:
    "
    out = {}
    for cty in sorted({meta.loc[t, "country"] for t in tickers}):
        cols = [t for t in tickers if meta.loc[t, "country"] == cty]
        out[f"mkt_{cty}"] = R[cols].mean(axis=1)
    return pd.DataFrame(out, index=R.index)


def spd_logeuclid_embed(R: pd.DataFrame, tickers: list[str], meta,
                        reg: float = 1e-2) -> np.ndarray:
    reg_ret = regional_returns(R, tickers, meta).to_numpy(float)     # (T, C)
    p = reg_ret.shape[1] + 1
    iu = np.triu_indices(p)
    sqrt2 = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))              # метрика Фробениуса
    descs = []
    for t in tickers:
        M = np.column_stack([R[t].to_numpy(float), reg_ret])        # (T, p)
        C = np.cov(M, rowvar=False)
        C = C + reg * (np.trace(C) / p) * np.eye(p)                 # SPD-регуляризация
        w, V = np.linalg.eigh(C)
        logC = (V * np.log(np.clip(w, 1e-12, None))) @ V.T
        descs.append(logC[iu] * sqrt2)
    return _z(np.array(descs))                                      # (n, p(p+1)/2)



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


def transport_agreement(Oij: np.ndarray, d: int) -> float:
    return float(max(0.0, 1.0 - np.linalg.norm(Oij - np.eye(d), "fro") / (2.0 * np.sqrt(d))))


def louvain_sheaf(G, frames: dict, tickers: list[str], d: int,
                  resolution: float = 1.0) -> np.ndarray:
    W = nx.Graph()
    W.add_nodes_from(tickers)
    for a, b, data in G.edges(data=True):
        w = float(data["weight"])
        agree = transport_agreement(procrustes(frames[a], frames[b]), d)
        we = w * agree
        if we > 1e-9:
            W.add_edge(a, b, weight=we)
    comms = louvain_communities(W, weight="weight", resolution=resolution, seed=RNG)
    lab = {}
    for c, nodes in enumerate(comms):
        for t in nodes:
            lab[t] = c
    return np.array([lab.get(t, -1) for t in tickers])


def adaptive_k(k_base: int, n_days: int, vol_annual: float, n_assets: int,
               vol_ref: float = 0.20) -> int:
    """Адаптивное число соседей kNN для локального PCA."""
    vol_factor = float(np.clip(vol_annual / vol_ref, 1.0, 2.5))
    k = int(round(k_base * vol_factor))
    k_cap_days = max(k_base, int(np.sqrt(max(n_days, 1))))
    k = min(k, k_cap_days, n_assets - 1)
    return max(3, k)


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
             m_sheaf: int, adaptive: bool = True):
    """Строит kNN-граф по корреляциям, затем пучковую и графовую кластеризацию."""
    n = len(tickers)
    if adaptive:
        vol_annual = float(R_win.mean(axis=1).std() * np.sqrt(252))
        k_eff = adaptive_k(k_graph, len(R_win), vol_annual, n)
    else:
        k_eff = k_graph
    G = build_graph_knn(sim.loc[tickers, tickers], meta, k=k_eff,
                        signed=True, mutual=False)
    y_sec = np.array([meta.loc[t, "sector"] for t in tickers])
    y_cty = np.array([meta.loc[t, "country"] for t in tickers])

    # пучок
    frames = tangent_frames(feat, G, tickers, d)
    L, deg = connection_laplacian(G, tickers, frames, d)
    emb_s, ev_s = sheaf_embed(L, deg, n, d, m_sheaf)
    pred_s = kmeans_labels(emb_s, n_clusters)

    # Лувен по пучково-взвешенному графу (Эксперимент Б)
    pred_l = louvain_sheaf(G, frames, tickers, d)
    res_l = eval_partition(pred_l, emb_s, y_sec, y_cty, G, tickers)

    # граф
    A = adjacency(G, tickers)
    emb_g, ev_g = graph_embed(A, n_clusters)
    pred_g = kmeans_labels(emb_g, n_clusters)

    res_s = eval_partition(pred_s, emb_s, y_sec, y_cty, G, tickers)
    res_g = eval_partition(pred_g, emb_g, y_sec, y_cty, G, tickers)
    return dict(G=G, L=L, deg=deg, emb_s=emb_s, emb_g=emb_g,
                pred_s=pred_s, pred_g=pred_g, pred_l=pred_l,
                res_s=res_s, res_g=res_g, res_l=res_l,
                ev_s=ev_s, k_eff=k_eff)



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


#  dim ker connection Laplacian на скользящем окне 
def rolling_kernel_indicator(R, tickers, meta, d, k_graph, corr, adaptive,
                             win=63, step=21, eps=0.1, tau=0.3):
    idx = R.index
    n = len(tickers)
    rows = []
    for s in range(0, len(idx) - win + 1, step):
        Rw = R.iloc[s:s + win]
        sim = market_corr(Rw, corr)
        vol = float(Rw.mean(axis=1).std() * np.sqrt(252))
        k_eff = adaptive_k(k_graph, len(Rw), vol, n) if adaptive else k_graph
        G = build_graph_knn(sim.loc[tickers, tickers], meta, k=k_eff,
                            signed=True, mutual=False)
        beta0 = nx.number_connected_components(G)
        frames = tangent_frames(stalk_returns(Rw, tickers), G, tickers, d)
        L, deg = connection_laplacian(G, tickers, frames, d)
        dvec = np.repeat(deg, d)
        dinv = 1.0 / np.sqrt(np.clip(dvec, 1e-12, None))
        Lsym = (L * dinv[None, :]) * dinv[:, None]
        Lsym = 0.5 * (Lsym + Lsym.T)
        ev = np.clip(np.linalg.eigvalsh(Lsym), 0, None)
        nz = ev[ev >= TOL]
        rows.append(dict(
            date=idx[s + win - 1], vol=round(vol, 4), beta0=beta0, k_eff=k_eff,
            dim_ker=int((ev < TOL).sum()),
            soft_ker=int((ev < eps).sum()),
            heat_dim=float(np.sum(np.exp(-ev / tau))),
            frustration=d * beta0 - int((ev < TOL).sum()),
            lam2=float(nz.min()) if len(nz) else 0.0,
        ))
    return pd.DataFrame(rows)


def load_vix(path: str, index: pd.DatetimeIndex) -> pd.Series | None:
    try:
        from preprocess import load_ticker
        d = load_ticker(path)
    except Exception:
        return None
    s = pd.Series(d["price"].values, index=pd.to_datetime(d["date"])).sort_index()
    return s.reindex(index).ffill().bfill()


def window_aggregate(series: pd.Series, index, win: int, step: int):
    vmean, vend = [], []
    for s in range(0, len(index) - win + 1, step):
        w = series.iloc[s:s + win]
        vmean.append(float(w.mean()))
        vend.append(float(w.iloc[-1]))
    return np.array(vmean), np.array(vend)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=2, help="размерность стебля d")
    ap.add_argument("--k-graph", type=int, default=5, help="kNN для связности")
    ap.add_argument("--stalk", choices=["returns", "prices"], default="returns")
    ap.add_argument("--m-sheaf", type=int, default=0,
                    help="число младших собств. векторов пучка (0 = как #кластеров)")
    ap.add_argument("--corr", choices=["pearson", "spearman", "robust"],
                    default="pearson",
                    help="мера корреляции для графа (Эксперимент А: spearman/robust "
                         "устойчивее к кризисным выбросам)")
    ap.add_argument("--no-adaptive-k", action="store_true",
                    help="отключить адаптивный k (использовать фиксированный --k-graph)")
    ap.add_argument("--w-vol", type=float, default=0.5,
                    help="вес блока волатильности в обогащённом стебле (H3)")
    ap.add_argument("--w-volume", type=float, default=0.5,
                    help="вес блока объёма в обогащённом стебле (H3)")
    ap.add_argument("--vix", type=str, default=None,
                    help="путь к CSV VIX (Investing 'Прошлые данные - ...'); если задан, "
                         "H5 сверяет индикатор с настоящим VIX вместо реализ. волатильности")
    args, _ = ap.parse_known_args()
    d = args.dim
    adaptive = not args.no_adaptive_k

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
    print(f"Активов={len(tickers)}, стебель d={d}, "
          f"kNN k={args.k_graph}{' (адаптивный)' if adaptive else ' (фиксированный)'}, "
          f"#кластеров={k_clusters} (=числу секторов), стебель-тип={args.stalk}, "
          f"корреляция={args.corr}")
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
        sim = market_corr(Rw, args.corr)
        feat = make_feat(Rw, P.loc[a:b], tickers, args.stalk)
        out = run_both(feat, Rw, tickers, meta, sim, d, args.k_graph,
                       k_clusters, m_sheaf, adaptive=adaptive)
        vol = float(Rw.mean(axis=1).std() * np.sqrt(252))
        print(f"[{pname}]  n_дней={len(Rw)}, годовая волат. рынка={vol:.2f}, "
              f"k(эфф.)={out['k_eff']}")
        print("  " + HDR)
        print(f"  {'пучок (sheaf, kmeans)':22}" + _fmt(out["res_s"]))
        print(f"  {'пучок (louvain)':22}" + _fmt(out["res_l"]))
        print(f"  {'граф (baseline)':22}" + _fmt(out["res_g"]))
        dari = out["res_s"]["ARI_sector"] - out["res_g"]["ARI_sector"]
        dq = out["res_l"]["modularity"] - out["res_g"]["modularity"]
        print(f"  Δ: ARI_sec(пучок−граф)={dari:+.3f}, "
              f"modularity(louvain−граф)={dq:+.3f}\n")
        for meth, r in [("sheaf", out["res_s"]), ("louvain", out["res_l"]),
                        ("graph", out["res_g"])]:
            h1_rows.append(dict(period=pname, method=meth, vol=round(vol, 3),
                                k_eff=out["k_eff"], **r))
    pd.DataFrame(h1_rows).to_csv(os.path.join(PROC, "cluster_H1.csv"),
                                 index=False, encoding="utf-8")

    # ---------------------------------------------------------------- H2 ----
    print("### Гипотеза 2: Фидлер пучка → ядро/периферия и системный риск\n")
    sim_full = market_corr(R, args.corr)
    feat_full = make_feat(R, P, tickers, args.stalk)
    vol_full = float(R.mean(axis=1).std() * np.sqrt(252))
    k_full = adaptive_k(args.k_graph, len(R), vol_full, len(tickers)) if adaptive \
        else args.k_graph
    G = build_graph_knn(sim_full.loc[tickers, tickers], meta, k=k_full,
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
    print(f"(веса блоков: доходности=1.0, волат.={args.w_vol}, объём={args.w_volume})")
    h3_variants = {
        "доходности": stalk_returns(R, tickers),
        "+волатильность": stalk_augmented(R, tickers, volumes, use_vol=True,
                                          use_volume=False, w_vol=args.w_vol),
        "+волат.+объём": stalk_augmented(R, tickers, volumes, use_vol=True,
                                         use_volume=True, w_vol=args.w_vol,
                                         w_volume=args.w_volume),
    }
    print(HDR)
    h3_rows = []
    for name, feat in h3_variants.items():
        out = run_both(feat, R, tickers, meta, sim_full, d, args.k_graph,
                       k_clusters, m_sheaf, adaptive=adaptive)
        print(f"{name:22}" + _fmt(out["res_s"]))
        h3_rows.append(dict(features=name, **out["res_s"]))
    # графовый baseline не зависит от признаков
    base = run_both(stalk_returns(R, tickers), R, tickers, meta, sim_full,
                    d, args.k_graph, k_clusters, m_sheaf, adaptive=adaptive)
    print(f"{'граф (baseline)':22}" + _fmt(base["res_g"]))
    h3_rows.append(dict(features="граф (baseline)", **base["res_g"]))
    pd.DataFrame(h3_rows).to_csv(os.path.join(PROC, "cluster_H3.csv"),
                                 index=False, encoding="utf-8")

    # ---------------------------------------------------------------- H4 ----
    print("\n### Гипотеза 4 (Эксперимент В): матричные стебли на многообразии SPD\n")
    y_sec = np.array([meta.loc[t, "sector"] for t in tickers])
    y_cty = np.array([meta.loc[t, "country"] for t in tickers])
    emb_spd = spd_logeuclid_embed(R, tickers, meta)
    pred_spd = kmeans_labels(emb_spd, k_clusters)
    res_spd = eval_partition(pred_spd, emb_spd, y_sec, y_cty, G, tickers)
    ref = run_both(stalk_returns(R, tickers), R, tickers, meta, sim_full,
                   d, args.k_graph, k_clusters, m_sheaf, adaptive=adaptive)
    print(HDR)
    print(f"{'SPD log-Euclid':22}" + _fmt(res_spd))
    print(f"{'пучок (returns)':22}" + _fmt(ref["res_s"]))
    print(f"{'пучок louvain':22}" + _fmt(ref["res_l"]))
    print(f"{'граф baseline':22}" + _fmt(ref["res_g"]))
    h4_rows = [dict(method="SPD log-Euclid", **res_spd),
               dict(method="пучок (returns)", **ref["res_s"]),
               dict(method="пучок louvain", **ref["res_l"]),
               dict(method="граф baseline", **ref["res_g"])]
    pd.DataFrame(h4_rows).to_csv(os.path.join(PROC, "cluster_H4_spd.csv"),
                                 index=False, encoding="utf-8")
    pd.DataFrame({"ticker": tickers, "country": y_cty, "sector": y_sec,
                  "cluster_spd": pred_spd}).to_csv(
        os.path.join(PROC, "cluster_H4_labels.csv"), index=False, encoding="utf-8")

    # ---------------------------------------------------------------- H5 ----
    print("\n### Гипотеза 5 (Эксперимент Г): dim ker L(t) как макро-индикатор стресса\n")
    kern = rolling_kernel_indicator(R, tickers, meta, d, args.k_graph, args.corr,
                                    adaptive, win=63, step=21)
    # выбор меры стресса: настоящий VIX (если передан) либо реализ. волатильность
    vix = load_vix(args.vix, R.index) if args.vix else None
    if vix is not None:
        vmean, vend = window_aggregate(vix, R.index, 63, 21)
        kern["vix_mean"], kern["vix_end"] = vmean, vend
        stress = kern["vix_mean"]
        stress_name, stress_lbl = "VIX", "VIX (среднее по окну)"
    else:
        stress = kern["vol"]
        stress_name, stress_lbl = "волат. рынка", "годовая волат. рынка (прокси VIX)"
    kern.to_csv(os.path.join(PROC, "cluster_H5_kernel.csv"),
                index=False, encoding="utf-8")

    rs_heat, ps_heat = spearmanr(kern["heat_dim"], stress)
    rs_l2, ps_l2 = spearmanr(kern["lam2"], stress)
    print(f"Окон={len(kern)} (win=63, step=21). Спирмен с {stress_name}"
          f"{' (' + os.path.basename(args.vix) + ')' if vix is not None else ''}:")
    print(f"  heat_dim (Σe^-λ/τ, эфф. ker) ↔ {stress_name:12} : rs={rs_heat:+.3f} "
          f"(p={ps_heat:.3f})   ожидаем < 0 (спокойно ⇒ выше эфф. dim ker)")
    print(f"  λ₂ (алг. связность)         ↔ {stress_name:12} : rs={rs_l2:+.3f} "
          f"(p={ps_l2:.3f})")
    if vix is not None:
        rs_vp, _ = spearmanr(kern["vol"], kern["vix_mean"])
        print(f"  валидация прокси: реализ. волат. ↔ VIX : rs={rs_vp:+.3f} "
              f"(наша волатильность как заменитель VIX)")
    smax = kern.loc[stress.idxmax()]
    smin = kern.loc[stress.idxmin()]
    sfmt = (lambda r: f"VIX={r['vix_mean']:.1f}") if vix is not None \
        else (lambda r: f"vol={r['vol']:.2f}")
    print(f"  макс. стресс {smax['date'].date()} ({sfmt(smax)}): "
          f"heat_dim={smax['heat_dim']:.2f}, λ₂={smax['lam2']:.3f}")
    print(f"  мин. стресс  {smin['date'].date()} ({sfmt(smin)}): "
          f"heat_dim={smin['heat_dim']:.2f}, λ₂={smin['lam2']:.3f}")

    figk, axk = plt.subplots(figsize=(11, 5))
    axk.plot(kern["date"], kern["heat_dim"], "o-", ms=3, color="#1f77b4",
             label="эфф. dim ker (heat trace)")
    axk.set_xlabel("дата (конец окна)")
    axk.set_ylabel("эфф. dim ker  Σe^{-λ/τ}", color="#1f77b4")
    axk.tick_params(axis="y", labelcolor="#1f77b4")
    axk2 = axk.twinx()
    axk2.plot(kern["date"], stress, "s--", ms=3, color="#d62728", alpha=0.7,
              label=stress_lbl)
    axk2.set_ylabel(stress_lbl, color="#d62728")
    axk2.tick_params(axis="y", labelcolor="#d62728")
    axk.set_title(f"H5: эфф. dim ker connection Laplacian vs {stress_lbl} "
                  f"(rs={rs_heat:+.2f})")
    figk.tight_layout()
    figk.savefig(os.path.join(PROC, "cluster_kernel_indicator.png"), dpi=150,
                 bbox_inches="tight")
    plt.close(figk)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    hp = pd.DataFrame(h1_rows)
    piv = hp.pivot(index="period", columns="method", values="ARI_sector")
    cmap = {"sheaf": "#1f77b4", "louvain": "#2ca02c", "graph": "#7f7f7f"}
    piv.plot(kind="bar", ax=axes[0],
             color=[cmap.get(c, "#333333") for c in piv.columns])
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
          "cluster_H4_spd.csv, cluster_H4_labels.csv, cluster_H5_kernel.csv, "
          "cluster_summary.png, cluster_kernel_indicator.png")


if __name__ == "__main__":
    main()
