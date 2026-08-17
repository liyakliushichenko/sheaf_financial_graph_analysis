from __future__ import annotations
import argparse
import glob
import os
import re
import sys
import numpy as np
import pandas as pd

try:
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    DATA_DIR = os.getcwd()
OUT_DIR = os.path.join(DATA_DIR, "processed")

# Секторная разметка узлов графа (страна, сектор) 
META = {
    # Россия
    "GAZP": ("RU", "Energy"),   "LKOH": ("RU", "Energy"),   "ROSN": ("RU", "Energy"),
    "GMKN": ("RU", "Materials"),"ALRS": ("RU", "Materials"),"PLZL": ("RU", "Materials"),
    "PHOR": ("RU", "Materials"),
    "SBER": ("RU", "Financials"),"MOEX": ("RU", "Financials"),
    "MGNT": ("RU", "Consumer"), "OZON": ("RU", "Consumer"),
    "MTSS": ("RU", "Telecom"),
    "GECO": ("RU", "Healthcare"), "GEMC": ("RU", "Healthcare"), "MDMG": ("RU", "Healthcare"),
    "POSI": ("RU", "Tech"),       "YDEX": ("RU", "Tech"),
    "RTKM": ("RU", "Telecom"),
    "FLOT": ("RU", "Industrials"),
    "HYDR": ("RU", "Utilities"),
    # США
    "XOM":  ("US", "Energy"),
    "LIN":  ("US", "Materials"),
    "JPM":  ("US", "Financials"),"GS":  ("US", "Financials"),"V":  ("US", "Financials"),
    "AMZN": ("US", "Consumer"), "WMT": ("US", "Consumer"),
    "META": ("US", "Tech"),     "NVDA": ("US", "Tech"),
    "JNJ":  ("US", "Healthcare"),
    "DUK":  ("US", "Utilities"),
    "CAT":  ("US", "Industrials"),
    "HON":  ("US", "Industrials"),
    "MRK":  ("US", "Healthcare"),
}


EN_FILES = {
    "GENETICO Stock Price History.csv":            "GECO",
    "Gruppa Pozitiv PAO Stock Price History.csv":  "POSI",
    "Honeywell Stock Price History.csv":           "HON",
    "Merck&Co Stock Price History.csv":            "MRK",
    "Rostelekom PJSC Stock Price History.csv":     "RTKM",
    "Sovkomflot PAO Stock Price History.csv":      "FLOT",
    "YANDEX Stock Price History.csv":              "YDEX",
}

NUM_RE = re.compile(r"^-?[\d\s\u00a0]*[\d](?:,\d+)?$")


def parse_number(x: str) -> float:
    if x is None:
        return np.nan
    s = str(x).strip().strip('"')
    if s == "" or s == "-":
        return np.nan
    s = s.replace("\u00a0", "").replace(" ", "").replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_volume(x: str) -> float:
    if x is None:
        return np.nan
    s = str(x).strip().strip('"').replace("\u00a0", "").replace(" ", "")
    if s == "" or s == "-":
        return np.nan
    mult = 1.0
    if s[-1] in "KMBТ": 
        suf = s[-1]
        mult = {"K": 1e3, "M": 1e6, "B": 1e9, "Т": 1e3}[suf]
        s = s[:-1]
    s = s.replace(".", "").replace(",", ".") 
    try:
        return float(s) * mult
    except ValueError:
        return np.nan


def parse_pct(x: str) -> float:
    if x is None:
        return np.nan
    s = str(x).strip().strip('"').replace("%", "").replace(",", ".")
    if s == "" or s == "-":
        return np.nan
    try:
        return float(s) / 100.0
    except ValueError:
        return np.nan


def parse_number_en(x: str) -> float:
    if x is None:
        return np.nan
    s = str(x).strip().strip('"').replace(",", "")
    if s == "" or s == "-":
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_volume_en(x: str) -> float:
    if x is None:
        return np.nan
    s = str(x).strip().strip('"').replace(",", "")
    if s == "" or s == "-":
        return np.nan
    mult = 1.0
    if s and s[-1] in "KMB":
        mult = {"K": 1e3, "M": 1e6, "B": 1e9}[s[-1]]
        s = s[:-1]
    try:
        return float(s) * mult
    except ValueError:
        return np.nan


def parse_pct_en(x: str) -> float:
    if x is None:
        return np.nan
    s = str(x).strip().strip('"').replace("%", "").replace(",", "")
    if s == "" or s == "-":
        return np.nan
    try:
        return float(s) / 100.0
    except ValueError:
        return np.nan


def load_ticker_en(path: str, tic: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    df.columns = [c.strip().strip('"') for c in df.columns]

    df["date"]  = pd.to_datetime(df["Date"].str.strip('"'), format="%m/%d/%Y")
    df["price"] = df["Price"].map(parse_number_en)
    df["open"]  = df["Open"].map(parse_number_en)
    df["high"]  = df["High"].map(parse_number_en)
    df["low"]   = df["Low"].map(parse_number_en)
    df["volume"]= df["Vol."].map(parse_volume_en)
    df["chg"]   = df["Change %"].map(parse_pct_en)

    df["_bad"] = df["chg"].abs().fillna(0).eq(0) & df["volume"].isna()
    df = (df.sort_values(["date", "_bad"])
            .drop_duplicates(subset="date", keep="first"))

    df = df[["date", "price", "open", "high", "low", "volume", "chg"]].copy()
    df["ticker"] = tic
    return df.sort_values("date").reset_index(drop=True)


def load_ticker(path: str) -> pd.DataFrame:
    tic = os.path.basename(path).split(" - ")[1].replace(".csv", "")
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    df.columns = [c.strip().strip('"') for c in df.columns]

    df["date"]  = pd.to_datetime(df["Дата"].str.strip('"'), format="%d.%m.%Y")
    df["price"] = df["Цена"].map(parse_number)
    df["open"]  = df["Откр."].map(parse_number)
    df["high"]  = df["Макс."].map(parse_number)
    df["low"]   = df["Мин."].map(parse_number)
    df["volume"]= df["Объём"].map(parse_volume)
    df["chg"]   = df["Изм. %"].map(parse_pct)


    df["_bad"] = df["chg"].abs().fillna(0).eq(0) & df["volume"].isna()
    df = (df.sort_values(["date", "_bad"])         
            .drop_duplicates(subset="date", keep="first"))

    df = df[["date", "price", "open", "high", "low", "volume", "chg"]].copy()
    df["ticker"] = tic
    return df.sort_values("date").reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop", nargs="*", default=[],
                    help="тикеры для исключения (напр. OZON для расширения окна)")
    ap.add_argument("--join", choices=["inner", "outer"], default="inner",
                    help="inner = пересечение дат (рекомендуется для пучка)")
    args, _ = ap.parse_known_args() 

    
    ru_files = sorted(glob.glob(os.path.join(DATA_DIR, "Прошлые данные - *.csv")))
    ru_tics = {os.path.basename(f).split(" - ")[1].replace(".csv", "") for f in ru_files}
    tasks = [(f, "ru", None) for f in ru_files]
    for fname, tic in EN_FILES.items():
       
        if tic in ru_tics:
            continue
        p = os.path.join(DATA_DIR, fname)
        if os.path.exists(p):
            tasks.append((p, "en", tic))
    drop = {t.upper() for t in args.drop}

    frames = []
    print(f"{'TIC':6}{'N':>6}{'dups_removed':>14}{'start':>12}{'end':>12}")
    for f, kind, tic_hint in sorted(tasks, key=lambda t: os.path.basename(t[0])):
        d = load_ticker_en(f, tic_hint) if kind == "en" else load_ticker(f)
        tic = d["ticker"].iloc[0]
        if tic in drop:
            continue
       
        raw = pd.read_csv(f, dtype=str, encoding="utf-8-sig")
        removed = len(raw) - len(d)
        print(f"{tic:6}{len(d):6}{removed:14}{str(d['date'].min().date()):>12}{str(d['date'].max().date()):>12}")
        frames.append(d)

    long = pd.concat(frames, ignore_index=True)


    prices = long.pivot(index="date", columns="ticker", values="price").sort_index()

    if args.join == "inner":
        before = len(prices)
        prices = prices.dropna(how="any")
        print(f"\ninner-join по датам: {before} -> {len(prices)} строк "
              f"({prices.index.min().date()} → {prices.index.max().date()})")
    else:
        print(f"\nouter-join: {len(prices)} строк, пропуски заполнять отдельно")


    returns = np.log(prices).diff().dropna(how="any")


    corr = returns.corr()

    os.makedirs(OUT_DIR, exist_ok=True)
    prices.to_csv(os.path.join(OUT_DIR, "prices.csv"), encoding="utf-8")
    returns.to_csv(os.path.join(OUT_DIR, "returns.csv"), encoding="utf-8")
    corr.to_csv(os.path.join(OUT_DIR, "corr.csv"), encoding="utf-8")

    meta_df = (pd.DataFrame.from_dict(META, orient="index",
                                      columns=["country", "sector"])
                 .reindex(prices.columns))
    meta_df.index.name = "ticker"
    meta_df.to_csv(os.path.join(OUT_DIR, "meta.csv"), encoding="utf-8")

    print(f"\nСохранено в {OUT_DIR}/:")
    print(f"  prices.csv  {prices.shape[0]} дат × {prices.shape[1]} тикеров")
    print(f"  returns.csv {returns.shape[0]} × {returns.shape[1]}")
    print(f"  corr.csv    {corr.shape[0]} × {corr.shape[1]}")
    print(f"  meta.csv    разметка (страна, сектор)")


    print("\nСредняя корреляция внутри стран / между странами:")
    ru = [t for t in prices.columns if META.get(t, ("",))[0] == "RU"]
    us = [t for t in prices.columns if META.get(t, ("",))[0] == "US"]
    def mean_block(a, b):
        sub = corr.loc[a, b].values
        if a == b:
            iu = np.triu_indices(len(a), k=1)
            return sub[iu].mean()
        return sub.mean()
    print(f"  RU–RU: {mean_block(ru, ru):+.3f}")
    print(f"  US–US: {mean_block(us, us):+.3f}")
    print(f"  RU–US: {mean_block(ru, us):+.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
