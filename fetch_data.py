from __future__ import annotations
import argparse
import json
import os
import sys
import time
import urllib.request
import urllib.error
from datetime import date

import numpy as np
import pandas as pd

try:
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    DATA_DIR = os.getcwd()
OUT_DIR = os.path.join(DATA_DIR, "processed")

DEFAULT_START = "2020-01-01"
DEFAULT_END = date.today().isoformat()
UA = {"User-Agent": "Mozilla/5.0 (research; sheaf-graph-study)"}

try:
    from preprocess import META  
    RU_TICKERS = [t for t, (c, _) in META.items() if c == "RU"]
    US_TICKERS = [t for t, (c, _) in META.items() if c == "US"]
except Exception:  
    RU_TICKERS = ["GAZP", "LKOH", "ROSN", "GMKN", "ALRS", "PLZL", "PHOR",
                  "SBER", "MOEX", "MGNT", "OZON", "MTSS", "GEMC", "MDMG",
                  "RTKM", "FLOT", "HYDR", "YDEX", "GECO", "POSI"]
    US_TICKERS = []



def _get_json(url: str, retries: int = 3, pause: float = 0.4) -> dict:
    last = None
    for _ in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8", "replace"))
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
            time.sleep(pause)
    raise RuntimeError(f"GET failed: {url}  ({last})")


def _moex_paged(base: str, start: str, end: str, wanted: list[str]) -> pd.DataFrame:
    rows_all: list[list] = []
    cols: list[str] | None = None
    pos = 0
    while True:
        url = f"{base}?from={start}&till={end}&start={pos}&iss.meta=off"
        d = _get_json(url)
        hist = d["history"]
        cols = hist["columns"]
        rows = hist["data"]
        if not rows:
            break
        rows_all.extend(rows)
        pos += len(rows)
        if len(rows) < 100:
            break
    if not rows_all or cols is None:
        return pd.DataFrame(columns=wanted)
    df = pd.DataFrame(rows_all, columns=cols)
    idx = [c for c in wanted if c in df.columns]
    return df[idx]


def moex_shares(sec: str, start=DEFAULT_START, end=DEFAULT_END,
                board="TQBR") -> pd.DataFrame:
    """Дневные OHLCV российской акции с основного борда TQBR."""
    base = ("https://iss.moex.com/iss/history/engines/stock/markets/shares/"
            f"boards/{board}/securities/{sec}.json")
    raw = _moex_paged(base, start, end,
                      ["TRADEDATE", "OPEN", "HIGH", "LOW", "CLOSE", "VOLUME"])
    if raw.empty:
        return raw
    out = pd.DataFrame({
        "date":   pd.to_datetime(raw["TRADEDATE"]),
        "open":   pd.to_numeric(raw["OPEN"], errors="coerce"),
        "high":   pd.to_numeric(raw["HIGH"], errors="coerce"),
        "low":    pd.to_numeric(raw["LOW"], errors="coerce"),
        "price":  pd.to_numeric(raw["CLOSE"], errors="coerce"),
        "volume": pd.to_numeric(raw["VOLUME"], errors="coerce"),
    }).dropna(subset=["price"]).sort_values("date").reset_index(drop=True)
    return out


def moex_index(sec="IMOEX", start=DEFAULT_START, end=DEFAULT_END) -> pd.Series:
    """Дневной close биржевого индекса (IMOEX, RTSI, ...)."""
    base = ("https://iss.moex.com/iss/history/engines/stock/markets/index/"
            f"securities/{sec}.json")
    raw = _moex_paged(base, start, end, ["TRADEDATE", "CLOSE"])
    if raw.empty:
        return pd.Series(name=sec, dtype=float)
    s = pd.Series(pd.to_numeric(raw["CLOSE"], errors="coerce").values,
                  index=pd.to_datetime(raw["TRADEDATE"]), name=sec).dropna()
    return s.sort_index()


def moex_currency(sec="USD000UTSTOM", start=DEFAULT_START, end=DEFAULT_END,
                  board="CETS") -> pd.Series:
    """Дневной close валютной пары (USD/RUB) на MOEX."""
    base = ("https://iss.moex.com/iss/history/engines/currency/markets/selt/"
            f"boards/{board}/securities/{sec}.json")
    raw = _moex_paged(base, start, end, ["TRADEDATE", "CLOSE"])
    if raw.empty:
        return pd.Series(name="USDRUB", dtype=float)
    s = pd.Series(pd.to_numeric(raw["CLOSE"], errors="coerce").values,
                  index=pd.to_datetime(raw["TRADEDATE"]), name="USDRUB").dropna()
    return s.sort_index()



def us_daily(sym: str, start=DEFAULT_START, end=DEFAULT_END) -> pd.DataFrame:
    """
    Дневные OHLCV американской бумаги
    """
    tiingo = os.environ.get("TIINGO_API_KEY")
    twelve = os.environ.get("TWELVEDATA_API_KEY")
    if tiingo:
        url = (f"https://api.tiingo.com/tiingo/daily/{sym}/prices"
               f"?startDate={start}&endDate={end}&format=json&token={tiingo}")
        req = urllib.request.Request(url, headers={**UA, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode("utf-8", "replace"))
        out = pd.DataFrame({
            "date":   pd.to_datetime([d["date"] for d in data]).tz_localize(None),
            "open":   [d.get("adjOpen", d.get("open")) for d in data],
            "high":   [d.get("adjHigh", d.get("high")) for d in data],
            "low":    [d.get("adjLow", d.get("low")) for d in data],
            "price":  [d.get("adjClose", d.get("close")) for d in data],
            "volume": [d.get("adjVolume", d.get("volume")) for d in data],
        })
        return out.dropna(subset=["price"]).sort_values("date").reset_index(drop=True)
    if twelve:
        url = (f"https://api.twelvedata.com/time_series?symbol={sym}"
               f"&interval=1day&start_date={start}&end_date={end}"
               f"&outputsize=5000&apikey={twelve}")
        d = _get_json(url)
        vals = d.get("values", [])
        if not vals:
            raise RuntimeError(f"Twelve Data: пусто для {sym} ({d.get('message')})")
        out = pd.DataFrame({
            "date":   pd.to_datetime([v["datetime"] for v in vals]),
            "open":   [float(v["open"]) for v in vals],
            "high":   [float(v["high"]) for v in vals],
            "low":    [float(v["low"]) for v in vals],
            "price":  [float(v["close"]) for v in vals],
            "volume": [float(v.get("volume", "nan") or "nan") for v in vals],
        })
        return out.dropna(subset=["price"]).sort_values("date").reset_index(drop=True)
    raise RuntimeError(
        "Нет ключа для US-данных. Получите бесплатный ключ (Tiingo или Twelve Data) "
        "и задайте TIINGO_API_KEY / TWELVEDATA_API_KEY в окружении.")


def _ru_num(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{x:.4f}".rstrip("0").rstrip(".").replace(".", ",")


def write_investing_csv(df: pd.DataFrame, tic: str, outdir: str,
                        force: bool = False) -> str | None:
    path = os.path.join(outdir, f"Прошлые данные - {tic}.csv")
    if os.path.exists(path) and not force:
        print(f"  {tic:6} уже есть, пропуск (--force для перезаписи)")
        return None
    d = df.sort_values("date", ascending=False)  
    chg = df.sort_values("date")["price"].pct_change()
    chg = chg.reindex(df.sort_values("date").index)
    chg_map = dict(zip(df.sort_values("date")["date"], chg))
    rows = []
    for _, r in d.iterrows():
        c = chg_map.get(r["date"], np.nan)
        rows.append({
            "Дата":  r["date"].strftime("%d.%m.%Y"),
            "Цена":  _ru_num(r["price"]),
            "Откр.": _ru_num(r.get("open", np.nan)),
            "Макс.": _ru_num(r.get("high", np.nan)),
            "Мин.":  _ru_num(r.get("low", np.nan)),
            "Объём": "" if pd.isna(r.get("volume", np.nan)) else str(int(r["volume"])),
            "Изм. %": "" if pd.isna(c) else f"{c*100:.2f}".replace(".", ",") + "%",
        })
    out = pd.DataFrame(rows, columns=["Дата", "Цена", "Откр.", "Макс.", "Мин.", "Объём", "Изм. %"])
    out.to_csv(path, index=False, encoding="utf-8-sig", quoting=1)
    print(f"  {tic:6} -> {os.path.basename(path)}  ({len(out)} строк, "
          f"{df['date'].min().date()}..{df['date'].max().date()})")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shares", nargs="*", default=[],
                    help="RU-тикеры для загрузки с MOEX (напр. SBER GAZP)")
    ap.add_argument("--shares-ru-all", action="store_true",
                    help="загрузить все RU-тикеры из META")
    ap.add_argument("--us", nargs="*", default=[],
                    help="US-тикеры (нужен ключ Tiingo/Twelve Data)")
    ap.add_argument("--factors", action="store_true",
                    help="скачать факторный блок IMOEX/RTSI/USDRUB в processed/factors.csv")
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default=DEFAULT_END)
    ap.add_argument("--outdir", default=DATA_DIR)
    ap.add_argument("--force", action="store_true", help="перезаписывать существующие CSV")
    args, _ = ap.parse_known_args()

    did = False

    ru = list(args.shares)
    if args.shares_ru_all:
        ru = RU_TICKERS
    if ru:
        did = True
        print(f"MOEX ISS — RU-акции ({len(ru)}), окно {args.start}..{args.end}:")
        for tic in ru:
            try:
                df = moex_shares(tic, args.start, args.end)
                if df.empty:
                    print(f"  {tic:6} пусто (нет данных на TQBR)")
                    continue
                write_investing_csv(df, tic, args.outdir, force=args.force)
            except Exception as e:
                print(f"  {tic:6} ОШИБКА: {type(e).__name__}: {e}")
            time.sleep(0.2)

    if args.us:
        did = True
        print(f"\nUS-акции ({len(args.us)}) через провайдера с ключом:")
        for tic in args.us:
            try:
                df = us_daily(tic, args.start, args.end)
                write_investing_csv(df, tic, args.outdir, force=args.force)
            except Exception as e:
                print(f"  {tic:6} ОШИБКА: {type(e).__name__}: {e}")
            time.sleep(0.2)

    if args.factors:
        did = True
        print(f"\nФакторный блок (MOEX), окно {args.start}..{args.end}:")
        cols = {}
        for name, fn in [("IMOEX", lambda: moex_index("IMOEX", args.start, args.end)),
                         ("RTSI",  lambda: moex_index("RTSI", args.start, args.end)),
                         ("USDRUB", lambda: moex_currency("USD000UTSTOM", args.start, args.end))]:
            try:
                s = fn()
                cols[name] = s
                print(f"  {name:7} {len(s)} строк  "
                      f"{s.index.min().date()}..{s.index.max().date()}")
            except Exception as e:
                print(f"  {name:7} ОШИБКА: {type(e).__name__}: {e}")
        if cols:
            fac = pd.concat(cols, axis=1, sort=True).sort_index()
            os.makedirs(OUT_DIR, exist_ok=True)
            fpath = os.path.join(OUT_DIR, "factors.csv")
            fac.to_csv(fpath, encoding="utf-8")
            print(f"  -> {fpath}  ({fac.shape[0]} дат × {fac.shape[1]} факторов)")

    if not did:
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
