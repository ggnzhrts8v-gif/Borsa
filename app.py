import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

st.set_page_config(page_title="BIST Analiz", layout="wide")

DEFAULTS = ["THYAO", "GARAN", "ASELS", "BIMAS", "EREGL", "KCHOL", "SISE", "TUPRS"]


@st.cache_data(ttl=900)
def load(sym: str, period: str) -> pd.DataFrame:
    return yf.Ticker(sym + ".IS").history(period=period, auto_adjust=True)


def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / down)


def macd(close):
    line = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    return line, line.ewm(span=9, adjust=False).mean()


def add_indicators(df):
    df = df.copy()
    df["SMA50"] = df.Close.rolling(50).mean()
    df["SMA200"] = df.Close.rolling(200).mean()
    df["RSI"] = rsi(df.Close)
    df["MACD"], df["SIGNAL"] = macd(df.Close)
    return df


def trend_score(df):
    """-4 (güçlü düşüş eğilimi) ... +4 (güçlü yükseliş eğilimi). Tahmin değil, gösterge özetidir."""
    last = df.iloc[-1]
    s = 0
    s += 1 if last.RSI < 30 else (-1 if last.RSI > 70 else 0)
    s += 1 if last.MACD > last.SIGNAL else -1
    s += 1 if last.Close > last.SMA50 else -1
    s += 1 if last.SMA50 > last.SMA200 else -1
    return s


def label(s):
    return "Yükseliş eğilimi" if s >= 2 else ("Düşüş eğilimi" if s <= -2 else "Nötr")


st.title("📈 BIST Hisse Analiz")

symbols = st.sidebar.multiselect("Takip listesi", DEFAULTS, default=DEFAULTS)
extra = st.sidebar.text_input("Başka hisse ekle (örn. PGSUS)").upper().strip()
if extra and extra not in symbols:
    symbols.append(extra)

# ---- Özet tablo ----
rows = []
for sym in symbols:
    df = load(sym, "5y")
    if df.empty or len(df) < 200:
        continue
    df = add_indicators(df)
    ret = df.Close.pct_change()
    s = trend_score(df)
    rows.append({
        "Hisse": sym,
        "Fiyat (₺)": round(df.Close.iloc[-1], 2),
        "1 Gün %": round(ret.iloc[-1] * 100, 2),
        "1 Yıl %": round((df.Close.iloc[-1] / df.Close.iloc[-252] - 1) * 100, 1) if len(df) > 252 else np.nan,
        "5 Yıl %": round((df.Close.iloc[-1] / df.Close.iloc[0] - 1) * 100, 1),
        "Volatilite %": round(ret.std() * np.sqrt(252) * 100, 1),
        "RSI": round(df.RSI.iloc[-1], 1),
        "Skor": s,
        "Eğilim": label(s),
    })

if rows:
    st.subheader("Takip listesi")
    st.dataframe(pd.DataFrame(rows).set_index("Hisse"), use_container_width=True)
else:
    st.warning("Veri alınamadı. İnternet bağlantını ve hisse kodlarını kontrol et.")

# ---- Uyarılar ----
import json
from pathlib import Path

ALERT_FILE = Path(__file__).with_name("alerts.json")


def save_alerts():
    ALERT_FILE.write_text(json.dumps(st.session_state.alerts, ensure_ascii=False))


if "alerts" not in st.session_state:
    try:
        st.session_state.alerts = json.loads(ALERT_FILE.read_text())
    except Exception:
        st.session_state.alerts = []

TYPES = ["Fiyat şunun üstüne çıkarsa", "Fiyat şunun altına inerse", "RSI şunun üstüne çıkarsa", "RSI şunun altına inerse"]

with st.sidebar.expander("🔔 Uyarılar", expanded=False):
    a_sym = st.selectbox("Hisse", symbols or DEFAULTS, key="a_sym")
    a_type = st.selectbox("Koşul", TYPES, key="a_type")
    a_val = st.number_input("Değer", value=100.0, key="a_val")
    if st.button("Uyarı ekle"):
        st.session_state.alerts.append((a_sym, a_type, a_val))
        save_alerts()
    for s_, t_, v_ in st.session_state.alerts:
        st.write(f"{s_}: {t_} {v_:g}")
    if st.session_state.alerts and st.button("Hepsini temizle"):
        st.session_state.alerts = []
        save_alerts()
        st.rerun()

latest = {r["Hisse"]: r for r in rows}
for s_, t_, v_ in st.session_state.alerts:
    r = latest.get(s_)
    if not r:
        continue
    hit = (
        (t_ == TYPES[0] and r["Fiyat (₺)"] > v_)
        or (t_ == TYPES[1] and r["Fiyat (₺)"] < v_)
        or (t_ == TYPES[2] and r["RSI"] > v_)
        or (t_ == TYPES[3] and r["RSI"] < v_)
    )
    if hit:
        st.warning(f"🔔 {s_}: {t_} {v_:g} (şu an fiyat {r['Fiyat (₺)']} ₺, RSI {r['RSI']})")

# ---- Detay grafik ----
if rows:
    st.subheader("Detay")
    c1, c2 = st.columns(2)
    sel = c1.selectbox("Hisse", [r["Hisse"] for r in rows])
    per = c2.selectbox("Dönem", ["1y", "3y", "5y", "10y", "max"], index=2)
    d = add_indicators(load(sel, "max" if per == "max" else "10y"))
    if per not in ("max", "10y"):
        d = d.iloc[-{"1y": 252, "3y": 756, "5y": 1260}[per]:]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.04)
    fig.add_trace(go.Candlestick(x=d.index, open=d.Open, high=d.High, low=d.Low, close=d.Close, name="Fiyat"), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d.SMA50, name="SMA50"), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d.SMA200, name="SMA200"), row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d.RSI, name="RSI"), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", row=2, col=1)
    fig.update_layout(height=650, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

# ---- Temel veriler ----
@st.cache_data(ttl=3600)
def fundamentals(sym: str) -> dict:
    try:
        return yf.Ticker(sym + ".IS").info or {}
    except Exception:
        return {}


if rows:
    st.subheader("Temel veriler")
    info = fundamentals(sel)

    def fmt(key, pct=False, big=False):
        v = info.get(key)
        if v is None:
            return "—"
        if pct:
            return f"%{v * 100:.1f}"
        if big:
            return f"{v / 1e9:.1f} milyar ₺"
        return f"{v:.2f}"

    f1, f2c, f3, f4 = st.columns(4)
    f1.metric("Piyasa değeri", fmt("marketCap", big=True))
    f2c.metric("F/K", fmt("trailingPE"))
    f3.metric("PD/DD", fmt("priceToBook"))
    f4.metric("Borç/Özkaynak", fmt("debtToEquity"))
    g1, g2, g3, g4 = st.columns(4)
    g1.metric("Özkaynak karlılığı", fmt("returnOnEquity", pct=True))
    g2.metric("Net kar marjı", fmt("profitMargins", pct=True))
    g3.metric("52 hafta yüksek", fmt("fiftyTwoWeekHigh"))
    g4.metric("52 hafta düşük", fmt("fiftyTwoWeekLow"))
    st.caption("Veriler Yahoo Finance'ten gelir; BIST hisselerinde bazı alanlar eksik veya gecikmeli olabilir (— ile gösterilir). Kesin değerler için KAP'a bakın.")

# ---- Geriye dönük test ----
if rows:
    st.subheader("Geriye dönük test: skor ≥ 2 iken hissede, değilse nakitte")
    b = add_indicators(load(sel, "max")).dropna(subset=["SMA200"]).copy()
    b["score"] = (
        (b.RSI < 30).astype(int) - (b.RSI > 70).astype(int)
        + np.where(b.MACD > b.SIGNAL, 1, -1)
        + np.where(b.Close > b.SMA50, 1, -1)
        + np.where(b.SMA50 > b.SMA200, 1, -1)
    )
    ret = b.Close.pct_change().fillna(0)
    pos = (b.score.shift(1) >= 2).astype(int)  # önceki günün sinyaliyle işlem
    eq_strat = (1 + ret * pos).cumprod()
    eq_hold = (1 + ret).cumprod()

    def max_dd(eq):
        return (eq / eq.cummax() - 1).min() * 100

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Strateji getirisi", f"%{(eq_strat.iloc[-1] - 1) * 100:.0f}")
    m2.metric("Al-tut getirisi", f"%{(eq_hold.iloc[-1] - 1) * 100:.0f}")
    m3.metric("Strateji maks. düşüş", f"%{max_dd(eq_strat):.0f}")
    m4.metric("Al-tut maks. düşüş", f"%{max_dd(eq_hold):.0f}")

    f2 = go.Figure()
    f2.add_trace(go.Scatter(x=eq_strat.index, y=eq_strat, name="Strateji"))
    f2.add_trace(go.Scatter(x=eq_hold.index, y=eq_hold, name="Al-tut"))
    f2.update_layout(height=350, yaxis_title="Büyüme (1 = başlangıç)")
    st.plotly_chart(f2, use_container_width=True)
    st.caption("Komisyon, vergi ve kayma dahil değildir. Geçmiş performans gelecekte tekrarlanmayabilir.")

st.caption("Bu program yatırım tavsiyesi değildir. Skorlar geçmiş verilere dayalı teknik göstergelerin özetidir, gelecekteki getiriyi garanti etmez.")
