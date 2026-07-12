"""
streamlit_app.py
================
Streamlit dashboard for the cross-exchange crypto arbitrage PAPER scanner.

Deploy on Streamlit Cloud:
  repo root must contain:
    streamlit_app.py      <- this file
    requirements.txt      <- ccxt, streamlit, pandas

NOTE on Streamlit Cloud: its servers run on US cloud IPs. Binance and Bybit
geo-block US IPs, so those venues will typically return zero quotes when
deployed there (they still work when you run locally in Canada). The health
panel below shows exactly which exchanges are responding.

Paper trading only. No order placement code exists in this app.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone

import ccxt
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
ALL_EXCHANGES = ["kraken", "kucoin", "okx", "binance", "bybit", "gate", "coinbase"]

ALL_SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT",
    "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "LINK/USDT",
]

TAKER_FEES = {          # default published taker tiers — set to YOUR tier
    "binance": 0.0010,
    "kraken":  0.0026,
    "kucoin":  0.0010,
    "bybit":   0.0010,
    "okx":     0.0010,
    "gate":    0.0020,
    "coinbase": 0.0060,
}

ORDER_BOOK_DEPTH = 20

# ----------------------------------------------------------------------------
# Core math (identical logic to the CLI version)
# ----------------------------------------------------------------------------
@dataclass
class Quote:
    exchange: str
    symbol: str
    vwap_ask: float
    vwap_bid: float

def vwap_through_book(levels, target_usd):
    """Depth-weighted average price to fill target_usd notional; None if thin."""
    remaining, cost, qty = target_usd, 0.0, 0.0
    for price, size in levels:
        take_usd = min(remaining, price * size)
        take_qty = take_usd / price
        cost += take_qty * price
        qty += take_qty
        remaining -= take_usd
        if remaining <= 1e-9:
            return cost / qty
    return None

@st.cache_resource(show_spinner=False)
def get_clients(exchange_ids):
    clients = {}
    for ex_id in exchange_ids:
        try:
            clients[ex_id] = getattr(ccxt, ex_id)(
                {"enableRateLimit": True, "timeout": 10_000}
            )
        except Exception:
            pass
    return clients

def fetch_quote(client, ex_id, symbol, clip_usd):
    try:
        ob = client.fetch_order_book(symbol, limit=ORDER_BOOK_DEPTH)
        ask = vwap_through_book(ob["asks"], clip_usd)
        bid = vwap_through_book(ob["bids"], clip_usd)
        if ask is None or bid is None:
            return None, ex_id, "thin book"
        return Quote(ex_id, symbol, ask, bid), ex_id, "ok"
    except ccxt.ExchangeNotAvailable:
        return None, ex_id, "geo-blocked / unavailable"
    except ccxt.BaseError as e:
        return None, ex_id, type(e).__name__
    except Exception as e:
        return None, ex_id, type(e).__name__

def scan(clients, symbols, clip_usd):
    quotes, health = [], {ex: {"ok": 0, "fail": 0, "last_err": ""} for ex in clients}
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [
            pool.submit(fetch_quote, cl, ex_id, sym, clip_usd)
            for ex_id, cl in clients.items()
            for sym in symbols
        ]
        for fut in as_completed(futures):
            q, ex_id, status = fut.result()
            if q:
                quotes.append(q)
                health[ex_id]["ok"] += 1
            else:
                health[ex_id]["fail"] += 1
                health[ex_id]["last_err"] = status
    return quotes, health

def find_opportunities(quotes, clip_usd):
    rows, by_symbol = [], {}
    for q in quotes:
        by_symbol.setdefault(q.symbol, []).append(q)
    now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    for symbol, qs in by_symbol.items():
        for b in qs:
            for s in qs:
                if b.exchange == s.exchange or s.vwap_bid <= b.vwap_ask:
                    continue
                gross = (s.vwap_bid - b.vwap_ask) / b.vwap_ask * 10_000
                fees = (TAKER_FEES.get(b.exchange, 0.002)
                        + TAKER_FEES.get(s.exchange, 0.002)) * 10_000
                net = gross - fees
                if net <= 0:
                    continue
                rows.append({
                    "time": now, "symbol": symbol,
                    "buy on": b.exchange, "sell on": s.exchange,
                    "buy px": round(b.vwap_ask, 6),
                    "sell px": round(s.vwap_bid, 6),
                    "gross bps": round(gross, 2),
                    "fees bps": round(fees, 2),
                    "net bps": round(net, 2),
                    "net pnl $": round(clip_usd * net / 10_000, 2),
                })
    return sorted(rows, key=lambda r: r["net bps"], reverse=True)

# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Crypto Arb Paper Scanner", layout="wide")
st.title("Cross-Exchange Arbitrage — Paper Scanner")
st.caption(
    "Fee-aware, depth-weighted spread scanner. **Paper trading only** — "
    "simulated fills assume zero latency; live results will be worse."
)

with st.sidebar:
    st.header("Config")
    exchanges = st.multiselect("Exchanges", ALL_EXCHANGES,
                               default=["kraken", "kucoin", "okx", "gate"])
    symbols = st.multiselect("Symbols", ALL_SYMBOLS, default=ALL_SYMBOLS[:6])
    clip_usd = st.number_input("Clip size (USD)", 500, 50_000, 5_000, step=500)
    min_edge = st.number_input("Min net edge (bps) to paper-trade",
                               1.0, 100.0, 5.0, step=0.5)
    st.divider()
    st.caption(
        "On Streamlit Cloud (US IPs), Binance and Bybit are usually "
        "geo-blocked — check the health panel. Running locally in Canada "
        "restores them."
    )

# Session-state paper book
if "pnl" not in st.session_state:
    st.session_state.pnl = 0.0
    st.session_state.trades = 0
    st.session_state.passes = 0
    st.session_state.trade_log = []

col_run, col_auto = st.columns([1, 3])
run = col_run.button("Scan now", type="primary", use_container_width=True)
auto = col_auto.toggle("Auto-rescan every 15s", value=False)

if run or auto:
    if not exchanges or not symbols:
        st.warning("Pick at least one exchange and one symbol.")
        st.stop()
    clients = get_clients(tuple(sorted(exchanges)))
    with st.spinner("Fetching order books..."):
        t0 = time.time()
        quotes, health = scan(clients, symbols, clip_usd)
        opps = find_opportunities(quotes, clip_usd)
    st.session_state.passes += 1

    # paper-execute qualifying opportunities (one per symbol/venue-pair)
    seen = set()
    for o in opps:
        key = (o["symbol"], o["buy on"], o["sell on"])
        if o["net bps"] >= min_edge and key not in seen:
            seen.add(key)
            st.session_state.pnl += o["net pnl $"]
            st.session_state.trades += 1
            st.session_state.trade_log.append(o)

    # ---- metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Scan passes", st.session_state.passes)
    m2.metric("Paper trades", st.session_state.trades)
    m3.metric("Cumulative paper P&L", f"${st.session_state.pnl:,.2f}")
    m4.metric("Quotes this pass", f"{len(quotes)} ({time.time()-t0:.1f}s)")

    # ---- exchange health
    with st.expander("Exchange health", expanded=any(
            h["ok"] == 0 for h in health.values())):
        hdf = pd.DataFrame([
            {"exchange": ex, "quotes ok": h["ok"], "failed": h["fail"],
             "last error": h["last_err"]}
            for ex, h in health.items()
        ])
        st.dataframe(hdf, hide_index=True, use_container_width=True)

    # ---- opportunities
    st.subheader("Net-positive opportunities this pass")
    if opps:
        st.dataframe(pd.DataFrame(opps), hide_index=True,
                     use_container_width=True)
    else:
        st.info(
            "None after fees — this is the normal state on liquid pairs. "
            "Persistent zero across many passes is itself the answer."
        )

    # ---- trade log download
    if st.session_state.trade_log:
        log_df = pd.DataFrame(st.session_state.trade_log)
        st.subheader("Paper trade log (this session)")
        st.dataframe(log_df.tail(50), hide_index=True,
                     use_container_width=True)
        st.download_button(
            "Download full log (CSV)",
            log_df.to_csv(index=False).encode(),
            "arb_paper_trades.csv", "text/csv",
        )

    if auto:
        time.sleep(15)
        st.rerun()
else:
    st.info("Press **Scan now** to run a pass, or enable auto-rescan.")
