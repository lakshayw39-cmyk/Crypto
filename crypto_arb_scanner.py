"""
crypto_arb_scanner.py
=====================
Cross-exchange crypto arbitrage scanner + PAPER-TRADING simulator.

Philosophy (same as the Ichimoku suite):
  - No look-ahead: decisions use only the order book snapshot in hand.
  - Conservative fills: buy at depth-weighted ASK, sell at depth-weighted BID
    (VWAP through the book for the full clip size, not top-of-book fantasy).
  - Full cost accounting: taker fees on BOTH legs deducted from every trade.
  - Paper mode ONLY by default. No order placement code paths are armed.

Model:
  Pre-funded inventory arbitrage. Assumes you hold USDT + coin on BOTH
  exchanges simultaneously, so both legs execute at snapshot time with no
  blockchain transfer (transfer-based arb is almost always dead on arrival
  because the spread closes during the 2-30 min transfer window).

Usage:
  pip install ccxt
  python crypto_arb_scanner.py                # continuous scan, paper trade
  python crypto_arb_scanner.py --once         # single scan pass, print table
  python crypto_arb_scanner.py --report       # P&L report from trade log

Outputs:
  arb_trades.csv          every simulated round trip
  arb_opportunities.csv   every net-positive spread observed (traded or not)
"""

import argparse
import csv
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone

import ccxt

# ----------------------------------------------------------------------------
# CONFIG — standard conventions: $100K capital, $5K per position
# ----------------------------------------------------------------------------
CAPITAL_USD          = 100_000.0
CLIP_USD             = 5_000.0        # notional per arbitrage round trip
MIN_NET_EDGE_BPS     = 5.0            # only "trade" if net edge >= 5 bps after fees
MAX_OPEN_CLIPS       = 10             # cash constraint: 10 x $5K = $50K deployed max
SCAN_INTERVAL_SEC    = 10             # pause between scan passes
ORDER_BOOK_DEPTH     = 20             # levels fetched per side

EXCHANGES = ["binance", "kraken", "kucoin", "bybit", "okx"]

SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT",
    "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "LINK/USDT",
]

# Taker fees (default published tiers, decimal). VERIFY against your actual
# account tier — fee tier is the single biggest determinant of viability.
TAKER_FEES = {
    "binance": 0.0010,
    "kraken":  0.0026,
    "kucoin":  0.0010,
    "bybit":   0.0010,
    "okx":     0.0010,
}

TRADES_CSV = "arb_trades.csv"
OPPS_CSV   = "arb_opportunities.csv"

# ----------------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------------
@dataclass
class Quote:
    exchange: str
    symbol: str
    vwap_ask: float      # depth-weighted price to BUY clip_usd notional
    vwap_bid: float      # depth-weighted price to SELL clip_usd notional
    ts: float

@dataclass
class Opportunity:
    ts: str
    symbol: str
    buy_ex: str
    sell_ex: str
    buy_px: float
    sell_px: float
    gross_bps: float
    fees_bps: float
    net_bps: float
    net_pnl_usd: float

@dataclass
class PaperBook:
    """Tracks simulated capital and realized P&L."""
    cash: float = CAPITAL_USD
    realized_pnl: float = 0.0
    trades: int = 0
    wins: int = 0
    open_clips: int = 0  # clips 'in flight' this scan pass (reset each pass)

# ----------------------------------------------------------------------------
# Market data
# ----------------------------------------------------------------------------
def build_clients():
    clients = {}
    for ex_id in EXCHANGES:
        try:
            klass = getattr(ccxt, ex_id)
            clients[ex_id] = klass({"enableRateLimit": True, "timeout": 10_000})
        except Exception as e:
            print(f"[warn] could not init {ex_id}: {e}")
    return clients

def vwap_through_book(levels, target_usd):
    """
    Walk order book levels [(price, size), ...] until target_usd notional is
    filled. Returns depth-weighted avg price, or None if book too thin.
    Thin-book rejection is a feature: 'opportunities' you can't fill at size
    are not opportunities.
    """
    remaining = target_usd
    cost = 0.0
    qty = 0.0
    for price, size in levels:
        level_usd = price * size
        take_usd = min(remaining, level_usd)
        take_qty = take_usd / price
        cost += take_qty * price
        qty += take_qty
        remaining -= take_usd
        if remaining <= 1e-9:
            return cost / qty
    return None  # insufficient depth for the clip

def fetch_quote(client, ex_id, symbol):
    try:
        ob = client.fetch_order_book(symbol, limit=ORDER_BOOK_DEPTH)
        ask = vwap_through_book(ob["asks"], CLIP_USD)
        bid = vwap_through_book(ob["bids"], CLIP_USD)
        if ask is None or bid is None:
            return None
        return Quote(ex_id, symbol, ask, bid, time.time())
    except Exception:
        return None  # symbol unlisted / rate limit / network — skip silently

def scan_pass(clients):
    """Fetch all (exchange, symbol) quotes concurrently. Returns quotes list."""
    quotes = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {
            pool.submit(fetch_quote, cl, ex_id, sym): (ex_id, sym)
            for ex_id, cl in clients.items()
            for sym in SYMBOLS
        }
        for fut in as_completed(futures):
            q = fut.result()
            if q:
                quotes.append(q)
    return quotes

# ----------------------------------------------------------------------------
# Opportunity detection — fee-aware, both legs
# ----------------------------------------------------------------------------
def find_opportunities(quotes):
    opps = []
    by_symbol = {}
    for q in quotes:
        by_symbol.setdefault(q.symbol, []).append(q)

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for symbol, qs in by_symbol.items():
        for buy_q in qs:
            for sell_q in qs:
                if buy_q.exchange == sell_q.exchange:
                    continue
                buy_px, sell_px = buy_q.vwap_ask, sell_q.vwap_bid
                if sell_px <= buy_px:
                    continue
                gross_bps = (sell_px - buy_px) / buy_px * 10_000
                fees_bps = (TAKER_FEES[buy_q.exchange]
                            + TAKER_FEES[sell_q.exchange]) * 10_000
                net_bps = gross_bps - fees_bps
                if net_bps <= 0:
                    continue
                net_pnl = CLIP_USD * net_bps / 10_000
                opps.append(Opportunity(
                    now, symbol, buy_q.exchange, sell_q.exchange,
                    round(buy_px, 6), round(sell_px, 6),
                    round(gross_bps, 2), round(fees_bps, 2),
                    round(net_bps, 2), round(net_pnl, 2),
                ))
    opps.sort(key=lambda o: o.net_bps, reverse=True)
    return opps

# ----------------------------------------------------------------------------
# Paper execution
# ----------------------------------------------------------------------------
def paper_execute(book: PaperBook, opps):
    """
    Simulate execution of qualifying opportunities.
    Conservative assumptions already baked into the quotes (VWAP through book,
    taker fees both legs). One clip per (symbol, ex-pair) per pass; cash cap.
    """
    executed = []
    seen = set()
    book.open_clips = 0
    for o in opps:
        if o.net_bps < MIN_NET_EDGE_BPS:
            continue
        key = (o.symbol, o.buy_ex, o.sell_ex)
        if key in seen:
            continue
        if book.open_clips >= MAX_OPEN_CLIPS:
            break
        seen.add(key)
        book.open_clips += 1
        book.realized_pnl += o.net_pnl_usd
        book.cash += o.net_pnl_usd
        book.trades += 1
        if o.net_pnl_usd > 0:
            book.wins += 1
        executed.append(o)
    return executed

# ----------------------------------------------------------------------------
# Logging / reporting
# ----------------------------------------------------------------------------
FIELDS = ["ts", "symbol", "buy_ex", "sell_ex", "buy_px", "sell_px",
          "gross_bps", "fees_bps", "net_bps", "net_pnl_usd"]

def append_csv(path, rows):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for o in rows:
            w.writerow(o.__dict__)

def print_table(opps, limit=15):
    if not opps:
        print("  no net-positive opportunities this pass "
              "(this is the normal state — fees eat almost everything)")
        return
    print(f"  {'symbol':<10}{'buy@':<10}{'sell@':<10}"
          f"{'gross':>8}{'fees':>8}{'net':>8}{'pnl$':>9}")
    for o in opps[:limit]:
        print(f"  {o.symbol:<10}{o.buy_ex:<10}{o.sell_ex:<10}"
              f"{o.gross_bps:>7.1f}b{o.fees_bps:>7.1f}b"
              f"{o.net_bps:>7.1f}b{o.net_pnl_usd:>9.2f}")

def report():
    if not os.path.exists(TRADES_CSV):
        print("No trade log yet. Run the scanner first.")
        return
    rows = list(csv.DictReader(open(TRADES_CSV)))
    if not rows:
        print("Trade log is empty.")
        return
    pnl = [float(r["net_pnl_usd"]) for r in rows]
    net_bps = [float(r["net_bps"]) for r in rows]
    total = sum(pnl)
    wins = sum(1 for p in pnl if p > 0)
    print("=" * 60)
    print("PAPER ARBITRAGE REPORT")
    print("=" * 60)
    print(f"Simulated trades      : {len(pnl)}")
    print(f"Win rate              : {wins/len(pnl)*100:.1f}%")
    print(f"Total simulated P&L   : ${total:,.2f}")
    print(f"Avg net edge          : {sum(net_bps)/len(net_bps):.2f} bps")
    print(f"Avg P&L per clip      : ${total/len(pnl):,.2f}")
    print(f"Return on capital     : {total/CAPITAL_USD*100:.3f}%")
    print(f"First trade           : {rows[0]['ts']}")
    print(f"Last trade            : {rows[-1]['ts']}")
    print("-" * 60)
    print("Remember: paper fills assume both legs execute at snapshot")
    print("prices with zero latency. Live results will be WORSE.")

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="single scan pass")
    ap.add_argument("--report", action="store_true", help="print P&L report")
    args = ap.parse_args()

    if args.report:
        report()
        return

    print("Initializing exchange clients (public endpoints, no API keys)...")
    clients = build_clients()
    print(f"Active: {', '.join(clients)} | {len(SYMBOLS)} symbols | "
          f"clip ${CLIP_USD:,.0f} | min edge {MIN_NET_EDGE_BPS} bps\n")

    book = PaperBook()
    passes = 0
    try:
        while True:
            passes += 1
            t0 = time.time()
            quotes = scan_pass(clients)
            opps = find_opportunities(quotes)
            if opps:
                append_csv(OPPS_CSV, opps)
            executed = paper_execute(book, opps)
            if executed:
                append_csv(TRADES_CSV, executed)

            ts = datetime.now().strftime("%H:%M:%S")
            print(f"[{ts}] pass {passes} | quotes {len(quotes)} | "
                  f"net-positive {len(opps)} | paper-traded {len(executed)} | "
                  f"cum P&L ${book.realized_pnl:,.2f} "
                  f"({time.time()-t0:.1f}s)")
            print_table(opps)

            if args.once:
                break
            time.sleep(SCAN_INTERVAL_SEC)
    except KeyboardInterrupt:
        pass
    finally:
        print(f"\nDone. {book.trades} paper trades, "
              f"P&L ${book.realized_pnl:,.2f}, "
              f"return {book.realized_pnl/CAPITAL_USD*100:.3f}%")
        print(f"Logs: {TRADES_CSV}, {OPPS_CSV}. "
              f"Run with --report for the summary.")

if __name__ == "__main__":
    main()
