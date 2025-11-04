"""Check MT5 trades and history"""
import MetaTrader5 as mt5
from datetime import datetime, timedelta

# Initialize MT5
if not mt5.initialize():
    print("Failed to initialize MT5")
    exit()

print("=" * 70)
print("MT5 ACCOUNT STATUS")
print("=" * 70)
info = mt5.account_info()
print(f"Balance:  ${info.balance:,.2f}")
print(f"Equity:   ${info.equity:,.2f}")
print(f"Profit:   ${info.profit:,.2f}")
print(f"Margin:   ${info.margin:,.2f}")
print()

print("=" * 70)
print("CLOSED TRADES (Last 24 hours)")
print("=" * 70)
history = mt5.history_deals_get(datetime.now() - timedelta(days=1), datetime.now())
if history and len(history) > 0:
    for deal in history[-20:]:  # Last 20 deals
        deal_type = "BUY" if deal.type == 0 else "SELL" if deal.type == 1 else "OTHER"
        print(f"{deal.time} | Order:{deal.order} | Ticket:{deal.ticket} | {deal.symbol}")
        print(f"  Type: {deal_type} | Volume: {deal.volume} | Price: {deal.price}")
        print(f"  Profit: ${deal.profit:,.2f} | Commission: ${deal.commission:,.2f}")
        print()
else:
    print("No trade history found")
print()

print("=" * 70)
print("OPEN POSITIONS")
print("=" * 70)
positions = mt5.positions_get()
if positions and len(positions) > 0:
    for p in positions:
        pos_type = "BUY" if p.type == 0 else "SELL"
        print(f"Ticket: {p.ticket} | {p.symbol} | {pos_type}")
        print(f"  Volume: {p.volume} | Entry: {p.price_open} | Current: {p.price_current}")
        print(f"  SL: {p.sl} | TP: {p.tp}")
        print(f"  Profit: ${p.profit:,.2f}")
        print()
else:
    print("No open positions")

mt5.shutdown()
