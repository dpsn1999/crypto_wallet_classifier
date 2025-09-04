"""
Auto-label real Ethereum transactions for fine-tuning.

- Pulls recent normal transactions for a list of seed addresses (contracts or EOAs)
- Labels with:
   1) Hard rules (contract deployment, simple transfer)
   2) Known protocol mapping (by 'to' address)
   3) Function selector hints (first 4 bytes of input)
   4) Fallback = Unknown
- Writes JSONL rows to data/tx_finetune.jsonl (append-safe; dedup by tx hash)

Usage (from project root):
  python scripts/auto_label.py --addresses 0x59728544B08AB483533076417FbBB2fD0B17CE3a 0x00000000006c3852cbEf3e08E8dF289169EdE581 --per 25

Requires:
- .env with ETHERSCAN_API_KEY
- etherscan.fetch_transactions(address, limit) available
"""

import argparse
import json
import os
import pathlib
import re
from dotenv import load_dotenv

# ---- import your existing fetcher ----
from etherscan import fetch_transactions

load_dotenv()

OUT_PATH = pathlib.Path("C:/Users/nedun/Desktop/crypto_wallet_classifier/data.jsonl")


OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
DEDUP_PATH = pathlib.Path("C:/Users/nedun\Desktop/crypto_wallet_classifier/upended_data.jsonl")

ZERO = "0x0000000000000000000000000000000000000000"

# === 1) Known protocol address → canonical label ==============================
#  (Keep these lean to avoid "invalid tests". Use them mainly to bootstrap dataset size.)
KNOWN_ADDR = {
    # NFT marketplaces
    "0x00000000006c3852cbef3e08e8df289169ede581": "NFT Purchase",   # Seaport
    "0x59728544b08ab483533076417fbbb2fd0b17ce3a": "NFT Purchase",   # LooksRare
    "0x000000000000ad05ccc4f10045630fb830b95127": "NFT Purchase",   # Blur

    # Swaps (DEX routers)
    "0x7a250d5630b4cf539739df2c5dacb4c659f2488d": "Token Swap",     # Uniswap V2 Router
    "0xe592427a0aece92de3edee1f18e0157c05861564": "Token Swap",     # Uniswap V3 Router
    "0x1111111254fb6c44bac0bed2854e76f90643097d": "Token Swap",     # 1inch Router

    # DeFi protocols
    "0xc36442b4a4522e871399cd717abdd847ab11fe88": "DeFi Interaction", # UniV3 Positions
    "0x7d2768de32b0b80b7a3454c06bdac94a69bf6b8": "DeFi Interaction",  # Aave v2 LendingPool
    "0x3d9819210a31b4961b30ef54be2aed79b9c9cd3b": "DeFi Interaction", # Compound Comptroller
    "0xbebc44782c7db0a1a60cb6fe97d0b483032ff1c7": "DeFi Interaction", # Curve 3pool
    "0x5ef30b9986345249bc32d8928b7ee64de9435e39": "DeFi Interaction", # Maker CDP Manager
}

# === 2) Function selectors → hint label ======================================
#  (These help when 'to' is unfamiliar but calldata is recognizable.)
SELECTOR_HINTS = {
    # Swaps
    "0x7ff36ab5": "Token Swap",   # swapExactETHForTokens
    "0x18cbafe5": "Token Swap",   # swapExactTokensForETH
    "0x38ed1739": "Token Swap",   # swapExactTokensForTokens
    "0x414bf389": "Token Swap",   # exactInput (Uni V3 style, also used in other routers)
    # NFT Seaport / LooksRare (common trade funcs)
    "0xf242432a": "NFT Purchase", # seaport fulfillBasicOrder (encoded in some traces)
    "0x3593564c": "NFT Purchase", # fulfillAdvancedOrder
    "0xab834bab": "NFT Purchase", # LooksRare matchOrder
    # DeFi common calls
    "0x095ea7b3": "DeFi Interaction", # approve
    "0xd0e30db0": "DeFi Interaction", # deposit (WETH) / generic deposit
    "0x2e1a7d4d": "DeFi Interaction", # withdraw (WETH)
    "0xe8e33700": "DeFi Interaction", # addLiquidity (var.)
    "0x219f5d17": "DeFi Interaction", # increaseLiquidity (UniV3)
}

CATEGORIES = [
    "Token Swap",
    "NFT Purchase",
    "DeFi Interaction",
    "Simple Transfer",
    "Contract Deployment",
    "Unknown",
]

def canonical_label(s: str) -> str:
    s = (s or "").strip().lower()
    for c in CATEGORIES:
        if s == c.lower():
            return c
    for c in CATEGORIES:
        if c.lower() in s:
            return c
    # aliases
    if "swap" in s: return "Token Swap"
    if "nft" in s or "purchase" in s: return "NFT Purchase"
    if "defi" in s or "liquidity" in s or "stake" in s: return "DeFi Interaction"
    if "transfer" in s or "send" in s: return "Simple Transfer"
    if "deploy" in s or "creation" in s: return "Contract Deployment"
    return "Unknown"

def cheap_label(tx: dict) -> str | None:
    to = (tx.get("to") or "").lower()
    input_data = (tx.get("input") or "0x").lower()

    # Contract deployment
    if not to or to == "0x" or to == ZERO:
        return "Contract Deployment"

    # Simple transfer (no calldata)
    if input_data == "0x":
        return "Simple Transfer"

    # Known address
    if to in KNOWN_ADDR:
        return KNOWN_ADDR[to]

    # Function selector hint (first 10 chars inc '0x')
    if input_data.startswith("0x") and len(input_data) >= 10:
        sel = input_data[:10]
        if sel in SELECTOR_HINTS:
            return SELECTOR_HINTS[sel]

    return None

def to_jsonl_row(tx: dict, label: str) -> dict:
    # Build the exact training row format
    prompt = (
        "Classify the Ethereum transaction.\n"
        f"From: {tx.get('from')}\n"
        f"To: {tx.get('to')}\n"
        f"Value(wei): {tx.get('value')}\n"
        f"Input: {tx.get('input')}\n"
        "Return one label: Token Swap | NFT Purchase | DeFi Interaction | Simple Transfer | Contract Deployment | Unknown\n"
        "Category:"
    )
    return {"prompt": prompt, "completion": f" {canonical_label(label)}"}

def load_seen() -> set[str]:
    if DEDUP_PATH.is_file():
        try:
            return set(json.loads(DEDUP_PATH.read_text(encoding="utf-8")))
        except Exception:
            return set()
    return set()

def save_seen(seen: set[str]):
    DEDUP_PATH.write_text(json.dumps(sorted(seen)), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--addresses", nargs="+", required=True, help="Seed addresses (contracts or EOAs)")
    ap.add_argument("--per", type=int, default=25, help="Transactions per address (recent)")
    ap.add_argument("--unknown_ok", action="store_true", help="Include Unknown rows (default: include)")
    ap.add_argument("--no_unknown", action="store_true", help="Exclude Unknown rows")
    args = ap.parse_args()

    include_unknown = not args.no_unknown

    seen = load_seen()
    appended = 0

    with OUT_PATH.open("a", encoding="utf-8") as fout:
        for addr in args.addresses:
            addr = addr.strip()
            try:
                txs = fetch_transactions(addr, limit=args.per)
            except Exception as e:
                print(f"[WARN] fetch failed for {addr}: {e}")
                continue

            for tx in txs:
                h = tx.get("hash")
                if not h or h in seen:
                    continue

                label = cheap_label(tx) or "Unknown"
                if (label == "Unknown") and (not include_unknown):
                    continue

                row = to_jsonl_row(tx, label)
                fout.write(json.dumps(row) + "\n")
                seen.add(h)
                appended += 1

    save_seen(seen)
    print(f"Appended {appended} new examples to {OUT_PATH}.")

if __name__ == "__main__":
    main()
