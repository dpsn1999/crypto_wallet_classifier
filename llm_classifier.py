#Uses Hugging Face base model + LoRA adapter (loaded in hf_infer.py)
# No Ollama calls anywhere.

from typing import Dict, Optional
import re
from typing import List, Dict, DefaultDict
from collections import defaultdict
from fastapi import Query

# Import the LoRA-powered classifier (loads model once, then reuses it)
from hf_infer import classify_with_lora

# -------------------- Labels & simple rules --------------------
CATEGORIES = [
    "Token Swap", "NFT Purchase", "DeFi Interaction",
    "Simple Transfer", "Contract Deployment", "Unknown",
]

ZERO = "0x0000000000000000000000000000000000000000"

# Obvious destinations we can map instantly
KNOWN = {
    # Seaport (NFT marketplace)
    "0x00000000006c3852cbef3e08e8df289169ede581": "NFT Purchase",
    # Uniswap V2 & V3 routers (swaps)
    "0x7a250d5630b4cf539739df2c5dacb4c659f2488d": "Token Swap",
    "0xe592427a0aece92de3edee1f18e0157c05861564": "Token Swap",
    # 1inch router
    "0x1111111254fb6c44bac0bed2854e76f90643097d": "Token Swap",
    # Uniswap V3 Nonfungible Position Manager (LP mint/add/remove)
    "0xc36442b4a4522e871399cd717abdd847ab11fe88": "DeFi Interaction",
    # Aave v2 LendingPool
    "0x7d2768de32b0b80b7a3454c06bdac94a69bf6b8": "DeFi Interaction",
}

# Function selectors (first 4 bytes) that give strong hints
SELECTOR_HINTS = {
    # Swaps
    "0x7ff36ab5": "Token Swap",
    "0x18cbafe5": "Token Swap",
    "0x38ed1739": "Token Swap",
    "0x414bf389": "Token Swap",
    # NFTs / marketplaces (examples)
    "0xf242432a": "NFT Purchase",
    "0x3593564c": "NFT Purchase",
    "0xab834bab": "NFT Purchase",
    # Common DeFi ops
    "0x095ea7b3": "DeFi Interaction",  # approve
    "0xd0e30db0": "DeFi Interaction",  # deposit (WETH)
    "0x2e1a7d4d": "DeFi Interaction",  # withdraw (WETH)
    "0xe8e33700": "DeFi Interaction",
    "0x219f5d17": "DeFi Interaction",
}

def _cheap_heuristics(tx: Dict, use_heuristics: bool = True) -> Optional[str]:
    """Fast path: deployment / known addresses / empty input / known selectors."""
    if not use_heuristics:
        return None
    to = (tx.get("to") or "").lower()
    inp = (tx.get("input") or "0x").lower()

    # Contract creation: no 'to' or to == ZERO
    if (not to) or to in ("0x", ZERO):
        return "Contract Deployment"

    # Known destination contracts
    if to in KNOWN:
        return KNOWN[to]

    # No calldata: simple ETH transfer
    if inp == "0x":
        return "Simple Transfer"

    # Strong selector hints (first 4 bytes after 0x => first 10 chars total incl '0x')
    if inp.startswith("0x") and len(inp) >= 10:
        sel = inp[:10]
        if sel in SELECTOR_HINTS:
            return SELECTOR_HINTS[sel]

    return None

def _impossible_guard(tx: Dict, label: str) -> str:
    """Light sanity check: cannot be 'Contract Deployment' if 'to' is a nonzero address."""
    to = (tx.get("to") or "").lower()
    if label == "Contract Deployment" and to and to not in ("0x", ZERO):
        return "Unknown"
    return label

# -------------------- Public API --------------------
def classify_transaction(
    tx: Dict,
    model: str = None,          # kept for backward-compat (ignored)
    shots: int = 0,             # kept for backward-compat (ignored)
    use_heuristics: bool = True,
    seed: int = 42,             # kept for backward-compat (ignored)
) -> str:
    """
    Classify an Ethereum transaction into one of:
    Token Swap | NFT Purchase | DeFi Interaction | Simple Transfer | Contract Deployment | Unknown

    Uses:
      1) quick heuristics for obvious cases
      2) your fine-tuned HF LoRA model for everything else
    """
    # 1) obvious cases
    h = _cheap_heuristics(tx, use_heuristics=use_heuristics)
    if h:
        return h

    # 2) LoRA model (strict JSON output + normalization handled inside hf_infer)
    label = classify_with_lora(tx)

    # 3) final guardrail
    return _impossible_guard(tx, label)



    
