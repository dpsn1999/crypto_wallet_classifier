import os
import re
from typing import List, Dict, Tuple

import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import llm_classifier as L

# ---------------- Env & config ----------------
load_dotenv()
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")
DEBUG = os.getenv("DEBUG_API", "0") == "1"

ETHERSCAN_URL = "https://api.etherscan.io/api"
ADDRESS_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")

# ---------------- App ----------------
app = FastAPI(title="Crypto Wallet Classifier (LoRA)")

# CORS for browser UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Helpers ----------------
def _validate_address(addr: str) -> None:
    if not ADDRESS_RE.match(addr or ""):
        raise HTTPException(status_code=400, detail="Invalid Ethereum address")

def _es_get(params: Dict, timeout: int = 20) -> List[Dict]:
    if not ETHERSCAN_API_KEY:
        raise HTTPException(status_code=500, detail="ETHERSCAN_API_KEY not configured")
    params = dict(params)
    params["apikey"] = ETHERSCAN_API_KEY
    try:
        r = requests.get(ETHERSCAN_URL, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Etherscan request failed: {e}")

    # OK
    if str(data.get("status")) == "1":
        return data.get("result", [])

    # Gracefully handle 'No transactions found' in either field
    res = data.get("result")
    msg = str(data.get("message") or "")
    if (isinstance(res, str) and "no transactions found" in res.lower()) or \
       ("no transactions found" in msg.lower()):
        return []

    # Everything else is a real error (rate limit, invalid key, etc.)
    raise HTTPException(status_code=502, detail=f"Etherscan error: {msg or res or 'Unknown error'}")


def _fetch_mixed_txs(address: str, limit: int) -> List[Dict]:
    """
    Fetch recent activity for an address:
      - normal transactions (txlist)
      - ERC-20 transfers (tokentx)
      - NFT transfers (tokennfttx)
    We first pull N normal txs, compute the min block among them, then fetch
    a larger window of logs within that block range so hashes overlap.
    """
    limit = max(1, min(limit, 50))

    # 1) Latest N normal txs
    normal = _es_get({
        "module": "account", "action": "txlist",
        "address": address, "startblock": 0, "endblock": 99999999,
        "page": 1, "offset": limit, "sort": "desc",
    })
    if not normal:
        return []

    # 2) Block window to ensure overlap
    blocks: List[int] = []
    for tx in normal:
        try:
            blocks.append(int(tx.get("blockNumber", "0")))
        except Exception:
            pass
    min_block = min(blocks) if blocks else 0
    max_block = 99999999

    # 3) Larger log window in SAME block range
    LOG_WINDOW = 1000

    erc20 = _es_get({
        "module": "account", "action": "tokentx",
        "address": address, "startblock": min_block, "endblock": max_block,
        "page": 1, "offset": LOG_WINDOW, "sort": "desc",
    })

    nft = _es_get({
        "module": "account", "action": "tokennfttx",
        "address": address, "startblock": min_block, "endblock": max_block,
        "page": 1, "offset": LOG_WINDOW, "sort": "desc",
    })

    # 4) Merge by hash
    by_hash: Dict[str, Dict[str, object]] = {}
    for tx in normal:
        by_hash[tx["hash"]] = {"normal": tx, "erc20": [], "nft": []}
    for t in erc20:
        h = t.get("hash")
        if not h:
            continue
        by_hash.setdefault(h, {"normal": None, "erc20": [], "nft": []})
        by_hash[h]["erc20"].append(t)
    for n in nft:
        h = n.get("hash")
        if not h:
            continue
        by_hash.setdefault(h, {"normal": None, "erc20": [], "nft": []})
        by_hash[h]["nft"].append(n)

    # 5) Sort by normal tx timestamp, take first `limit`
    merged: List[Tuple[int, str, Dict[str, object]]] = []
    for h, b in by_hash.items():
        ts = 0
        if b["normal"]:
            try:
                ts = int(b["normal"].get("timeStamp", "0"))
            except Exception:
                ts = 0
        merged.append((ts, h, b))
    merged.sort(key=lambda x: x[0], reverse=True)

    out: List[Dict] = []
    for _, h, b in merged[:limit]:
        nt = b["normal"] or {}
        item = {
            "hash": h,
            "from": nt.get("from"),
            "to": nt.get("to"),
            "value": nt.get("value") or "0",
            "input": nt.get("input") or "0x",
            "erc20": b["erc20"],
            "nft": b["nft"],
        }
        if DEBUG:
            print(h, "erc20:", len(item["erc20"]), "nft:", len(item["nft"]), "input_empty:", (item["input"] or "").lower() == "0x")
        out.append(item)
    return out

# ---------------- API ----------------
@app.get("/api/analyze")
def analyze(
    address: str = Query(..., description="Ethereum wallet address (0x...)"),
    limit: int = Query(10, description="Number of recent transactions to analyze"),
):
    _validate_address(address)
    txs = _fetch_mixed_txs(address, limit)

    results: List[Dict] = []
    for tx in txs:
        # High-precision external heuristics from logs
        if tx["nft"]:
            category = "NFT Purchase"
        elif len({(t.get("contractAddress"), t.get("tokenSymbol")) for t in tx["erc20"]}) >= 2:
            category = "Token Swap"
        else:
            # Let the LoRA model decide; disable its cheap "input==0x => Simple Transfer" rule
            try:
                category = L.classify_transaction(
                    {"from": tx["from"], "to": tx["to"], "value": tx["value"], "input": tx["input"]},
                    shots=6,
                    use_heuristics=False,
                )
            except Exception as e:
                category = f"Error: {e}"

        results.append({"hash": tx["hash"], "category": category})

    return {"address": address, "transactions": results}

@app.get("/health", include_in_schema=False)
def health():
    ok = bool(ETHERSCAN_API_KEY)
    return JSONResponse({"status": "ok" if ok else "misconfigured", "etherscan_key": ok})

# ---------------- Web UI ----------------
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/ui/")

app.mount("/ui", StaticFiles(directory="web", html=True), name="web")

# ---------------- Entrypoint ----------------
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)






