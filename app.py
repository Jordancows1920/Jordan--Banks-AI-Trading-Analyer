import os, io
from pathlib import Path
import pandas as pd
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from analysis import analyse_candles

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(Path(__file__).with_name(".env"))
app = FastAPI(title="TrendScope AI", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/", response_class=HTMLResponse)
def home():
    return (ROOT / "frontend" / "index.html").read_text(encoding="utf-8")

@app.get("/api/health")
def health():
    return {"status":"ok", "app":"TrendScope AI", "live_data_configured": bool(os.getenv("TWELVE_DATA_API_KEY"))}

@app.post("/api/analyse-csv")
async def analyse_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a CSV file.")
    try:
        raw = await file.read()
        df = pd.read_csv(io.BytesIO(raw))
        return analyse_candles(df)
    except Exception as e:
        raise HTTPException(400, str(e))

@app.get("/api/market")
def market(symbol: str = "EUR/USD", interval: str = "1h", outputsize: int = 200):
    key = os.getenv("TWELVE_DATA_API_KEY")
    if not key:
        raise HTTPException(503, "Live data is not configured. Add TWELVE_DATA_API_KEY to backend/.env.")
    allowed = {"1min","5min","15min","30min","1h","4h","1day"}
    if interval not in allowed:
        raise HTTPException(400, "Unsupported interval.")
    try:
        r = requests.get("https://api.twelvedata.com/time_series", params={
            "symbol": symbol, "interval": interval, "outputsize": min(max(outputsize, 30), 500),
            "apikey": key, "format": "JSON"
        }, timeout=20)
        data = r.json()
        if "values" not in data:
            raise HTTPException(502, data.get("message", "Market data provider returned no candle data."))
        df = pd.DataFrame(data["values"])
        df = df.rename(columns={"datetime":"datetime"})
        for c in ["open","high","low","close","volume"]:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.iloc[::-1].reset_index(drop=True)
        result = analyse_candles(df)
        result["symbol"] = symbol
        result["interval"] = interval
        result["source"] = "Twelve Data"
        result["latest_candle_time"] = str(data["values"][0].get("datetime"))
        result["provider_note"] = "Provider coverage, delays and quotas depend on your subscription."
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(502, f"Market data request failed: {e}")

@app.post("/api/multi-timeframe")
async def multi_timeframe(files: list[UploadFile] = File(...)):
    results = {}
    for f in files:
        try:
            df = pd.read_csv(io.BytesIO(await f.read()))
            results[f.filename] = analyse_candles(df)
        except Exception as e:
            results[f.filename] = {"error": str(e)}
    valid = [v for v in results.values() if "bias" in v]
    bulls = sum(v["bias"] == "BULLISH" for v in valid)
    bears = sum(v["bias"] == "BEARISH" for v in valid)
    consensus = "BULLISH" if bulls > bears and bulls >= 2 else "BEARISH" if bears > bulls and bears >= 2 else "MIXED / INSUFFICIENT"
    return {"timeframes": results, "consensus": consensus, "note":"Upload one chronological OHLC CSV per timeframe. This is a simple consensus, not a validated forecast."}
