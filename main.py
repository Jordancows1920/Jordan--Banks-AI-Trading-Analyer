import base64, json, os
from typing import Optional
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

load_dotenv()
app = FastAPI(title="Jordan Banks Trading Analyzer API", version="1.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"]
)

@app.get("/health")
def health():
    return {"status": "ok", "service": "Jordan Banks Trading Analyzer"}

async def get_news(symbol: str):
    key = os.getenv("NEWS_API_KEY")
    if not key:
        return []
    # NewsAPI is a broad headline source; symbol-to-company matching is imperfect for FX.
    query = symbol.replace("/", " ") + " markets finance"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get("https://newsapi.org/v2/everything", params={
                "q": query, "language": "en", "sortBy": "publishedAt",
                "pageSize": 5, "apiKey": key
            })
            r.raise_for_status()
            data = r.json()
            return [{
                "title": a.get("title"), "source": (a.get("source") or {}).get("name"),
                "publishedAt": a.get("publishedAt"), "url": a.get("url"),
                "description": a.get("description")
            } for a in data.get("articles", [])]
    except Exception:
        return []

@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    symbol: str = Form("Unknown"),
    timeframe: str = Form("Unknown"),
    notes: str = Form("")
):
    if image.content_type not in ("image/png", "image/jpeg", "image/webp"):
        raise HTTPException(415, "Please upload a PNG, JPG or WEBP chart screenshot.")
    raw = await image.read()
    if len(raw) > 12 * 1024 * 1024:
        raise HTTPException(413, "Image is too large. Maximum size is 12 MB.")
    if not raw:
        raise HTTPException(400, "The uploaded image is empty.")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(503, "Backend is not configured. Add OPENAI_API_KEY to backend/.env.")
    news = await get_news(symbol)
    mime = image.content_type
    data_url = f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"

    system_prompt = """You are a cautious technical-analysis assistant. Analyze only what is visually supported by the supplied chart screenshot and provided context.
Do not invent prices, indicators, candles, news impacts, or exact levels. Distinguish observed facts from uncertain visual estimates.
A screenshot alone does NOT provide underlying OHLC data, so do not claim to calculate RSI, MACD, ATR, exact moving averages, or statistically calibrated probabilities.
If indicators are not visibly plotted, say they cannot be determined from the image.
Return valid JSON only, with this schema:
{
 "symbol":"string", "timeframe":"string",
 "direction":"Bullish|Bearish|Neutral|Unclear",
 "probabilities":{"bullish":0,"bearish":0,"neutral":0},
 "confidence":"Low|Medium|High",
 "trend_summary":"string",
 "market_structure":"string",
 "candlestick_patterns":[{"name":"string","type":"Bullish|Bearish|Neutral","evidence":"string","reliability":"Low|Medium|High"}],
 "indicators":[{"name":"string","observation":"string","status":"Visible|Not visible|Unclear"}],
 "support_resistance":[{"kind":"Support|Resistance|Possible zone","description":"string"}],
 "trendlines":"string",
 "news_context":"string",
 "bullish_factors":["string"], "bearish_factors":["string"],
 "what_would_invalidate":"string",
 "limitations":["string"],
 "disclaimer":"This is educational analysis, not financial advice."
}
Probabilities are subjective visual-model estimates, not calibrated odds. Make them sum to 100. If chart quality or context is inadequate, assign a high neutral share and low confidence. Do not imply a probability of profit or guarantee."""
    user_prompt = f"""Analyze this trading chart screenshot.
Symbol: {symbol}
Timeframe: {timeframe}
User notes: {notes or 'None'}
Recent news headlines (may be irrelevant; assess relevance cautiously): {json.dumps(news, ensure_ascii=False)}
Look for pin bars, bullish/bearish engulfing, morning/evening stars, doji, inside bars, trend, visible moving averages, support/resistance and chart structure.
Use neutral probability when evidence is mixed or screenshot is insufficient."""
    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}}
                ]}
            ]
        )
        result = json.loads(response.choices[0].message.content)
        result["news_headlines"] = news
        result["probability_note"] = "Subjective AI visual estimates only; not statistically calibrated or a probability of profit."
        return result
    except Exception as e:
        raise HTTPException(502, f"Analysis service error: {str(e)[:300]}")
