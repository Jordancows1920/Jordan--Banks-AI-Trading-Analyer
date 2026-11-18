# Jordan Banks — AI Trading Chart Analyzer

An Android Studio project (Kotlin + Jetpack Compose) with a small FastAPI analysis server.

## What works
- Pick a chart screenshot from the Android gallery (MT5 / TradingView screenshots).
- Enter symbol, timeframe and optional notes.
- Upload screenshot to the API.
- Server sends the image to an OpenAI vision model for a structured chart read:
  - visible candlestick patterns (pin bar, engulfing, morning/evening star, etc.)
  - apparent trend / market structure
  - visible support/resistance and trendlines
  - moving-average observations **only when moving averages are visible**
  - bullish / bearish / neutral estimated probabilities and reasoning
- Optional NewsAPI headlines are added as context when `NEWS_API_KEY` is configured.
- Results show risk notes and missing/uncertain information.

## Important limitations
This is an AI-assisted visual interpretation, not a trading system or guaranteed prediction.
A screenshot does not contain the underlying OHLC price series. The model cannot reliably calculate RSI, MACD, moving averages, exact support/resistance, or statistically calibrated probabilities from pixels alone. The app therefore labels probabilities as **model estimates, not statistically calibrated**. For genuine calibration and precise indicators, connect a trusted OHLC data feed and validate the model against out-of-sample historical data before relying on it.

## Run the backend
Requires Python 3.10+.

```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Set OPENAI_API_KEY. Optionally set NEWS_API_KEY.
uvicorn main:app --host 0.0.0.0 --port 8000
```

For Android emulator testing, the app defaults to `http://10.0.2.2:8000/`.
For a physical Android phone, change the API URL in the app to your computer's LAN IP, e.g. `http://192.168.1.10:8000/`, and ensure both devices are on the same Wi-Fi. For production, deploy the backend over HTTPS and use that URL.

## Build Android app
Open `android/` in Android Studio (Giraffe or newer), allow Gradle sync, then Run.
Minimum Android: API 26.

## API security
Do not put your OpenAI or NewsAPI keys in the Android app. Keep them on the backend. The included backend is a development starter; add authentication, rate limiting, HTTPS, logging controls and usage limits before public deployment.
