import pandas as pd
import numpy as np

def _clean(df):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "volume" not in df.columns:
        df["volume"] = 0
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    df = df.dropna(subset=required).reset_index(drop=True)
    if len(df) < 20:
        raise ValueError("Please provide at least 20 valid candles.")
    return df

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(close, n=14):
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100/(1+rs)).fillna(50)

def analyse_candles(raw):
    df = _clean(raw)
    o, h, l, c = (df[x] for x in ["open", "high", "low", "close"])
    body = (c-o).abs()
    rng = (h-l).replace(0, np.nan)
    upper = h - pd.concat([o,c], axis=1).max(axis=1)
    lower = pd.concat([o,c], axis=1).min(axis=1) - l

    # Indicators
    e20, e50, e200 = ema(c,20), ema(c,50), ema(c, min(200, max(20, len(c)-1)))
    r = rsi(c)
    macd_line = ema(c,12) - ema(c,26)
    macd_signal = ema(macd_line,9)
    tr = pd.concat([(h-l), (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    # Candlestick patterns: use completed candles only.
    patterns = []
    i = len(df)-1
    prev = i-1
    body_i = float(body.iloc[i])
    range_i = float(rng.iloc[i]) if pd.notna(rng.iloc[i]) else 0
    if range_i > 0:
        if body_i/range_i <= 0.12:
            patterns.append({"name":"Doji", "bias":"neutral", "detail":"Small real body relative to total range."})
        if lower.iloc[i] >= max(body_i*2.0, range_i*0.45) and upper.iloc[i] <= range_i*0.20 and c.iloc[i] > o.iloc[i]:
            patterns.append({"name":"Bullish pin bar / hammer", "bias":"bullish", "detail":"Long lower wick with a relatively small upper wick."})
        if upper.iloc[i] >= max(body_i*2.0, range_i*0.45) and lower.iloc[i] <= range_i*0.20 and c.iloc[i] < o.iloc[i]:
            patterns.append({"name":"Bearish pin bar / shooting star", "bias":"bearish", "detail":"Long upper wick with a relatively small lower wick."})

    # Engulfing
    if c.iloc[prev] < o.iloc[prev] and c.iloc[i] > o.iloc[i] and o.iloc[i] <= c.iloc[prev] and c.iloc[i] >= o.iloc[prev]:
        patterns.append({"name":"Bullish engulfing", "bias":"bullish", "detail":"Current bullish body engulfs prior bearish body."})
    if c.iloc[prev] > o.iloc[prev] and c.iloc[i] < o.iloc[i] and o.iloc[i] >= c.iloc[prev] and c.iloc[i] <= o.iloc[prev]:
        patterns.append({"name":"Bearish engulfing", "bias":"bearish", "detail":"Current bearish body engulfs prior bullish body."})

    # Three-candle stars (simple heuristic)
    if len(df) >= 3:
        a,b,z = i-2,i-1,i
        small_mid = body.iloc[b] <= max(body.iloc[a]*0.55, (h.iloc[b]-l.iloc[b])*0.25)
        if c.iloc[a] < o.iloc[a] and small_mid and c.iloc[z] > o.iloc[z] and c.iloc[z] >= (o.iloc[a]+c.iloc[a])/2:
            patterns.append({"name":"Morning star (heuristic)", "bias":"bullish", "detail":"Three-candle reversal shape; context and gaps vary by market."})
        if c.iloc[a] > o.iloc[a] and small_mid and c.iloc[z] < o.iloc[z] and c.iloc[z] <= (o.iloc[a]+c.iloc[a])/2:
            patterns.append({"name":"Evening star (heuristic)", "bias":"bearish", "detail":"Three-candle reversal shape; context and gaps vary by market."})

    # Local swing points using a 2-left / 2-right fractal window
    highs, lows = [], []
    for j in range(2, len(df)-2):
        if h.iloc[j] > max(h.iloc[j-2:j].max(), h.iloc[j+1:j+3].max()):
            highs.append((j, float(h.iloc[j])))
        if l.iloc[j] < min(l.iloc[j-2:j].min(), l.iloc[j+1:j+3].min()):
            lows.append((j, float(l.iloc[j])))
    last_highs = [x[1] for x in highs[-2:]]
    last_lows = [x[1] for x in lows[-2:]]
    hh = len(last_highs)==2 and last_highs[-1] > last_highs[-2]
    lh = len(last_highs)==2 and last_highs[-1] < last_highs[-2]
    hl = len(last_lows)==2 and last_lows[-1] > last_lows[-2]
    ll = len(last_lows)==2 and last_lows[-1] < last_lows[-2]
    if hh and hl: structure = "Higher highs + higher lows"
    elif lh and ll: structure = "Lower highs + lower lows"
    else: structure = "Mixed / range / structure not confirmed"

    last_close = float(c.iloc[-1])
    bullish_points = 0
    bearish_points = 0
    evidence = []
    if e20.iloc[-1] > e50.iloc[-1]:
        bullish_points += 1; evidence.append("EMA 20 is above EMA 50")
    else:
        bearish_points += 1; evidence.append("EMA 20 is below EMA 50")
    if last_close > e20.iloc[-1]:
        bullish_points += 1; evidence.append("Price is above EMA 20")
    else:
        bearish_points += 1; evidence.append("Price is below EMA 20")
    if r.iloc[-1] >= 55:
        bullish_points += 1; evidence.append(f"RSI is {r.iloc[-1]:.1f} (positive momentum)")
    elif r.iloc[-1] <= 45:
        bearish_points += 1; evidence.append(f"RSI is {r.iloc[-1]:.1f} (negative momentum)")
    else:
        evidence.append(f"RSI is {r.iloc[-1]:.1f} (middle zone)")
    if macd_line.iloc[-1] > macd_signal.iloc[-1]:
        bullish_points += 1; evidence.append("MACD line is above signal line")
    else:
        bearish_points += 1; evidence.append("MACD line is below signal line")
    if structure.startswith("Higher"):
        bullish_points += 2
    elif structure.startswith("Lower"):
        bearish_points += 2
    for p in patterns:
        if p["bias"] == "bullish": bullish_points += 1
        elif p["bias"] == "bearish": bearish_points += 1

    if bullish_points >= bearish_points + 2: bias = "BULLISH"
    elif bearish_points >= bullish_points + 2: bias = "BEARISH"
    else: bias = "NEUTRAL / WAIT"

    return {
        "candles": int(len(df)),
        "last_close": last_close,
        "bias": bias,
        "bullish_score": bullish_points,
        "bearish_score": bearish_points,
        "structure": structure,
        "swings": {
            "last_two_swing_highs": last_highs,
            "last_two_swing_lows": last_lows,
            "higher_highs": bool(hh), "higher_lows": bool(hl),
            "lower_highs": bool(lh), "lower_lows": bool(ll)
        },
        "patterns": patterns,
        "indicators": {
            "ema20": round(float(e20.iloc[-1]), 6),
            "ema50": round(float(e50.iloc[-1]), 6),
            "ema200_proxy": round(float(e200.iloc[-1]), 6),
            "rsi14": round(float(r.iloc[-1]), 2),
            "macd": round(float(macd_line.iloc[-1]), 6),
            "macd_signal": round(float(macd_signal.iloc[-1]), 6),
            "atr14": None if pd.isna(atr.iloc[-1]) else round(float(atr.iloc[-1]), 6)
        },
        "evidence": evidence,
        "disclaimer": "Heuristic educational analysis, not a prediction or guaranteed trading signal."
    }
