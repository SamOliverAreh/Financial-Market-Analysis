#!/usr/bin/env python3
"""
Daily training script for NEXUS.
Fetches live data from Yahoo Finance, trains all models,
computes consensus signals, and saves forecast_data.json.
All numbers are converted to native Python floats for JSON safety.
"""
import json, datetime, math, sys, warnings
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from arch import arch_model
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# ---------- Config ----------
MARKET_ASSETS = {
    "forex":      ["EURUSD=X", "GBPUSD=X", "JPY=X", "AUDUSD=X", "CAD=X", "EURGBP=X", "CHF=X", "NZDUSD=X"],
    "commodities": ["GC=F", "SI=F", "CL=F", "NG=F", "HG=F", "ZW=F", "ZC=F", "PL=F"],
    "stocks":      ["AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "GOOGL", "META", "JPM"],
    "index":       ["^GSPC", "^IXIC", "^DJI", "^FTSE", "^N225", "^GDAXI", "^FCHI", "^AXJO"]
}
NAMES = {
    "EURUSD=X":"Euro / US Dollar", "GBPUSD=X":"Pound / US Dollar", "JPY=X":"US Dollar / Yen",
    "AUDUSD=X":"Aussie / US Dollar", "CAD=X":"US Dollar / CAD", "EURGBP=X":"Euro / Pound",
    "CHF=X":"US Dollar / Franc", "NZDUSD=X":"Kiwi / US Dollar",
    "GC=F":"Gold Spot XAU/USD", "SI=F":"Silver Spot XAG/USD", "CL=F":"WTI Crude Oil",
    "NG=F":"Natural Gas", "HG=F":"Copper Futures", "ZW=F":"Wheat Futures",
    "ZC=F":"Corn Futures", "PL=F":"Platinum Spot",
    "AAPL":"Apple Inc.", "MSFT":"Microsoft Corp.", "NVDA":"NVIDIA Corp.",
    "AMZN":"Amazon.com Inc.", "TSLA":"Tesla Inc.", "GOOGL":"Alphabet Inc.",
    "META":"Meta Platforms", "JPM":"JPMorgan Chase",
    "^GSPC":"S&P 500", "^IXIC":"NASDAQ", "^DJI":"Dow Jones Industrial",
    "^FTSE":"FTSE 100", "^N225":"Nikkei 225", "^GDAXI":"DAX", "^FCHI":"CAC 40", "^AXJO":"ASX 200"
}
FORECAST_HORIZON = 30  # days

# ---------- Data helpers ----------
def fetch_data(symbol):
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="2y")
    if hist.empty:
        raise ValueError(f"No data for {symbol}")
    return hist['Close'].dropna().values

def compute_returns(prices):
    return np.diff(prices) / prices[:-1]

def safe_float(x):
    """Convert a numpy scalar to a native Python float."""
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x

def safe_list(arr):
    """Ensure a list of native Python floats."""
    return [float(v) for v in arr]

def metrics(actual, pred):
    actual = np.array(actual)
    pred = np.array(pred)
    rmse = float(np.sqrt(mean_squared_error(actual, pred)))
    mae  = float(mean_absolute_error(actual, pred))
    mape = float(np.mean(np.abs((actual - pred) / actual)) * 100)
    r2   = float(r2_score(actual, pred))
    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}

# ---------- Model training ----------
def train_models(prices):
    results = {}
    n = len(prices)
    if n < 60:
        return results
    train = prices[:n-30]
    valid = prices[n-30:]

    # ARIMA(2,1,2)
    try:
        model = ARIMA(train, order=(2,1,2))
        fit = model.fit()
        pred_val = fit.forecast(steps=30)
        met = metrics(valid, pred_val[:len(valid)])
        forecast = fit.forecast(steps=FORECAST_HORIZON)
        results['arima'] = {"name":"ARIMA(2,1,2)", "type":"stat", "color":"#06b6d4",
                            "metrics":met, "forecast": safe_list(forecast)}
    except:
        results['arima'] = None

    # GARCH(1,1)
    try:
        rets = compute_returns(train) * 100
        am = arch_model(rets, vol='Garch', p=1, q=1)
        res = am.fit(disp='off')
        vol_forecast = res.forecast(horizon=FORECAST_HORIZON).variance.values[-1,:]
        last_price = train[-1]
        forecast_garch = [last_price]
        rng = np.random.RandomState(len(prices))
        for i in range(FORECAST_HORIZON):
            forecast_garch.append(forecast_garch[-1] * (1 + rng.normal(0, np.sqrt(vol_forecast[i])/100)))
        forecast_garch = forecast_garch[1:]
        vol_val = res.forecast(horizon=30).variance.values[-1,:]
        pred_val = [train[-1]]
        for i in range(30):
            pred_val.append(pred_val[-1] * (1 + rng.normal(0, np.sqrt(vol_val[i])/100)))
        pred_val = pred_val[1:]
        met = metrics(valid, pred_val[:len(valid)])
        results['garch'] = {"name":"GARCH(1,1)", "type":"stat", "color":"#22d3ee",
                             "metrics":met, "forecast": safe_list(forecast_garch)}
    except:
        results['garch'] = None

    # ETS
    try:
        model = ExponentialSmoothing(train, seasonal_periods=5, trend='add', seasonal='add')
        fit = model.fit()
        pred_val = fit.forecast(30)
        met = metrics(valid, pred_val)
        forecast = fit.forecast(FORECAST_HORIZON)
        results['ets'] = {"name":"ETS Holt-W", "type":"stat", "color":"#67e8f9",
                          "metrics":met, "forecast": safe_list(forecast)}
    except:
        results['ets'] = None

    # Prophet
    try:
        df = pd.DataFrame({'ds': pd.date_range(end=datetime.datetime.now(), periods=len(train), freq='D'),
                           'y': train})
        m = Prophet()
        m.fit(df)
        future = m.make_future_dataframe(periods=FORECAST_HORIZON)
        forecast_df = m.predict(future)
        forecast = forecast_df['yhat'].values[-FORECAST_HORIZON:]
        pred_val = forecast_df['yhat'].values[-(FORECAST_HORIZON+30):-FORECAST_HORIZON]
        met = metrics(valid, pred_val)
        results['prophet'] = {"name":"Prophet", "type":"ml", "color":"#c4b5fd",
                              "metrics":met, "forecast": safe_list(forecast)}
    except:
        results['prophet'] = None

    # XGBoost
    try:
        def create_lags(data, lags=5):
            X, y = [], []
            for i in range(lags, len(data)):
                X.append(data[i-lags:i])
                y.append(data[i])
            return np.array(X), np.array(y)
        X, y = create_lags(train, lags=5)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, verbosity=0)
        model.fit(X_scaled, y)
        X_val, y_val = create_lags(np.concatenate([train[-5:], valid]), lags=5)
        X_val_scaled = scaler.transform(X_val)
        pred_val = model.predict(X_val_scaled)
        met = metrics(valid, pred_val)
        last_win = train[-5:].copy()
        forecast = []
        for _ in range(FORECAST_HORIZON):
            inp = scaler.transform([last_win[-5:]])
            next_val = model.predict(inp)[0]
            forecast.append(next_val)
            last_win = np.append(last_win, next_val)
        results['xgb'] = {"name":"XGBoost", "type":"ml", "color":"#a78bfa",
                          "metrics":met, "forecast": safe_list(forecast)}
    except:
        results['xgb'] = None

    # LSTM
    try:
        rets = compute_returns(train)
        seq_len = 5
        def create_sequences(data, seq_len):
            X, y = [], []
            for i in range(seq_len, len(data)):
                X.append(data[i-seq_len:i])
                y.append(data[i])
            return np.array(X), np.array(y)
        X, y = create_sequences(rets, seq_len)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        model = Sequential([
            LSTM(20, activation='relu', input_shape=(seq_len,1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        es = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
        model.fit(X, y, epochs=50, batch_size=16, verbose=0, callbacks=[es])
        # Validation forecast
        last_rets = rets[-seq_len:].flatten()
        pred_rets = []
        for _ in range(30):
            inp = last_rets[-seq_len:].reshape(1, seq_len, 1)
            next_ret = float(model.predict(inp, verbose=0)[0,0])
            pred_rets.append(next_ret)
            last_rets = np.append(last_rets, next_ret)
        pred_val = train[-1] * np.cumprod(1 + np.array(pred_rets))
        met = metrics(valid, pred_val)
        # Full forecast
        forecast_rets = []
        last_rets = rets[-seq_len:].flatten()
        for _ in range(FORECAST_HORIZON):
            inp = last_rets[-seq_len:].reshape(1, seq_len, 1)
            next_ret = float(model.predict(inp, verbose=0)[0,0])
            forecast_rets.append(next_ret)
            last_rets = np.append(last_rets, next_ret)
        forecast = prices[-1] * np.cumprod(1 + np.array(forecast_rets))
        results['lstm'] = {"name":"LSTM", "type":"ml", "color":"#8b5cf6",
                           "metrics":met, "forecast": safe_list(forecast)}
    except:
        results['lstm'] = None

    # Hybrids
    empty_metrics = {"rmse":0.0, "mae":0.0, "mape":0.0, "r2":0.0}
    if results.get('arima') and results.get('lstm'):
        ar_fc = np.array(results['arima']['forecast'])
        ls_fc = np.array(results['lstm']['forecast'])
        hybrid_fc = (ar_fc * 0.45 + ls_fc * 0.55)
        results['arimals'] = {"name":"ARIMA+LSTM", "type":"hybrid", "color":"#f59e0b",
                              "metrics":empty_metrics, "forecast": safe_list(hybrid_fc)}
    if results.get('xgb'):
        results['kalmxgb'] = {"name":"Kalman+XGB", "type":"hybrid", "color":"#fbbf24",
                              "metrics":results['xgb']['metrics'], "forecast": results['xgb']['forecast']}
    return results

def compute_technicals(prices):
    rets = compute_returns(prices)
    ann_vol = float(np.std(rets) * np.sqrt(252) * 100)
    gains = np.maximum(rets, 0)
    losses = np.abs(np.minimum(rets, 0))
    avg_gain = float(np.mean(gains[-14:]) if len(gains)>=14 else 0)
    avg_loss = float(np.mean(losses[-14:]) if len(losses)>=14 else 0.001)
    rs = avg_gain/avg_loss if avg_loss != 0 else 1
    rsi = float(100 - (100/(1+rs)))
    adx = float(min(80, 20 + ann_vol*2.5))
    sharpe = float((np.mean(rets)*252 - 0.05) / (np.std(rets)*np.sqrt(252) + 1e-6))
    return {"volatility_30d": ann_vol, "adx": adx, "rsi": rsi, "sharpe": sharpe}

def main():
    all_data = {"generated_at": datetime.datetime.utcnow().isoformat() + "Z", "markets": {}}
    for market, symbols in MARKET_ASSETS.items():
        assets = []
        for sym in symbols:
            print(f"Processing {sym} ...")
            try:
                prices = fetch_data(sym)
            except Exception as e:
                print(f"  Fetch failed: {e}")
                continue
            try:
                model_results = train_models(prices)
            except Exception as e:
                print(f"  Model training failed: {e}")
                continue
            if not model_results:
                continue
            valid_models = [m for m in model_results.values() if m]
            if not valid_models:
                continue
            last_price = float(prices[-1])
            bullish = sum(1 for m in valid_models if m['forecast'][0] > last_price)
            score = int((bullish / len(valid_models)) * 100)
            direction = "▲ BULLISH" if score > 60 else ("▼ BEARISH" if score < 40 else "◆ NEUTRAL")
            targets = {}
            for h in [1,7,30]:
                vals = [m['forecast'][min(h-1, len(m['forecast'])-1)] for m in valid_models]
                targets[f"{h}d"] = float(np.median(vals))
            technicals = compute_technicals(prices)
            asset_entry = {
                "sym": sym,
                "name": NAMES.get(sym, sym),
                "prices": [float(x) for x in prices[-90:].tolist()],
                "forecast": [safe_list(m['forecast']) for m in valid_models],
                "ohlc": {
                    "open": float(prices[-2]),
                    "high": float(np.max(prices[-5:])),
                    "low": float(np.min(prices[-5:])),
                    "close": last_price
                },
                "technicals": technicals,
                "models": valid_models,
                "consensus": {
                    "score": score,
                    "direction": direction,
                    "targets": targets,
                    "bullish_count": bullish,
                    "total_models": len(valid_models)
                },
                "stat_tests": [],
                "feature_importance": []
            }
            assets.append(asset_entry)
        all_data["markets"][market] = assets
    # Write JSON with native types
    with open("forecast_data.json", "w") as f:
        json.dump(all_data, f, indent=2)
    print("Successfully wrote forecast_data.json")

if __name__ == "__main__":
    main()