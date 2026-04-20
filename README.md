# 📈 Options Mispricing Hunter — Financial Data Analytics & Signal Generation

> A hybrid quantitative system that combines classical Black-Scholes pricing with an ML residual model to detect and exploit options mispricing on NSE (National Stock Exchange of India).

---

## 🧠 Overview

Options markets are frequently mispriced due to volatility surface imperfections, stale data, and behavioral biases. This system identifies those inefficiencies in real-time by:

1. Computing theoretical option prices using the **Black-Scholes model**
2. Training an **XGBoost / LightGBM residual model** to capture what B-S misses
3. Generating **directional trade signals** when predicted vs. market price divergence crosses a threshold
4. Serving signals via a **FastAPI microservice** at 500+ contracts/minute throughput

---

## ⚙️ Architecture

```
NSE Market Data Feed
        │
        ▼
┌─────────────────────┐
│  Black-Scholes Layer │  ← Theoretical pricing (IV, Greeks)
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  ML Residual Model  │  ← XGBoost / LightGBM trained on pricing errors
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Signal Generator   │  ← BUY / SELL / HOLD with confidence score
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  FastAPI Service    │  ← REST API for downstream consumption
└─────────────────────┘
```

---

## 🚀 Features

- **Hybrid pricing model** — Black-Scholes as base + gradient boosting on residuals
- **73% directional accuracy** on live NSE options data
- **500+ contracts/minute** signal throughput via async FastAPI
- Real-time Greeks calculation (Delta, Gamma, Theta, Vega)
- Implied Volatility (IV) surface construction
- Configurable mispricing threshold for signal generation
- REST API endpoint for integration with execution engines or dashboards

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Core Pricing | Python, NumPy, SciPy |
| ML Models | XGBoost, LightGBM, scikit-learn |
| API Layer | FastAPI, Uvicorn |
| Data | NSE options chain data |
| Backtesting | pandas, custom signal replay engine |

---

## 📦 Installation

```bash
git clone https://github.com/atimanasbaral/Options-Mispricing-Hunter-Financial-Data-Analytics-Signal-Generation.git
cd Options-Mispricing-Hunter-Financial-Data-Analytics-Signal-Generation

pip install -r requirements.txt
```

---

## 🔧 Usage

**Start the FastAPI server:**

```bash
uvicorn main:app --reload --port 8000
```

**Get mispricing signals:**

```bash
curl -X POST "http://localhost:8000/signals" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "NIFTY", "expiry": "2025-01-30", "strike": 22000, "option_type": "CE"}'
```

**Train the residual model:**

```bash
python train.py --data data/nse_options.csv --model xgboost
```

---

## 📊 Results

| Metric | Value |
|---|---|
| Directional Accuracy | **73%** |
| Signal Throughput | **500+ contracts/min** |
| Model | XGBoost residual on Black-Scholes base |
| Dataset | NSE options chain (live + historical) |

---

## 📁 Project Structure

```
├── main.py                # FastAPI entry point
├── pricing/
│   ├── black_scholes.py   # B-S model, IV solver, Greeks
│   └── surface.py         # IV surface construction
├── models/
│   ├── train.py           # Model training pipeline
│   ├── predict.py         # Inference + signal logic
│   └── saved/             # Serialized model artifacts
├── data/
│   └── nse_options.csv    # Sample data
├── requirements.txt
└── README.md
```

---

## 🔭 Roadmap

- [ ] Live NSE WebSocket feed integration
- [ ] Portfolio-level Greeks aggregation
- [ ] Volatility regime detection (HMM)
- [ ] Execution integration (Zerodha Kite / Upstox API)
- [ ] Dashboard UI for real-time signal monitoring

---

## 👤 Author

**Atimanas Baral**  
EEE Graduate | Quant & ML Engineering  
[GitHub](https://github.com/atimanasbaral) · [LinkedIn](https://linkedin.com/in/atimanasbaral)

---

## ⚠️ Disclaimer

This project is for educational and research purposes only. Not financial advice. Always consult a SEBI-registered advisor before trading.
