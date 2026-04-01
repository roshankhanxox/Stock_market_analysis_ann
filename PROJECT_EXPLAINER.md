# Stock Market Trend Prediction — ANN Project Explainer

---

## 1. What is OHLCV Data?

OHLCV stands for **Open, High, Low, Close, Volume** — the five core data points recorded for every trading day.

| Field | Meaning | Example (AAPL) |
|---|---|---|
| **Open** | Price at market open (9:30 AM) | 250.74 |
| **High** | Highest price reached during the day | 255.52 |
| **Low** | Lowest price reached during the day | 249.40 |
| **Close** | Price at market close (4:00 PM) | 254.81 |
| **Volume** | Number of shares traded | 41,994,100 |

### Why OHLCV?
- **Open vs Close**: Tells you if the day was bullish (Close > Open) or bearish (Close < Open)
- **High - Low (Range)**: Measures daily volatility — a wide range means uncertainty or big moves
- **Volume**: Confirms price moves. A price rise on high volume = strong conviction. A rise on low volume = weak signal
- **Close**: The most important value — it's the final agreed price of the day and what most indicators are built on

---

## 2. Feature Engineering — What We Derived and Why

Raw OHLCV alone is not enough. We engineer 6 additional features to give the model more signal.

### 2.1 Return (Daily % Change)
```
Return = (Close_today - Close_yesterday) / Close_yesterday
```
- Measures how much the stock moved in percentage terms
- Normalises across stocks (a $1 move means different things for a $10 stock vs a $500 stock)
- Positive return = up day, negative = down day

### 2.2 MA5 — 5-Day Moving Average
```
MA5 = Average of Close over last 5 trading days
```
- Smooths out daily noise to show the short-term trend
- If today's Close > MA5, the stock is trending upward in the short term
- 5 days ≈ 1 trading week

### 2.3 MA10 — 10-Day Moving Average
```
MA10 = Average of Close over last 10 trading days
```
- Represents the medium-term trend (2 trading weeks)
- Used alongside MA5 to detect crossovers

### 2.4 Diff (Intraday Movement)
```
Diff = Close - Open
```
- Positive: buyers dominated the day (bullish candle)
- Negative: sellers dominated the day (bearish candle)
- Directly captures what happened during that trading session

### 2.5 Range (Daily Volatility)
```
Range = High - Low
```
- Measures how wide the price swung during the day
- High range = high uncertainty or news-driven trading
- Low range = calm market, consolidation

### 2.6 Trend (Momentum Signal)
```
Trend = MA5 - MA10
```
- The most powerful derived feature
- **Positive Trend**: Short-term average above medium-term → upward momentum (bullish)
- **Negative Trend**: Short-term average below medium-term → downward momentum (bearish)
- This is the core of the classic "Golden Cross / Death Cross" trading strategy

### Final Feature Set (11 features fed to the model)
```
Open, High, Low, Close, Volume, Return, MA5, MA10, Diff, Range, Trend
```

---

## 3. Preprocessing Pipeline

### Step 1 — Load Raw CSV
Each stock has its own CSV file with historical daily OHLCV data (e.g. AAPL goes back to 1980).

### Step 2 — Feature Engineering
`create_features()` computes the 6 derived columns above. The first 10 rows are dropped because MA10 requires 10 previous rows to compute — those rows become NaN and are removed.

### Step 3 — Target Label Creation (done during training in the notebook)
```
Target = 1  if tomorrow's Close > today's Close  (UP)
Target = 0  if tomorrow's Close <= today's Close (DOWN)
```
This is a binary classification problem.

### Step 4 — Scaling
Each stock has its own `MinMaxScaler` fitted on training data:
```
Scaled value = (value - min) / (max - min)  →  range [0, 1]
```
**Why scale per stock?** AAPL trades at ~$250, NVDA at ~$800. A single global scaler would distort the relative magnitudes. Each stock's scaler preserves its own dynamics.

### Step 5 — Train/Test Split
Standard 80/20 time-based split (no shuffling — future data must never leak into training).

---

## 4. The ANN Model Architecture

```
Input Layer           →  11 features
Dense(64, ReLU)       →  64 neurons, learns complex patterns
BatchNormalization    →  normalises activations, stabilises training
Dropout(0.2)          →  randomly drops 20% of neurons to prevent overfitting
Dense(32, ReLU)       →  32 neurons, extracts higher-level patterns
BatchNormalization    →  normalises again
Dense(16, ReLU)       →  16 neurons, further abstraction
Dense(1, Sigmoid)     →  outputs a probability between 0 and 1
```

### Why each component?

**Dense layers**: The core computational units. Each neuron learns a weighted combination of its inputs. Multiple layers let the model learn non-linear relationships that a simple rule couldn't capture.

**ReLU activation**: `max(0, x)` — passes positive values unchanged, kills negative values. Prevents the vanishing gradient problem and trains faster than sigmoid/tanh in hidden layers.

**BatchNormalization**: Normalises the output of each layer to have mean=0, std=1. This makes training more stable and allows higher learning rates.

**Dropout(0.2)**: During training, randomly sets 20% of neurons to zero each pass. Forces the network to not rely on any single neuron — acts as regularisation and reduces overfitting.

**Sigmoid output**: `1 / (1 + e^(-x))` — squashes output to (0, 1), interpretable as a probability. Values > 0.52 → predict UP, values ≤ 0.52 → predict DOWN.

### Training Configuration
- **Optimizer**: Adam (adaptive learning rate — efficient for financial data)
- **Loss**: Binary Crossentropy (standard for binary classification)
- **Metric**: Accuracy

---

## 5. Prediction Flow (Step by Step)

When you enter values in the app and click Predict:

```
1. User inputs: Open, High, Low, Close, Volume (1 new row)

2. App takes the last 20 rows of historical data for that stock
   + appends the new row = 21 rows total

3. create_features() runs on these 21 rows:
   - Computes MA5, MA10 using rolling windows
   - Computes Return, Diff, Range, Trend
   - Drops NaN rows from rolling windows

4. Takes only the LAST row (the new day you entered)
   - Now has all 11 features computed

5. scaler.transform() scales the 11 features to [0, 1]
   using the same scaler fitted during training

6. model.predict() runs the scaled input through the ANN
   - Returns a sigmoid probability between 0 and 1

7. Small random jitter added (±0.05) to simulate real-world noise

8. Decision threshold:
   - prob > 0.52 → UP  📈
   - prob ≤ 0.52 → DOWN 📉
```

**Why append to 20 historical rows?** The MA5 and MA10 features need prior days to compute. You can't compute a 5-day average from 1 data point.

---

## 6. Model Evaluation Metrics

### Accuracy
```
Accuracy = Correct Predictions / Total Predictions
```
| Stock | Accuracy |
|---|---|
| AAPL | 54.8% |
| ORCL | 55.5% |
| NFLX | 54.3% |
| MSFT | 53.8% |
| IBM | 52.8% |
| TSLA | 52.9% |

### Confusion Matrix
A 2×2 matrix showing:
```
                Predicted UP    Predicted DOWN
Actual UP   |   True Positive  |  False Negative |
Actual DOWN |   False Positive |  True Negative  |
```
- **True Positive**: Correctly predicted UP
- **True Negative**: Correctly predicted DOWN
- **False Positive**: Predicted UP but it went DOWN (costly in trading)
- **False Negative**: Predicted DOWN but it went UP (missed opportunity)

### Why 51–55% is Actually Meaningful
- A random coin flip gives 50% accuracy
- The **Efficient Market Hypothesis** states all public information is already priced in
- Achieving consistent 52–55% is non-trivial
- In live trading, even 53% accuracy with proper risk management can be profitable when compounded across hundreds of trades
- The real benchmark is not 100% — it's beating 50%

---

## 7. Stocks Covered and Why They Were Chosen

| Stock | Company | Sector | Why Interesting |
|---|---|---|---|
| AAPL | Apple | Technology | Most traded, high liquidity |
| TSLA | Tesla | EV/Energy | High volatility, sentiment-driven |
| MSFT | Microsoft | Technology | Stable large-cap |
| GOOGL | Alphabet | Technology | Ad revenue cycles |
| AMZN | Amazon | E-commerce/Cloud | Macro-sensitive |
| NVDA | NVIDIA | Semiconductors | AI boom stock, extreme swings |
| NFLX | Netflix | Streaming | Subscriber-driven volatility |
| IBM | IBM | Enterprise Tech | Slow-moving, stable |
| ORCL | Oracle | Cloud/Database | Steady enterprise stock |
| JPM | JPMorgan | Finance | Interest rate sensitive |

---

## 8. Limitations of the Model

1. **No news/sentiment**: The model only sees price and volume. A tweet or earnings report can move a stock 20% in minutes — this model cannot predict that.

2. **Stationarity assumption**: Financial time series are non-stationary (their statistical properties change over time). A model trained on 2018 data may not generalise to a 2024 market regime.

3. **No sequence memory**: The ANN treats each day as independent. It doesn't remember that the last 5 days were all down. An LSTM would handle this better.

4. **Look-ahead bias risk**: If the train/test split is not strictly time-ordered, the model could accidentally "see" future data during training — inflating accuracy.

5. **Single-step prediction**: Only predicts the next day's direction, not magnitude. Knowing a stock will go "up" but only by 0.01% isn't useful.

6. **Random jitter in production**: The app adds ±0.05 random noise to predictions, which is not appropriate for real use — it was added to make demos look more variable.

---

## 9. How to Improve Accuracy (Extension Ideas)

1. **More features**: RSI (momentum oscillator), MACD (trend indicator), Bollinger Bands (volatility bands), Volume Moving Average
2. **LSTM model**: Feed sequences of 10–30 days instead of a single snapshot — captures temporal patterns
3. **XGBoost**: Tree-based models often outperform ANNs on tabular financial data
4. **Better target**: Filter out near-zero moves (e.g. only label if move > 0.5%) to remove noise from the training signal
5. **Ensemble**: Combine ANN + XGBoost + LSTM predictions by majority vote

---

## 10. Key Terms Glossary

| Term | Definition |
|---|---|
| ANN | Artificial Neural Network — a model of interconnected neurons that learns patterns from data |
| Binary Classification | Predicting one of two outcomes — here: UP or DOWN |
| Sigmoid | Activation function that outputs a value between 0 and 1 (probability) |
| BatchNorm | Normalises layer outputs to stabilise and speed up training |
| Dropout | Regularisation technique that randomly disables neurons during training |
| Scaler | Transforms feature values to a standard range (e.g. 0 to 1) |
| Moving Average | Average of the last N values — smooths out noise to show trend |
| Overfitting | Model performs well on training data but fails on new data |
| Confusion Matrix | Table showing correct and incorrect predictions broken down by class |
| Streamlit | Python library for building interactive web apps for ML projects |
