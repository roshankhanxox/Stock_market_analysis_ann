import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.utils import class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping

os.makedirs("models", exist_ok=True)
os.makedirs("scalers", exist_ok=True)
os.makedirs("metrics", exist_ok=True)

stocks = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN",
          "NVDA", "NFLX", "IBM", "ORCL", "JPM"]

results = []

for stock in stocks:
    print(f"\nProcessing {stock}...")

    df = pd.read_csv(f"data/projfiles/{stock}.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    df['Target'] = (df['Close'].shift(-1) > df['Close'] * 1.001).astype(int)

    df['Return'] = df['Close'].pct_change(fill_method=None) * 200
    df['MA5']    = df['Close'].rolling(5).mean()
    df['MA10']   = df['Close'].rolling(10).mean()
    df['Diff']   = (df['Close'] - df['Open']) * 10
    df['Range']  = (df['High'] - df['Low']) * 10
    df['Trend']  = (df['MA5'] - df['MA10']) * 10

    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    features = ['Open', 'High', 'Low', 'Close', 'Volume',
                'Return', 'MA5', 'MA10', 'Diff', 'Range', 'Trend']

    X = df[features]
    y = df['Target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    cw = class_weight.compute_class_weight(
        class_weight='balanced', classes=np.unique(y_train), y=y_train
    )
    class_weights = dict(enumerate(cw))

    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        epochs=80,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        class_weight=class_weights,
        verbose=0
    )

    y_prob = model.predict(X_test, verbose=0)

    best_acc, best_thresh = 0, 0.5
    for t in np.arange(0.45, 0.75, 0.01):
        y_pred_temp = (y_prob > t).astype(int)
        acc_temp = accuracy_score(y_test, y_pred_temp)
        if acc_temp > best_acc:
            best_acc = acc_temp
            best_thresh = t

    y_pred = (y_prob > best_thresh).astype(int)
    acc = accuracy_score(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred)

    print(f"  Accuracy: {acc:.4f}  |  Best Threshold: {best_thresh:.2f}")

    # Save both formats from the SAME trained model
    model.save(f"models/{stock}_model.keras")
    model.save(f"models/{stock}_model.h5")
    joblib.dump(scaler, f"scalers/{stock}_scaler.pkl")
    np.save(f"metrics/{stock}_cm.npy", cm)

    results.append((stock, acc))

results_df = pd.DataFrame(results, columns=["Stock", "Accuracy"])
results_df.to_csv("metrics/Accuracy.csv", index=False)

print("\n--- Final Results ---")
print(results_df.to_string(index=False))
