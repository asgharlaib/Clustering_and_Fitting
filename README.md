# Time-Series Forecasting: Johnson & Johnson Earnings Analysis

An end-to-end econometric analysis comparing traditional **ARIMA** and seasonal **SARIMAX** architectures to model and forecast the quarterly Earnings Per Share (EPS) of Johnson & Johnson.

## 📊 Executive Summary
* **Objective:** Build a robust forecasting engine to model historical quarterly earnings data and project future performance.
* **Key Finding:** While the data exhibits clear quarterly seasonality ($s=4$), a heavily optimized, high-order **ARIMA(4, 1, 6)** model operating on the trend-stabilized series empirically outperformed the SARIMAX architecture across all primary evaluation metrics.
* **Core Result:** The optimized **ARIMA(4, 1, 6)** model achieved an outstanding **MAPE of 8.77%** and a **Root Mean Squared Error (RMSE) of 0.40**, proving highly accurate at capturing the underlying patterns.

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Time-Series Analysis:** `statsmodels.tsa.arima.model.ARIMA`, `statsmodels.tsa.statespace.sarimax.SARIMAX`
* **Statistical Testing:** Augmented Dickey-Fuller (`adfuller`)
* **Data Processing & Visualization:** Pandas, NumPy, Matplotlib

---

## 📈 Methodology & Workflow

### 1. Exploratory Data Analysis & Stationarization
* **Initial Assessment:** The raw time series showed a clear exponential upward trend and non-constant variance. An initial Augmented Dickey-Fuller (ADF) test yielded a **$p$-value of 1.0**, confirming non-stationarity.
* **Transformation:** Applied a log-transformation to stabilize variance, followed by first-order differencing (`.diff()`) to eliminate the trend.
* **Validation:** The ADF test on the transformed data yielded a **$p$-value of 0.0004**, confirming stationarity and readiness for modeling.

### 2. Model Optimization Grid Search
Two distinct optimization workflows were executed to locate the lowest Akaike Information Criterion (AIC):
1. **ARIMA Grid Search:** Evaluated parameters $(p, 1, q)$ ranging from 0 to 7. The optimal model selected was **ARIMA(4, 1, 6)** with an $AIC$ of **115.85**.
2. **SARIMAX Grid Search:** Evaluated parameters over a seasonal grid $(p, d, q) \times (P, D, Q)_4$. The optimal configuration was `SARIMAX(1, 1, 1)x(0, 1, 0, 4)` with an $AIC$ of **23.35**.

---

## ⚖️ Comparative Evaluation

Rather than blindly deploying the seasonal model, both configurations were rigorously backtested. **ARIMA(4, 1, 6)** emerged as the statistically superior model for this dataset:

| Model Configuration | MAPE | RMSE | MAE |
| :--- | :--- | :--- | :--- |
| **ARIMA(4, 1, 6)** *(Winner)* | **8.77%** | **0.403** | **0.271** |
| **SARIMAX(1, 1, 1)x(1, 1, 1, 4)** | 16.29% | 2.677 | 2.150 |

### Econometric Insight: Why did ARIMA win?
While J&J data has a seasonal cycle, the high lag coefficients in the **ARIMA(4, 1, 6)** model (specifically the strong significance of $ar.L4$ with a $p$-value $< 0.001$) effectively allowed the non-seasonal model to map the 4-quarter recurring cyclical dependencies directly. When combined with the log-transformed variance stabilization, it yielded a much tighter fit to the physical data track than the lower-order SARIMAX configuration.

---

## 🔮 Visualizations & Outputs
The notebook generates production-grade visual assets to back up these conclusions:
* Raw vs. Transformed data tracking.
* ACF and PACF plots identifying lag structure.
* Full residual diagnostics (`plot_diagnostics()`) checking for white noise residuals.
* A 24-month executive out-of-sample projection band complete with a shaded 95% confidence interval.

## 📂 Project Structure
```text
├── ARMA_J&J_Casestudy.ipynb   <- Cleaned, widget-free production notebook
├── jj.csv                     <- Quarterly earnings per share source dataset
└── README.md                  <- Project documentation 
