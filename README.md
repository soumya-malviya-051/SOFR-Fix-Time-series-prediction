# 📈 Secured Overnight Financing Rate (SOFR) Forecasting using Advanced ML/DL

## 🌟 Project Overview
This project implements and compares advanced **machine learning** and **deep learning** models to predict the **Secured Overnight Financing Rate (SOFR) Fix**, a critical benchmark interest rate in global financial markets.  
The goal is to develop a highly accurate forecasting system by leveraging **time-series decomposition**, **sophisticated feature engineering**, and **robust model architectures**.

The core methodologies employed are:
- **Long Short-Term Memory (LSTM) Neural Networks** – for capturing long-term dependencies in sequential data.  
- **eXtreme Gradient Boosting (XGBoost)** – for its effectiveness with complex, structured time-series features.

---

## 📊 Data & Feature Engineering

The model utilizes **historical SOFR Fix data**, enriched with a comprehensive set of financial indicators:

- **Target Variable**  
  - SOFR Fix (secured overnight lending rate)

- **Endogenous Features**  
  - Lagged SOFR values  
  - Calendar features (day of week, month, quarter-end)  
  - Seasonal components (yearly, weekly)

- **Exogenous Financial Indicators**  
  - Reverse Repurchase Agreement Amount (**RRP Amt**)  
  - Treasury General Account Balance (**TGA Balance**)  
  - DTCC Treasury (**DTCC Tsy**)  
  - Bloomberg General Collateral (**Bloomberg GC**)  
  - Daily Federal Funds Rate (**DFF**)  

### 🔑 Key Feature Engineering Steps
- **Decomposition** of the SOFR time series to isolate strong seasonal spikes (e.g., year-end and quarter-end effects).  
- **Time-series preprocessing** using `PowerTransformer` and `MinMaxScaler` for optimal model training.  

---

## 🚀 Key Results (LSTM Model)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **R² (Coefficient of Determination)** | `0.9930` | High correlation between predicted and actual rates |
| **Mean Absolute Error (MAE)** | `0.0948` | Average error in prediction (in rate percentage points) |
| **Mean Absolute Percentage Error (MAPE)** | `1.95%` | Low percentage error → high reliability |

---

## 🛠️ Technology Stack
- **Python**
- Data Manipulation: `pandas`, `numpy`  
- Deep Learning: `TensorFlow`, `Keras` (LSTM implementation)  
- Machine Learning: `scikit-learn`, `XGBoost`  
- Visualization: `matplotlib`, `seaborn`  
- Environment: Jupyter Notebook / Google Colab  

---

## ⚙️ Setup and Installation

### ✅ Prerequisites
- Python **3.8+**

### 📥 Installation
```bash
# Clone repository
git clone https://github.com/your-username/sofr-forecasting.git
cd sofr-forecasting

# Install dependencies
pip install -r requirements.txt
