# Stock Price Forecasting - Seq2Seq + Prophet + XGBoost + Ensemble

A comprehensive machine learning framework for multi-step stock price forecasting using ensemble methods combining deep learning, statistical models, and gradient boosting.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Model Details](#model-details)
- [Results](#results)
- [File Structure](#file-structure)
- [Configuration](#configuration)
- [Risk Analysis](#risk-analysis)
- [Disclaimer](#disclaimer)

## 🎯 Overview

This project implements a sophisticated stock price forecasting system that combines multiple machine learning approaches:

- **Seq2Seq LSTM**: Deep learning model for sequence-to-sequence prediction
- **Facebook Prophet**: Statistical forecasting model
- **XGBoost**: Gradient boosting for time series
- **Ensemble Method**: Intelligent weighting of all models
- **Monte Carlo Simulation**: Risk assessment and uncertainty quantification
- **Volatility Forecasting**: Multi-model volatility prediction

The system automatically identifies correlated assets, performs feature engineering, and generates both price and volatility forecasts with confidence intervals.

## ✨ Features

- **Multi-Asset Correlation Analysis**: Identifies top correlated stocks for enhanced forecasting
- **Multi-Model Ensemble**: Combines Seq2Seq, Prophet, and XGBoost with dynamic weights
- **Probabilistic Forecasting**: Monte Carlo simulations for confidence intervals
- **Volatility Prediction**: Separate models for price and volatility forecasting
- **Comprehensive Evaluation**: Multiple metrics (RMSE, MAE, MAPE, Directional Accuracy)
- **Risk Assessment**: VaR, CVaR, and Sharpe ratio calculations
- **Automated Pipeline**: End-to-end from data download to final recommendation
- **Visual Analytics**: Comparative plots and diagnostic charts

## 🏗️ Architecture

### Data Flow
```
Yahoo Finance Data → Feature Engineering → Correlation Analysis → 
Multi-Model Training → Ensemble Forecasting → Risk Analysis → Final Recommendation
```

### Model Stack
- **Primary**: Seq2Seq LSTM (Encoder-Decoder architecture)
- **Secondary**: Facebook Prophet (Additive regression)
- **Tertiary**: XGBoost (Gradient boosting)
- **Ensemble**: Weighted combination (Seq2Seq: 0.5, Prophet: 0.3, XGBoost: 0.2)

## 🚀 Installation

### Prerequisites

```bash
# Core data science libraries
pip install numpy pandas matplotlib seaborn scikit-learn joblib tqdm

# Financial data
pip install yfinance

# Deep learning
pip install tensorflow

# Statistical models
pip install prophet

# Gradient boosting
pip install xgboost
```

### Required Files

- `src/finance_utils.py`: Utility functions for financial analysis
- `data/top100_stocks.csv`: Dataset of top 100 most traded stocks with metadata

## 📖 Usage

### Basic Configuration

```python
# Set target ticker and parameters
TICKER = "TSLA"
top_n = 2          # Number of correlated assets
period = "5y"      # Historical data period
SEQ_LEN = 60       # Historical window (days)
HORIZON = 30       # Forecast horizon (days ahead)
```

### Running the Pipeline

1. **Asset Correlation Analysis**:
   ```python
   results = get_top_correlated_stocks(
       ticker=TICKER,
       top_n=top_n,
       period=period,
       csv_path=csv_path
   )
   ```

2. **Data Preparation**:
   - Downloads 5 years of historical data
   - Calculates log returns and technical indicators
   - Selects relevant features based on correlation

3. **Model Training**:
   - Automatic training of Seq2Seq, Prophet, and XGBoost models
   - Validation splitting with temporal consistency
   - Early stopping and learning rate scheduling

4. **Forecast Generation**:
   - Multi-step predictions from all models
   - Ensemble combination with intelligent weighting
   - Monte Carlo simulations for uncertainty

5. **Results Analysis**:
   - Comprehensive metrics calculation
   - Visual comparisons and diagnostics
   - Risk assessment and final recommendation

## 🔬 Methodology

### Feature Engineering

- **Price Features**: Raw prices, log returns
- **Volume Features**: Trading volumes
- **Technical Indicators**:
  - 5-day rolling mean of returns
  - 10-day rolling volatility (standard deviation)
  - Lag features (1, 2, 3, 5, 10 days)

### Seq2Seq Architecture

```
Encoder: LSTM(128) → [state_h, state_c]
Decoder: LSTM(128) → TimeDistributed(Dense(1))
Input: [SEQ_LEN, n_features] → Output: [HORIZON, 1]
```

### Training Strategy

- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam with ReduceLROnPlateau
- **Validation**: 15% temporal split
- **Regularization**: Early stopping with patience=10

## 🤖 Model Details

### Seq2Seq LSTM
- **Encoder**: Processes historical sequences to extract temporal patterns
- **Decoder**: Generates multi-step forecasts autoregressively
- **Advantage**: Captures complex temporal dependencies

### Facebook Prophet
- **Approach**: Additive regression with trend, seasonality, and holiday effects
- **Configuration**: Weekly seasonality enabled, optimized changepoint prior
- **Advantage**: Handles missing data and outliers robustly

### XGBoost
- **Features**: Lag returns, rolling statistics, technical indicators
- **Configuration**: 200 estimators, max_depth=4, learning_rate=0.05
- **Advantage**: Fast training, handles non-linear relationships

### Ensemble Method
```python
def smart_ensemble(seq2seq_preds, prophet_preds, xgb_preds):
    weights = {'seq2seq': 0.5, 'prophet': 0.3, 'xgb': 0.2}
    # Normalizes weights and computes weighted average
```

## 📊 Results

### Output Metrics
- **Point Forecasts**: Price predictions for next HORIZON days
- **Uncertainty Quantification**: 5th-95th percentile confidence intervals
- **Performance Metrics**: RMSE, MAE, MAPE, Directional Accuracy
- **Risk Metrics**: VaR 95%, CVaR 95%, Sharpe Ratio

### Visual Outputs
1. **Price Forecast Comparison**: All models vs historical prices
2. **Monte Carlo Simulations**: Probabilistic forecast ranges
3. **Error Analysis**: Model error distributions and correlations
4. **Volatility Forecasts**: Predicted volatility trajectories

## 📁 File Structure

```
project/
├── src/
│   └── finance_utils.py          # Utility functions
├── data/
│   └── top100_stocks.csv         # Stock metadata
├── model_outputs/                # Generated during execution
│   ├── scaler_X.pkl             # Feature scaler
│   ├── scaler_y.pkl             # Target scaler
│   ├── final_forecasts.csv      # Complete forecast results
│   └── training_windows.npy     # Preprocessed sequences
├── notebooks/
│   └── stock_forecasting.ipynb  # Main analysis notebook
└── README.md
```

## ⚙️ Configuration

### Key Parameters

```python
# Model Architecture
SEQ_LEN = 60        # Historical sequence length
HORIZON = 30        # Forecast horizon
LATENT_DIM = 128    # LSTM hidden units
BATCH_SIZE = 64     # Training batch size
EPOCHS = 60         # Maximum training epochs

# Data Parameters
USE_MULTIVAR = False  # Use multiple correlated assets
VALIDATION_SPLIT = 0.15  # Validation set ratio

# Feature Selection
CORRELATION_THRESHOLD = 0.2  # Minimum correlation to include asset
```

### Asset Selection Logic

The system automatically:
1. Downloads top 100 stocks metadata
2. Calculates historical correlations with target ticker
3. Selects top N correlated assets (correlation > 0.2)
4. Includes their price, volume, and derived features

## 📈 Risk Analysis

### Monte Carlo Simulation
- **Paths**: 500 simulated price trajectories per model
- **Volatility Estimation**: Based on recent 10-day returns std
- **Output**: Mean forecast with 5th-95th confidence intervals

### Risk Metrics
- **Value at Risk (VaR)**: Worst expected loss in 5% tail
- **Conditional VaR (CVaR)**: Average loss in worst 5% cases
- **Sharpe Ratio**: Risk-adjusted return (annualized)

## ⚠️ Disclaimer

**Important**: This tool is for educational and research purposes only. The forecasts are statistical estimates based on historical data and should not be considered financial advice. 

- Always conduct your own research
- Consider macroeconomic factors and company fundamentals
- Implement proper risk management strategies
- Past performance does not guarantee future results

## 🔧 Customization

### Adding New Models
```python
# Example: Add ARIMA model
from statsmodels.tsa.arima.model import ARIMA

def arima_forecast(series, horizon):
    model = ARIMA(series, order=(5,1,0))
    fitted = model.fit()
    return fitted.forecast(horizon)
```

### Modifying Ensemble Weights
```python
# Adjust based on model performance
weights = {
    'seq2seq': 0.6,    # Increase if LSTM performs best
    'prophet': 0.25,   # Decrease if needed
    'xgb': 0.15        # Adjust based on validation
}
```

## 📞 Support

For questions or issues:
1. Check the configuration parameters
2. Verify data availability from Yahoo Finance
3. Ensure all dependencies are correctly installed
4. Review the error messages and adjust parameters accordingly

---

**Note**: This system is designed for professional quantitative analysis and requires understanding of financial markets and machine learning concepts.
