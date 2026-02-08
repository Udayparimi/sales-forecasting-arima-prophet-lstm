# Sales Forecasting Project: ARIMA, Prophet, LSTM

Predict daily sales for a grocery chain using time series models. **Prophet achieved best MAE of 74k on 90-day holdout**.

## Dataset
- **Kaggle Store Sales**: 5+ years daily sales across 54 stores, 33 product families
- Features: sales, onpromotion (promotional items), seasonality, trend

## Models Implemented
| Model | MAE | RMSE | Strengths |
|-------|-----|------|-----------|
| SARIMAX | 99k | 130k | Baseline with promotions |
| **Prophet** | **74k** | **100k** | Best: seasonality + regressors |
| LSTM | 99k | 130k | Neural net baseline |

## Key Insights
- **Promotions boost sales** by ~38% intensity (promo_intensity feature)
- **Seasonality**: Weekly peaks (Fridays), yearly cycles
- **Trend**: Gradual upward sales growth

## How to run
1. Upload `train.csv` from [Kaggle](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)
2. Open `sales_forecasting_project.ipynb` in Colab/Jupyter
3. Run all cells

## Visualizations
- Forecast vs actual plots for all models
- Prophet trend/seasonality decomposition

![Prophet Forecast](https://via.placeholder.com/800x400?text=Prophet+Forecast+vs+Actual) <!-- Add screenshot here -->

Built with Python, Pandas, Statsmodels, Prophet, TensorFlow.
