# **BitPredict: Bitcoin Price Prediction Using Time Series Models** 📈

This project aims to predict Bitcoin prices using various machine learning and deep learning models, leveraging time series analysis techniques. The project explores effective preprocessing, windowing strategies, and modeling to forecast cryptocurrency prices.

---

## **Project Overview**
Cryptocurrencies like Bitcoin are highly volatile, making accurate price prediction challenging. This project utilizes time series modeling techniques to predict Bitcoin's closing prices based on historical data. Models range from baseline naive forecasts to advanced dense neural networks with varying window sizes.

Key objectives:
- Explore data preprocessing for time series.
- Apply naive, window-based, and dense models for prediction.
- Compare models using metrics like MAE, MSE, RMSE, and MAPE.

---

## **Dataset**
- **Source**: [Bitcoin Historical Data](https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/extras/BTC_USD_2013-10-01_2021-05-18-CoinDesk.csv)
- **Features**:
  - `Date`: The timestamp of the price.
  - `Closing Price (USD)`: The daily closing price of Bitcoin.
  - `24h High`, `24h Low`, and `24h Open`.

---

## **Project Workflow**
1. **Data Preprocessing**:
   - Extracted daily closing prices.
   - Visualized historical trends using matplotlib.
   - Split data into training and testing sets (80/20).

2. **Windowing for Time Series**:
   - Defined windows of size `7` (week) and `30` days.
   - Implemented supervised learning problems with window-label pairs.

3. **Baseline Model**:
   - Naive forecast: Predict the price of the next timestep as the current timestep's value.
   - Metrics: MAE: 567.98, MSE: 1,147,547.

4. **Deep Learning Models**:
   - **Dense Neural Network**:
     - Layer structure: 1 hidden dense layer with 128 neurons (ReLU) + 1 output layer (linear activation).
     - Best model achieved:
       - MAE: 568.95
       - MAPE: 2.54%
   - **Dense Neural Network with Extended Window Size**:
     - Increased window size to 30 days.
     - Improved accuracy with longer historical context.

---

## **Model Evaluation Metrics**
The following metrics were used for evaluation:
- **MAE** (Mean Absolute Error): Measures average absolute differences between actual and predicted values.
- **MSE** (Mean Squared Error): Measures average squared differences.
- **RMSE** (Root Mean Squared Error): Square root of MSE for better interpretability.
- **MAPE** (Mean Absolute Percentage Error): Measures prediction errors as percentages.

---

## **Results**
### Baseline Model
- **MAE**: 567.98  
- **MSE**: 1,147,547  
- **RMSE**: 1071.23  

### Dense Neural Network (7-day window)
- **MAE**: 568.95  
- **MSE**: 1,171,744  
- **RMSE**: 1082.47  

### Dense Neural Network (30-day window)
- Improved learning and validation metrics with extended window size.

---

## **Setup and Usage**
1. Clone the repository:
   ```bash
   git clone https://github.com/ShreCodes2809/bitpredict.git
   cd BitPredict
   ```

2. Download the dataset:
   ```bash
   wget https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/extras/BTC_USD_2013-10-01_2021-05-18-CoinDesk.csv
   ```

---

## **Future Enhancements**
- Implement transformer-based models for improved predictions.
- Introduce additional features like trading volume and sentiment analysis.
- Optimize hyperparameters for better model performance.
- Explore LSTM and GRU models for sequential data handling.

---

## **Acknowledgments**
This project was inspired by TensorFlow's time series forecasting tutorial. Thanks to the cryptocurrency and machine learning communities for the resources and inspiration.

---
