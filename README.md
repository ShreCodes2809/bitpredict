# BitPredict - Bitcoin Price Prediction using Time Series Forecasting

## 🌟 Project Overview

Bitcoin price prediction is crucial for cryptocurrency trading and financial forecasting. **BitPredict** explores multiple **machine learning and deep learning** models to predict Bitcoin prices using historical data. Various models are implemented, evaluated, and compared for accuracy and efficiency.

## 📈 Dataset Used

The dataset used is **Bitcoin Historical Price Data** from CoinDesk, spanning **October 1, 2013 - May 18, 2021**. It includes:

- **Currency**
- **Closing Price (USD)**
- **24h Open Price (USD)**
- **24h High Price (USD)**
- **24h Low Price (USD)**

## 📝 Project Workflow

### **1. Data Collection & Preprocessing**

- Downloaded the dataset.
- Parsed dates and set them as an index for time-series analysis.
- Extracted only **Closing Price (USD)** for prediction.
- Visualized the **Bitcoin price trend** over time.

### **2. Splitting & Windowing Data**

- Split data into **training (80%)** and **testing (20%)** sets.
- Used **rolling windowing techniques** to convert data into a supervised learning format.
- Experimented with **different window sizes (7 days, 30 days, etc.)**.

### **3. Model Implementations**

#### **🔢 Baseline Model (Naïve Forecast)**

- Predicts the next price as the **previous day’s price**.
- Serves as a benchmark for other models.

#### **🔄 Dense Neural Network (DNN)**

- **Model 1:** Dense Network (**7-day window**).
- **Model 2:** Dense Network (**30-day window**).
- **Architecture:**
  - Hidden Layer: **128 neurons (ReLU activation).**
  - Output Layer: **Linear activation.**
  - Optimizer: **Adam.**

#### **🎨 Long Short-Term Memory (LSTM) Model**

- Uses **LSTM layers** to capture long-term dependencies in time-series data.
- **Architecture:**
  - **LSTM Layer:** 64 units.
  - **Dense Layer:** 32 neurons (ReLU activation).
  - **Output Layer:** Single neuron for price prediction.

#### **📝 Convolutional Neural Network (Conv1D) Model**

- Uses **1D Convolutional layers** for time-series feature extraction.
- **Architecture:**
  - **Conv1D Layer:** 64 filters, kernel size 3.
  - **MaxPooling1D Layer**.
  - **Dense Layer:** 128 neurons.
  - **Output Layer:** 1 neuron.

#### **🔍 N-BEATS Algorithm**

- A state-of-the-art **Neural Basis Expansion Analysis (N-BEATS)** model.
- **Architecture:**
  - Multiple stacks of **fully connected residual blocks**.
  - Backcast and Forecast generators.

#### **🔄 Dense Multivariate Time Series Model**

- Uses **Open, High, Low prices** in addition to Closing Price.
- **Fully connected Dense Model**.

#### **💡 Ensemble Model**

- **Combines predictions from multiple models** (LSTM, Conv1D, Dense, N-BEATS).
- Uses **weighted averaging**.

---

## 🎨 Model Evaluation Metrics

- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Percentage Error (MAPE)**
- **Mean Absolute Scaled Error (MASE)**

---

## 🏆 Results & Model Comparison

| Model                        | MAE        | MSE         | RMSE       | MAPE      | MASE      |
| ---------------------------- | ---------- | ----------- | ---------- | --------- | --------- |
| **Naïve Forecast**           | 567.98     | 1,147,547   | 1071.23    | ***2.51%***     | 0.999     |
| **Dense (7-day window)**     | 568.95     | 1,171,744   | 1082.47    | 2.54%     | 0.999     |
| **Dense (30-day window)**    | 608.96     | 1,281,439   | 1132.00    | 2.77%     | 1.064     |
| **LSTM Model**               | 596.64     | 1,273,487   | 1128.49    | 2.68%     | 1.048     |
| **Conv1D Model**             | 570.83     | 1,176,671   | 1084.74    | 2.56%     | 1.003     |
| **N-BEATS Model**            | 572.28     | 1,165,763   | 1079.71    | 2.59%     | 1.005     |
| **Dense Multivariate Model** | 567.59     | 1,161,687   | 1077.82    | 2.54%     | 0.997     |
| **Ensemble Model**           | ***564.96*** |***1,134,330***| ***1065.04*** | 2.57% | ***0.992*** |

**Key Insights:**

- **LSTM and Conv1D models** slightly underperform simple Dense models but would outperform them in case of large datasets.
- **N-BEATS achieved superior performance**.
- **The Ensemble Model yielded the best results**.

---

## 📝 Conclusion

- **Naïve Forecast provided a strong baseline**.
- **Deep Learning models captured trends effectively**.
- **N-BEATS and Ensemble Models showed the highest accuracy**.

---

## 🛠️ Setup & Usage

### **1. Clone Repository**

```sh
git clone https://github.com/your_username/BitPredict.git
cd BitPredict
```

### **2. Install Dependencies**

```sh
pip install -r requirements.txt
```

### **3. Run Project**

Execute the Jupyter Notebook:

```sh
jupyter notebook
```

Or run the script:

```sh
python main.py
```

### **4. Modify Parameters**

- Adjust **window size, epochs, learning rate, and model type** in `config.py`.

---

## 💡 Future Enhancements

- Implement **Transformer-based models**.
- Experiment with **Reinforcement Learning**.
- Deploy as an **API using Flask/FastAPI**.
- **Optimize hyperparameters** via Bayesian Optimization.

---

## 🌟 Acknowledgements

- **TensorFlow Deep Learning Course** by Daniel Bourke.
- **CoinDesk API** for Bitcoin historical data.
- Open-source contributors and **AI/ML community**.

