# 📈 Stock Market Prediction and Forecasting using Stacked LSTM  

## 📌 Overview  
This project predicts and forecasts **Apple Stock Prices (AAPL)** using a **Stacked LSTM (Long Short-Term Memory) model**.  
It uses historical stock data, applies preprocessing with MinMaxScaler, and trains an LSTM deep learning model with TensorFlow/Keras.  

---

## 🛠 Tools & Technologies  
- Python  
- NumPy & Pandas  
- Matplotlib & Seaborn  
- Scikit-learn  
- TensorFlow / Keras  
- Jupyter Notebook  

---

## 📂 Project Structure  

├── app.py # (Optional) Streamlit app for deployment

├── stock_lstm.ipynb # Jupyter Notebook (EDA + Model training)

├── AAPL.csv # Stock data file

├── requirements.txt # Dependencies

├── model/ # Saved trained model & scaler

└── .gitignore # Ignored files


---

## ⚙️ Model Workflow  
1. **Data Collection** – Fetch stock price data using `pandas_datareader`.  
2. **Preprocessing** – Scaling with MinMaxScaler.  
3. **Data Splitting** – Train/Test split (65% training, 35% testing).  
4. **Sequence Creation** – Creating time-step-based input sequences.  
5. **Model Building** – Stacked LSTM layers with Keras.  
6. **Model Training** – 100 epochs, batch size = 64.  
7. **Evaluation** – RMSE calculation for training & testing.  
8. **Prediction** – Forecasting future stock prices (next 30 days).  
9. **Visualization** – Comparing actual vs. predicted stock trends.  


## 🚀 How to Run  

Clone the repository and install requirements:  git clone https://github.com/your-username/Stock-Market-Prediction.git

cd Stock-Market-Prediction

pip install -r requirements.txt 

Run the Jupyter Notebook: jupyter notebook stock_lstm.ipynb

(Optional) Run the Streamlit app: streamlit run app.py

## 📸 Historical Closing Price Trend of Apple (AAPL)


<img width="628" height="427" alt="Screenshot 2025-09-29 053436" src="https://github.com/user-attachments/assets/690cc1a4-dfcf-41db-aa32-56c1fb831e40" />

## 📊 Results
The model achieves a low RMSE score on both training and test data.

Predicted trends closely follow the actual Apple stock price.

Next 30 days of stock prices are forecasted using the trained LSTM.


## 📧 Support
For queries or suggestions, feel free to connect:

📩 Email: zuhairzia1@gmail.com

💼 LinkedIn: www.linkedin.com/in/zuhairzia
📩 Email: zuhairzia1@gmail.com
