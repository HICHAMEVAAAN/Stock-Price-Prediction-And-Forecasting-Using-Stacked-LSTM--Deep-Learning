from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load model and scaler
model = load_model("model/lstm_model.h5")
scaler = joblib.load("model/scaler.joblib")

n_steps = 100  # LSTM time steps

def predict_next_days(input_data, n_steps=100, days=30):
    temp_input = list(input_data)
    lst_output = []

    for i in range(days):
        x_input = np.array(temp_input[-n_steps:])
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.append(yhat[0][0])
        lst_output.append(yhat[0][0])
    return scaler.inverse_transform(np.array(lst_output).reshape(-1,1))

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    plot_url = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            df = pd.read_csv(file)
            close_prices = df['close'].values[-n_steps:]
            scaled_input = scaler.transform(close_prices.reshape(-1,1))
            prediction = predict_next_days(scaled_input, n_steps, 30)

            # Plotting
            plt.figure(figsize=(10,5))
            plt.plot(range(len(close_prices)), close_prices, label='Historical Prices')
            plt.plot(range(len(close_prices), len(close_prices)+30), prediction, label='Predicted Prices', color='red')
            plt.xlabel('Days')
            plt.ylabel('Price')
            plt.legend()
            plt.tight_layout()
            plot_path = "static/plot.png"
            plt.savefig(plot_path)
            plt.close()
            plot_url = plot_path

    return render_template('index.html', prediction=prediction, plot_url=plot_url)

if __name__ == "__main__":
    app.run(debug=True)


