from flask import Flask, render_template, request, jsonify
import numpy as np
import yfinance as yf

app = Flask(__name__)

# Function to calculate stock VaR using Monte Carlo simulation
def calculate_var(stock_symbol, investment, confidence_level, days):
    # Retrieve stock data
    stock_data = yf.download(stock_symbol, period="5y")['Close']
    
    # Calculate log returns
    log_returns = np.log(1 + stock_data.pct_change())
    mu = log_returns.mean() * 252  # Annualized return
    sigma = log_returns.std() * np.sqrt(252)  # Annualized volatility

    # Monte Carlo Simulation
    simulations = 10000
    dt = 1 / 252
    price_paths = np.zeros((days, simulations))
    price_paths[0] = stock_data.iloc[-1]
    for t in range(1, days):
        Z = np.random.standard_normal(simulations)
        price_paths[t] = price_paths[t - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

    # Calculate returns
    final_prices = price_paths[-1]
    returns = (final_prices - price_paths[0]) / price_paths[0]
    sorted_returns = np.sort(returns)
    
    # Calculate VaR
    var_index = int((1 - confidence_level / 100) * len(sorted_returns))
    var_percent = sorted_returns[var_index]
    var_dollar = investment * var_percent

    return var_dollar

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate-risk', methods=['POST'])
def risk():
    data = request.json
    stock_symbol = data['stockSymbol']
    investment = float(data['investment'])
    confidence = float(data['confidence'])
    days = int(data['days'])

    # Calculate VaR
    var = calculate_var(stock_symbol, investment, confidence, days)

    return jsonify({'var': f"${var:.2f}"})

if __name__ == '__main__':
    app.run(debug=True)
