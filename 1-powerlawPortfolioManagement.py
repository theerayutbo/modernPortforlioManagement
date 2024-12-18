#pip install numpy pandas matplotlib scipy powerlaw

import numpy as np
import pandas as pd
import powerlaw
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm

# Simulated asset returns (replace with real data)
np.random.seed(42)
returns = np.random.normal(loc=0.0005, scale=0.02, size=1000)  # Simulate returns for one asset

# Power Law Estimation: Estimating the exponent using the `powerlaw` package
def estimate_power_law(data):
    results = powerlaw.Fit(data)
    alpha = results.alpha
    xmin = results.xmin
    return alpha, xmin

# Estimate Power Law exponent
alpha, xmin = estimate_power_law(returns)
print(f"Estimated Power Law Exponent: {alpha}, x_min: {xmin}")

# VaR Calculation based on Power Law distribution
def calculate_var(data, alpha, quantile=0.05):
    sorted_data = np.sort(data)
    # Tail distribution follows Power Law, find the quantile at the tail
    return sorted_data[int(len(sorted_data) * quantile)]

# Calculate Value at Risk (VaR) at 5% quantile
var_5 = calculate_var(returns, alpha, quantile=0.05)
print(f"Value at Risk (5% quantile): {var_5}")

# Conditional VaR (CVaR) Calculation
def calculate_cvar(data, var_5):
    # Calculate CVaR as the average loss beyond the VaR
    return np.mean(data[data <= var_5])

cvar_5 = calculate_cvar(returns, var_5)
print(f"Conditional VaR (CVaR) at 5% quantile: {cvar_5}")

# Portfolio Optimization
def portfolio_performance(weights, mean_returns, cov_matrix):
    # Assuming Power Law returns, calculate the portfolio return and risk
    portfolio_return = np.sum(weights * mean_returns)
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_risk, portfolio_return

def optimize_portfolio(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    initial_guess = np.ones(num_assets) / num_assets  # Equal distribution initially
    bounds = tuple((0, 1) for asset in range(num_assets))  # Weights must sum to 1
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})  # Sum of weights equals 1
    result = minimize(lambda w: -portfolio_performance(w, mean_returns, cov_matrix)[1], 
                      initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

# Example: Portfolio with 3 assets (use real return data for more assets)
assets_returns = np.random.normal(0.0005, 0.02, size=(1000, 3))  # Simulating 3 assets
mean_returns = np.mean(assets_returns, axis=0)
cov_matrix = np.cov(assets_returns.T)

optimal_weights = optimize_portfolio(mean_returns, cov_matrix)
print(f"Optimal Portfolio Weights: {optimal_weights.x}")

# Plotting the Power Law fit
plt.figure(figsize=(10,6))
powerlaw.Fit(returns).plot_pdf(color='blue')
plt.title('Power Law Distribution Fit for Asset Returns')
plt.xlabel('Returns')
plt.ylabel('PDF')
plt.show()