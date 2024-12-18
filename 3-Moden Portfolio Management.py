#pip install yfinance numpy pandas matplotlib scipy
#pip install numpy pandas matplotlib scipy powerlaw

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Function to get data
def get_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)['Adj Close']
    return data

# Function to calculate annualized returns and covariance matrix
def calc_annualized_returns(data):
    returns = data.pct_change().mean() * 252  # Daily returns to annualized returns
    return returns

def calc_cov_matrix(data):
    returns = data.pct_change()
    cov_matrix = returns.cov() * 252  # Annualizing the covariance matrix
    return cov_matrix

# Function to calculate portfolio performance
def portfolio_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(weights * mean_returns)
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return std_dev, returns

# Function to minimize the negative Sharpe ratio
def min_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
    portfolio_volatility, portfolio_return = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(portfolio_return - risk_free_rate) / portfolio_volatility

# Function to optimize the portfolio (minimize risk for a given return)
def optimize_portfolio(mean_returns, cov_matrix, risk_free_rate=0.0):
    num_assets = len(mean_returns)
    initial_guess = np.ones(num_assets) / num_assets  # Equal distribution initially
    bounds = tuple((0, 1) for asset in range(num_assets))  # Weights must sum to 1
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})  # Sum of weights equals 1
    result = minimize(min_sharpe_ratio, initial_guess, args=(mean_returns, cov_matrix, risk_free_rate),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result

# Function to plot the efficient frontier
def plot_efficient_frontier(mean_returns, cov_matrix, num_portfolios=10000):
    results = np.zeros((3, num_portfolios))
    for i in range(num_portfolios):
        weights = np.random.random(len(mean_returns))
        weights /= np.sum(weights)  # Normalize weights
        portfolio_volatility, portfolio_return = portfolio_performance(weights, mean_returns, cov_matrix)
        results[0,i] = portfolio_volatility
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return - 0.0) / portfolio_volatility  # Sharpe Ratio

    # Plot
    plt.scatter(results[0,:], results[1,:], c=results[2,:], cmap='viridis', marker='o')
    plt.title('Efficient Frontier')
    plt.xlabel('Volatility (Risk)')
    plt.ylabel('Return')
    plt.colorbar(label='Sharpe Ratio')

# Main function
def main():
    # Define tickers, time period
    tickers = ['AAPL', 'GOOG', 'MSFT', 'AMZN', 'TSLA']  # Example stock tickers
    start_date = '2010-01-01'
    end_date = '2023-01-01'

    # Get data
    data = get_data(tickers, start_date, end_date)

    # Calculate mean returns and covariance matrix
    mean_returns = calc_annualized_returns(data)
    cov_matrix = calc_cov_matrix(data)

    # Optimize portfolio to maximize Sharpe ratio
    optimal_weights = optimize_portfolio(mean_returns, cov_matrix)

    # Print optimal weights
    print("Optimal Portfolio Weights:")
    for i in range(len(tickers)):
        print(f"{tickers[i]}: {optimal_weights.x[i]:.4f}")

    # Plot Efficient Frontier
    plot_efficient_frontier(mean_returns, cov_matrix)
    plt.show()

if __name__ == "__main__":
    main()