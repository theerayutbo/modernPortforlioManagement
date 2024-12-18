import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Define your stock tickers in the array
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'O']  # Example stock tickers
start_date = '2015-01-01'
end_date = '2023-01-01'

# Fetch historical stock data
data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

# Calculate historical returns
returns = data.pct_change().dropna()

# Monte Carlo Simulation setup
num_simulations = 10000  # Number of simulations
num_assets = len(tickers)  # Number of assets based on the tickers array

# Pre-allocate arrays
simulated_returns = np.zeros((num_simulations, num_assets))
simulated_portfolio_returns = np.zeros(num_simulations)
simulated_portfolio_volatility = np.zeros(num_simulations)
simulated_weights = np.zeros((num_simulations, num_assets))

# Simulate portfolio returns and weights
for i in range(num_simulations):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)  # Normalize to get the sum of weights = 1
    simulated_weights[i, :] = weights

    # Portfolio return and volatility calculation
    portfolio_return = np.sum(weights * returns.mean()) * 252  # Annualize return
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))  # Annualize volatility
    
    simulated_portfolio_returns[i] = portfolio_return
    simulated_portfolio_volatility[i] = portfolio_volatility

# Find the optimal portfolios
# Maximum Sharpe Ratio Portfolio
sharpe_ratios = simulated_portfolio_returns / simulated_portfolio_volatility
max_sharpe_indices = np.argsort(sharpe_ratios)[-5:]  # Top 5 Max Sharpe Ratio

# Minimum Volatility Portfolio
min_volatility_indices = np.argsort(simulated_portfolio_volatility)[:5]  # Top 5 Min Volatility

# Maximum Profit Portfolio (Maximum Return)
max_profit_indices = np.argsort(simulated_portfolio_returns)[-5:]  # Top 5 Max Profit

# Results for Maximum Sharpe Ratio
max_sharpe_results = pd.DataFrame({
    'Simulation Number': max_sharpe_indices + 1,  # Add simulation number
    'Portfolio Return': simulated_portfolio_returns[max_sharpe_indices],
    'Portfolio Volatility': simulated_portfolio_volatility[max_sharpe_indices],
    'Sharpe Ratio': sharpe_ratios[max_sharpe_indices],
    **{ticker: simulated_weights[max_sharpe_indices, idx] for idx, ticker in enumerate(tickers)}
})

# Results for Minimum Volatility
min_volatility_results = pd.DataFrame({
    'Simulation Number': min_volatility_indices + 1,  # Add simulation number
    'Portfolio Return': simulated_portfolio_returns[min_volatility_indices],
    'Portfolio Volatility': simulated_portfolio_volatility[min_volatility_indices],
    'Sharpe Ratio': simulated_portfolio_returns[min_volatility_indices] / simulated_portfolio_volatility[min_volatility_indices],
    **{ticker: simulated_weights[min_volatility_indices, idx] for idx, ticker in enumerate(tickers)}
})

# Results for Maximum Profit
max_profit_results = pd.DataFrame({
    'Simulation Number': max_profit_indices + 1,  # Add simulation number
    'Portfolio Return': simulated_portfolio_returns[max_profit_indices],
    'Portfolio Volatility': simulated_portfolio_volatility[max_profit_indices],
    'Sharpe Ratio': simulated_portfolio_returns[max_profit_indices] / simulated_portfolio_volatility[max_profit_indices],
    **{ticker: simulated_weights[max_profit_indices, idx] for idx, ticker in enumerate(tickers)}
})

# Display the top 5 results for each portfolio type
print("Top 5 Maximum Sharpe Ratio Portfolios:")
print(max_sharpe_results)

print("\nTop 5 Minimum Volatility Portfolios:")
print(min_volatility_results)

print("\nTop 5 Maximum Profit Portfolios:")
print(max_profit_results)

# Plotting the results
plt.figure(figsize=(10, 6))

# Plot Efficient Frontier
plt.scatter(simulated_portfolio_volatility, simulated_portfolio_returns, c=simulated_portfolio_returns / simulated_portfolio_volatility, cmap='viridis', marker='o', s=10)
plt.title('Efficient Frontier')
plt.xlabel('Volatility (Standard Deviation)')
plt.ylabel('Return')
plt.colorbar(label='Sharpe Ratio')

# Plot Capital Market Line (CML)
max_sharpe_volatility = simulated_portfolio_volatility[max_sharpe_indices][0]
max_sharpe_return = simulated_portfolio_returns[max_sharpe_indices][0]
plt.plot([0, max_sharpe_volatility], [0, max_sharpe_return], color='r', linewidth=2, label="Capital Market Line")
plt.scatter(max_sharpe_volatility, max_sharpe_return, color='red', marker='*', s=200, label="Market Portfolio")

# Plot Maximum Sharpe Ratio Portfolio (green)
plt.scatter(max_sharpe_volatility, max_sharpe_return, color='green', marker='^', s=200, label="Maximum Sharpe Ratio Portfolio")

# Plot Minimum Volatility Portfolio (red)
min_volatility_volatility = simulated_portfolio_volatility[min_volatility_indices][0]
min_volatility_return = simulated_portfolio_returns[min_volatility_indices][0]
plt.scatter(min_volatility_volatility, min_volatility_return, color='red', marker='v', s=200, label="Minimum Volatility Portfolio")

# Plot Maximum Profit Portfolio (yellow)
max_profit_volatility = simulated_portfolio_volatility[max_profit_indices][0]
max_profit_return = simulated_portfolio_returns[max_profit_indices][0]
plt.scatter(max_profit_volatility, max_profit_return, color='yellow', marker='s', s=200, label="Maximum Profit Portfolio")

plt.legend(loc='upper left')
plt.show()

# Maximum Sharpe Ratio Portfolio:
# The Sharpe Ratio is a measure of the risk-adjusted return of an investment.
# It is calculated as:
#   Sharpe Ratio = (Portfolio Return - Risk-Free Rate) / Portfolio Volatility
# In this case, we assume the Risk-Free Rate is zero for simplicity.
# The Maximum Sharpe Ratio Portfolio is the portfolio that offers the highest 
# return per unit of risk (volatility). 
# It is considered the best portfolio in terms of risk-return trade-off, as it 
# maximizes the return relative to the amount of risk taken.
# 
# To calculate this portfolio, we compute the Sharpe ratio for each simulated 
# portfolio and select the one with the highest Sharpe ratio.
#
# In the code, `max_sharpe_indices` stores the indices of the top 5 portfolios 
# with the highest Sharpe ratios.

# Minimum Volatility Portfolio:
# The Minimum Volatility Portfolio is the portfolio that minimizes the overall risk (volatility) 
# regardless of the expected return. 
# This portfolio aims to provide the least amount of risk (i.e., the lowest portfolio volatility) 
# for investors who are risk-averse.
# 
# In this case, the portfolios are selected based on the lowest volatility values, 
# regardless of their returns. This approach is suitable for conservative investors 
# who prioritize stability and seek to reduce the risk of significant losses.
#
# The `min_volatility_indices` stores the indices of the portfolios with the lowest volatility 
# from the simulation.

# Maximum Profit Portfolio (Maximum Return Portfolio):
# The Maximum Profit Portfolio, also known as the Maximum Return Portfolio, 
# aims to maximize the expected return of the portfolio. 
# This strategy focuses on achieving the highest possible return, regardless of the risk 
# or volatility associated with the portfolio.
# 
# It may not be suitable for all investors because it could lead to high levels of risk, 
# but it can be attractive for investors with a high risk tolerance.
#
# In the code, the portfolios are ranked based on their expected returns, 
# and the top 5 portfolios with the highest returns are selected using `max_profit_indices`.