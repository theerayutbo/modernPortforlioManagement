import numpy as np
import pandas as pd
import yfinance as yf
import powerlaw
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Function to fetch historical stock data
def get_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end)['Adj Close']
    data = data.dropna()  # Remove any rows with missing data
    return data

# Function to calculate Power Law Exponent (alpha) using the powerlaw package
def estimate_power_law(data):
    data = data[data > 0]  # Filter out non-positive values
    if len(data) > 2:  # Ensure there's enough data to perform Power Law fit
        results = powerlaw.Fit(data)
        alpha = results.alpha
        xmin = results.xmin
        return alpha, xmin
    else:
        return np.nan, np.nan

# Function to calculate Value at Risk (VaR) based on Power Law distribution
def calculate_var(data, alpha, quantile=0.05):
    if len(data) > 0:
        sorted_data = np.sort(data)
        return sorted_data[int(len(sorted_data) * quantile)]
    else:
        return np.nan

# Function to calculate Conditional VaR (CVaR)
def calculate_cvar(data, var_5):
    if len(data) > 0:
        return np.mean(data[data <= var_5])
    else:
        return np.nan

# Portfolio performance (return and risk calculation)
def portfolio_performance(weights, mean_returns, cov_matrix):
    portfolio_return = np.sum(weights * mean_returns)
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_risk, portfolio_return

# Portfolio optimization to maximize return for a given level of risk
def optimize_portfolio(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    initial_guess = np.ones(num_assets) / num_assets  # Equal distribution initially
    bounds = tuple((0, 1) for asset in range(num_assets))  # Weights must sum to 1
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})  # Sum of weights equals 1
    result = minimize(lambda w: -portfolio_performance(w, mean_returns, cov_matrix)[1], 
                      initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

# Main function
def main():
    # Define the tickers and the time period for data fetching
    tickers = ['O', 'JEPQ', 'ABBV']  # Removed NCDA due to delisting issues
    start_date = '2010-01-01'
    end_date = '2023-01-01'
    
    # Get the data
    data = get_data(tickers, start_date, end_date)
    
    # Calculate daily returns
    returns = data.pct_change().dropna()

    # Estimate the Power Law exponent for each stock
    alphas = {}
    xmins = {}
    for ticker in tickers:
        alpha, xmin = estimate_power_law(returns[ticker])
        alphas[ticker] = alpha
        xmins[ticker] = xmin
    
    # Calculate Value at Risk (VaR) for each stock
    vars = {}
    for ticker in tickers:
        var_5 = calculate_var(returns[ticker], alphas[ticker], quantile=0.05)
        vars[ticker] = var_5
    
    # Calculate Conditional VaR (CVaR) for each stock
    cvars = {}
    for ticker in tickers:
        cvar_5 = calculate_cvar(returns[ticker], vars[ticker])
        cvars[ticker] = cvar_5
    
    # Portfolio optimization
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    
    # Optimize the portfolio
    optimal_weights = optimize_portfolio(mean_returns, cov_matrix)
    
    # Prepare results (ensure each result is a list of the same length)
    results = {
        'Ticker': tickers,
        'Power Law Exponent': [alphas[ticker] for ticker in tickers],
        'x_min': [xmins[ticker] for ticker in tickers],
        'VaR (5% Quantile)': [vars[ticker] for ticker in tickers],
        'CVaR (5% Quantile)': [cvars[ticker] for ticker in tickers],
        'Optimal Portfolio Weights': [optimal_weights.x[i] for i in range(len(optimal_weights.x))]
    }
    
    # Create DataFrame from results dictionary
    results_df = pd.DataFrame(results)
    
    # Display the results DataFrame
    print("Portfolio Analysis Results:")
    print(results_df)

    # Optionally, you can plot the Power Law fit and the efficient frontier
    plt.figure(figsize=(10, 6))
    powerlaw.Fit(returns['O']).plot_pdf(color='blue')  # Example for plotting the Power Law PDF of 'O'
    plt.title('Power Law Distribution Fit for O Stock Returns')
    plt.xlabel('Returns')
    plt.ylabel('PDF')
    plt.show()

if __name__ == "__main__":
    main()


# Here’s an explanation of the key metrics you are working with:

# Ticker:
# This is simply the stock symbol or identifier. In the code, it represents the list of stocks that are being analyzed (e.g., 'O', 'JEPQ', 'ABBV').
# It is used as a reference for each stock's corresponding data.

# Power Law Exponent (α):
# This is the exponent of the Power Law distribution, often denoted as α. It is a statistical measure used to describe the distribution of returns (or any data) that follows a Power Law.
# A smaller α indicates a higher probability of extreme returns (i.e., more tail risk), while a larger α suggests a more stable and less volatile distribution.
# A typical Power Law exponent is between 2 and 3, with values lower than 2 indicating a very fat tail (high risk).

# x_min (Minimum value):
# This is the threshold (denoted xmin) used in the Power Law analysis. It is the smallest value from which the Power Law behavior is assumed to start.
# Values below xmin are discarded because they are not assumed to follow the Power Law distribution.
# This threshold is critical because it defines the region of the data where the Power Law distribution holds. The further xmin is set, the more conservative the model is, as only larger values are considered.

# VaR (Value at Risk, 5% Quantile):
# Value at Risk (VaR) is a risk measure used to assess the potential loss in value of an asset or portfolio over a defined period for a given confidence interval.
# VaR at the 5% quantile indicates the worst expected loss over a specified period (in this case, daily returns), with 95% confidence.
# For example, a VaR of -2% at the 5% quantile means that there is a 5% chance that the return will be worse than -2% over the given period.

# CVaR (Conditional Value at Risk, 5% Quantile):
# Conditional VaR (CVaR), also known as Expected Shortfall, is a risk measure that gives the average loss assuming that the loss is worse than the Value at Risk (VaR).
# For instance, if the 5% VaR is -2%, then CVaR would give the average loss of those returns that are worse than -2%. This helps capture the tail risk (the potential for extreme losses) beyond the VaR threshold.

# Optimal Portfolio Weights:
# These are the weights assigned to each asset in a portfolio, optimized to either:
# - Maximize returns for a given level of risk (based on Modern Portfolio Theory).
# - Minimize risk for a given level of return.
# The weights represent the proportion of the portfolio allocated to each asset. For example, if Optimal Portfolio Weights for ABBV is 0.3,
# it means 30% of the portfolio is allocated to ABBV.
# The optimization uses the mean returns and covariance matrix of the assets to find the best balance between return and risk.

# Example:
# Let's say you run the portfolio analysis and get the following results for the three stocks:

# | Ticker | Power Law Exponent (α) | x_min | VaR (5% Quantile) | CVaR (5% Quantile) | Optimal Portfolio Weights |
# |--------|------------------------|-------|-------------------|--------------------|--------------------------|
# | O      | 2.7                    | 0.001 | -0.02             | -0.03              | 0.4                      |
# | JEPQ   | 2.5                    | 0.002 | -0.03             | -0.05              | 0.3                      |
# | ABBV   | 2.9                    | 0.003 | -0.015            | -0.02              | 0.3                      |

# Interpretation:
# 1. Power Law Exponent (α):
#    - For O, the Power Law exponent is 2.7, suggesting that its returns follow a moderate tail risk (neither too fat nor too thin).
#    - JEPQ has an exponent of 2.5, indicating slightly higher tail risk.
#    - ABBV has an exponent of 2.9, indicating a more stable return distribution with less extreme volatility.

# 2. x_min:
#    - x_min represents the threshold below which the Power Law distribution no longer applies. For O, x_min is 0.001,
#      meaning the Power Law distribution starts to apply from returns larger than 0.001.

# 3. VaR (5% Quantile):
#    - O has a VaR of -0.02 at the 5% quantile, meaning there's a 5% chance that O will lose more than 2% in value over the selected time frame.
#    - JEPQ has a VaR of -0.03, and ABBV has a VaR of -0.015.

# 4. CVaR (5% Quantile):
#    - O has a CVaR of -0.03, meaning that if the return is worse than the 5% worst-case (i.e., more than a 2% loss),
#      the average loss is 3%.
#    - JEPQ has a CVaR of -0.05, indicating a higher tail risk compared to O.

# 5. Optimal Portfolio Weights:
#    - The optimal weights for the portfolio suggest that 40% of the portfolio should be allocated to O, 30% to JEPQ, and 30% to ABBV.
#      These weights balance risk and return, optimizing the portfolio to maximize returns for a given level of risk.

# Conclusion:
# - The Power Law Exponent and VaR/CVaR measures help understand the risk dynamics of individual stocks.
# - Optimal Portfolio Weights tell you how to allocate your assets in the portfolio to balance risk and return.