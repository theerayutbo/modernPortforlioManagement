import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# 1. ตั้งค่า Tickers และช่วงเวลา
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'O']  
start_date = '2015-01-01'
end_date   = '2023-01-01'

# 2. ดึงข้อมูล Adjusted Close ด้วย auto_adjust=False แล้ว slice 'Adj Close'
data = yf.download(
    tickers,
    start=start_date,
    end=end_date,
    auto_adjust=False
)['Adj Close']

# 3. คำนวณ daily returns และ drop missing
returns = data.pct_change().dropna()

# 4. เตรียมค่าเฉลี่ยและ covariance matrix (daily)
mean_returns = returns.mean().values        # shape (5,)
cov_matrix   = returns.cov().values         # shape (5,5)

# 5. Monte Carlo Simulation parameters
num_simulations = 10000
num_assets      = len(tickers)

# pre-allocate
simulated_weights            = np.zeros((num_simulations, num_assets))
simulated_portfolio_returns  = np.zeros(num_simulations)
simulated_portfolio_volatility = np.zeros(num_simulations)

# 6. รัน simulation
for i in range(num_simulations):
    # สุ่ม weights แล้ว normalize
    w = np.random.random(num_assets)
    w /= w.sum()
    simulated_weights[i, :] = w

    # annualized return: dot(weights, mean_daily_returns) * 252
    simulated_portfolio_returns[i] = np.dot(w, mean_returns) * 252

    # annualized volatility: sqrt(wᵀ @ (cov_matrix * 252) @ w)
    simulated_portfolio_volatility[i] = np.sqrt(w @ (cov_matrix * 252) @ w)

# 7. หา Top 5 แต่ละกลยุทธ์
sharpe_ratios        = simulated_portfolio_returns / simulated_portfolio_volatility
top5_sharpe_idx      = np.argsort(sharpe_ratios)[-5:]
top5_minvol_idx      = np.argsort(simulated_portfolio_volatility)[:5]
top5_maxreturn_idx   = np.argsort(simulated_portfolio_returns)[-5:]

# 8. สร้าง DataFrame ผลลัพธ์
def build_results(idxs):
    df = pd.DataFrame({
        'Simulation': idxs + 1,
        'Return': simulated_portfolio_returns[idxs],
        'Volatility': simulated_portfolio_volatility[idxs],
        'Sharpe Ratio': (simulated_portfolio_returns[idxs] /
                         simulated_portfolio_volatility[idxs])
    })
    for j, ticker in enumerate(tickers):
        df[ticker] = simulated_weights[idxs, j]
    return df

max_sharpe_results   = build_results(top5_sharpe_idx)
min_vol_results      = build_results(top5_minvol_idx)
max_return_results   = build_results(top5_maxreturn_idx)

# 9. แสดงผลลัพธ์
print("=== Top 5 Maximum Sharpe Ratio Portfolios ===")
print(max_sharpe_results.to_string(index=False))
print("\n=== Top 5 Minimum Volatility Portfolios ===")
print(min_vol_results.to_string(index=False))
print("\n=== Top 5 Maximum Return Portfolios ===")
print(max_return_results.to_string(index=False))

# 10. Plot Efficient Frontier และไฮไลต์พอร์ตสำคัญ
plt.figure(figsize=(10, 6))
sc = plt.scatter(
    simulated_portfolio_volatility,
    simulated_portfolio_returns,
    c=sharpe_ratios,
    cmap='viridis',
    s=10,
    alpha=0.5
)
plt.colorbar(sc, label='Sharpe Ratio')
plt.xlabel('Annualized Volatility')
plt.ylabel('Annualized Return')
plt.title('Efficient Frontier')

# ไฮไลต์พอร์ตโฟลิโอสำคัญ
# Max Sharpe
i = top5_sharpe_idx[-1]
plt.scatter(
    simulated_portfolio_volatility[i],
    simulated_portfolio_returns[i],
    marker='*', s=200, color='green',
    label='Max Sharpe'
)
# Min Volatility
i = top5_minvol_idx[0]
plt.scatter(
    simulated_portfolio_volatility[i],
    simulated_portfolio_returns[i],
    marker='v', s=200, color='red',
    label='Min Volatility'
)
# Max Return
i = top5_maxreturn_idx[-1]
plt.scatter(
    simulated_portfolio_volatility[i],
    simulated_portfolio_returns[i],
    marker='s', s=200, color='orange',
    label='Max Return'
)

# Capital Market Line (จาก risk-free=0)
# CML ผ่านจุด origin ถึง max Sharpe
x_cml = [0, simulated_portfolio_volatility[top5_sharpe_idx[-1]]]
y_cml = [0, simulated_portfolio_returns[top5_sharpe_idx[-1]]]
plt.plot(x_cml, y_cml, linestyle='--', linewidth=2, label='Capital Market Line')

plt.legend(loc='best')
plt.show()
