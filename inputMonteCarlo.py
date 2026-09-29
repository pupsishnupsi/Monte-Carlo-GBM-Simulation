import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# Fetch latest TSM price
tick_input = input("Enter Yahoo Ticker Symbol: ").strip().upper()
ticker = yf.Ticker(tick_input)
ticker_year = ticker.history(period="1y")["Close"]
latest = ticker.history(period="1d")
latest_close = latest['Close'].iloc[-1]

# Mu
beta = ticker.info.get("beta")
r_f = yf.Ticker("^TNX").history(period="1y")["Close"].iloc[-1] / 100
sp500 = yf.Ticker("^GSPC").history(period="1y")["Close"]
r_m = (sp500.iloc[-1] / sp500.iloc[0]) -1
log_tick = np.log(tick_year / tick_year.shift(1)).dropna()

# Simulation parameters
S0 = latest_close        # starting stock price
mu = r_f + beta * (r_m - r_f)     # expected annual return
sigma = log_tick.std() * np.sqrt(252)    # annual volatility
T = 1           # time in years
N = 252         # number of steps
dt = T / N      # time step size
M = 1000       # number of simulation paths

# Set up matrix for all paths
price_paths = np.zeros((M, N+1))
price_paths[:, 0] = S0   # initialize all paths at S0

# Generate all random shocks (Z) at once
Z = np.random.normal(0, 1, size=(M, N))

# Simulate paths
for t in range(1, N+1):
    price_paths[:, t] = price_paths[:, t-1] * np.exp(
        (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t-1]
    )

final_prices = price_paths[:, -1]

# Create a figure with two subplots side by side
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# Plot sample price paths in the first subplot
for i in range(1000):
    axs[0].plot(price_paths[i])
axs[0].set_title(f"Monte Carlo {tick} Price Paths")
axs[0].set_xlabel("Trading Days")
axs[0].set_ylabel("Price")
axs[0].grid(True)

# Plot histogram of final prices in the second subplot
axs[1].hist(final_prices, bins=100, edgecolor="black")  
axs[1].set_title(f"Distribution of Final {tick} Prices (1 Year)")
axs[1].set_xlabel("Final Price")
axs[1].set_ylabel("Frequency")
axs[1].grid(True)

mean_price = np.mean(final_prices)
var_5 = np.percentile(final_prices, 5)
prob_loss = np.mean(final_prices < S0)

# Add text box with metrics
risk_text = (
    f"Mean Final Price: ${mean_price:.2f}\n"
    f"5% VaR: ${var_5:.2f} (loss: {S0-var_5:.2f})\n"
    f"Probability of Loss: {prob_loss:.2%}"
)

# Place box in upper right
plt.gca().text(
    0.95, 0.95, risk_text,
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment='top',
    horizontalalignment='right',
    bbox=dict(boxstyle="round,pad=0.4", facecolor='white', edgecolor='gray')
)

# Adjust layout and show both plots together
plt.tight_layout()
plt.show()
