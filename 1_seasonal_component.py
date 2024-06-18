import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.seasonal import seasonal_decompose

stock_symbol = "AAPL"  
start_date = "2020-01-01"
end_date = "2021-01-01"
ts_data = yf.download(stock_symbol, start=start_date, end=end_date)

# Plot opening prices
plt.figure(figsize=(10, 6))
plt.plot(ts_data.index, ts_data['Open'])
plt.ylabel("Price")
plt.xlabel("Date")
plt.title("Opening Price of the Stocks")
plt.xticks(rotation=45)
plt.xlim(ts_data.index.min(), ts_data.index.max())
plt.show()

# Decompose and plot
decompose_result = seasonal_decompose(ts_data['Open'], model='multiplicative', period=30)
decompose_result.plot()
plt.show()
