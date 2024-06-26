import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error

# Set seed for reproducibility
np.random.seed(42)

# Define MA parameters
ma_params = {
    'MA(1)': [1, 0.5],
    'MA(2)': [1, 0.5, 0.3],
    'MA(3)': [1, 0.5, 0.3, 0.2]
}

# Generate and plot ACF and PACF for each MA process
fig, axs = plt.subplots(3, 2, figsize=(14, 18))

for i, (label, params) in enumerate(ma_params.items()):
    ma_process = ArmaProcess(np.array([1]), np.array(params))
    ts = ma_process.generate_sample(nsample=500)
    
    # Evaluation metrics
    mean_val = np.mean(ts)
    var_val = np.var(ts)
    rmse_val = np.sqrt(mean_squared_error(np.zeros_like(ts), ts))
    
    # ACF and PACF plots
    plot_acf(ts, ax=axs[i, 0], lags=20)
    plot_pacf(ts, ax=axs[i, 1], lags=20)
    axs[i, 0].set_title(f'ACF of {label}')
    axs[i, 1].set_title(f'PACF of {label}')
    
    # Print evaluation metrics
    print(f'{label} - Mean: {mean_val:.4f}, Variance: {var_val:.4f}, RMSE: {rmse_val:.4f}')

plt.tight_layout()
plt.show()
