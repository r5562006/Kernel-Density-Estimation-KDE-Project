import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# 生成隨機數據
data = np.random.randn(1000)

# 應用核密度估計
kde = gaussian_kde(data)
x = np.linspace(min(data), max(data), 1000)
kde_values = kde(x)

# 可視化結果
plt.plot(x, kde_values, label='KDE')
plt.hist(data, bins=30, density=True, alpha=0.5, label='Histogram')
plt.legend()
plt.show()