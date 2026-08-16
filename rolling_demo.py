import pandas as pd
import numpy as np

# Simulate one patient's HR readings over 10 hours
hr = pd.Series([80, 82, 85, 90, 95, 98, 102, 105, 110, 115])

rolling_mean = hr.rolling(window=6, min_periods=1).mean()
rolling_std = hr.rolling(window=6, min_periods=1).std()

print("HR:", hr.tolist())
print("Rolling mean:", rolling_mean.round(1).tolist())
print("Rolling std:", rolling_std.round(1).tolist())


def get_slope(values):
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values))  # [0, 1, 2, ...] representing hour position
    slope = np.polyfit(x, values, 1)[0]  # fit a line, take its slope
    return slope


rolling_slope = hr.rolling(window=6, min_periods=1).apply(get_slope, raw=True)
print("Rolling slope:", rolling_slope.round(2).tolist())