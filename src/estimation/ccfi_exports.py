import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.graphics as smgraphics
from statsmodels.tsa.seasonal import seasonal_decompose

ccfi = pd.read_excel("./data/CCFI.xlsx").dropna()
ccfi["Date"] = ccfi["Date"].astype("datetime64[ns]") + pd.Timedelta(days=1)

exports = pd.read_csv("./data/ChinaExports.csv")
exports["DATE"] = exports["DATE"].astype("datetime64[ns]")


data = pd.merge(ccfi, exports, left_on="Date", right_on="DATE")
data.drop(columns=["DATE"], inplace=True)
data.rename(columns={"SHSPCCFI Index": "ccfi", "XTEXVA01CNM667N": "Exports"}, inplace=True)

# result = seasonal_decompose(data["ccfi"], model="additive", period=12)
# trend, seasonal, resid = result.trend, result.seasonal, result.resid
# data["ccfi"] = pd.Series(trend)


# result = seasonal_decompose(data["Exports"], period=12)
# trend, seasonal, resid = result.trend, result.seasonal, result.resid
# data["Exports"] = pd.Series(trend)

data = data.groupby(data.Date.dt.year).mean()


data["Exports"] = data["Exports"].pct_change() * 100
data["ccfi"] = data["ccfi"].pct_change() * 100
data = data.dropna()

print(data)

data = data[data["ccfi"] < 120]

x = data["Exports"]
# X = sm.add_constant(x)
X = x
y = data["ccfi"]

model = sm.OLS(y, X)
results = model.fit()

print(results.summary())

fig, ax = plt.subplots()
ax.set_xlabel("Exports")
ax.set_ylabel("ccfi")
ax.scatter(x, y, label="original data")
ax.plot(x, results.fittedvalues, label="predictions", color="r")
plt.legend()
plt.show()
# fig.savefig("./fig/_exports.png")
