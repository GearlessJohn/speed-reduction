import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.graphics as smgraphics
from statsmodels.tsa.seasonal import seasonal_decompose

clarksons = pd.read_excel(
    "./data/Indices Clarksons.xlsx", skiprows=range(0, 5), sheet_name="Container"
)
clarksons["Date"] = clarksons["Date"].astype("datetime64[ns]")
# plot_clark = clarksons.plot(x="Date").get_figure()
# plot_clark.savefig("./fig/clarksons.png")

exports = pd.read_csv("./data/ChinaExports.csv")
exports["DATE"] = exports["DATE"].astype("datetime64[ns]")
# exports["DATE"] = exports["DATE"] + pd.offsets.DateOffset(months=5)
# plot_exports = exports.plot(x="DATE").get_figure()
# plot_exports.savefig("./fig/exports.png")

data = pd.merge(clarksons, exports, left_on="Date", right_on="DATE")
data.drop(columns=["DATE"], inplace=True)
data.rename(columns={"$/day": "Clarksons", "XTEXVA01CNM667N": "Exports"}, inplace=True)

result = seasonal_decompose(data["Clarksons"], model="additive", period=12)
trend, seasonal, resid = result.trend, result.seasonal, result.resid
data["Clarksons"] = pd.Series(trend)


result = seasonal_decompose(data["Exports"], period=12)
trend, seasonal, resid = result.trend, result.seasonal, result.resid
data["Exports"] = pd.Series(trend)

data["Exports"] = data["Exports"].pct_change() * 100
data["Clarksons"] = data["Clarksons"].pct_change() * 100

data = data[(data["Date"] >= "2019-01-01") & (data["Date"] <= "2022-07-01")]
# print(data.info)

# data = data[np.abs(data["Clarksons"])<5]
# data = data[np.abs(data["Exports"])<20]

x = data["Exports"]
# X = sm.add_constant(x)
X = x
y = data["Clarksons"]

model = sm.RLM(y, X)
results = model.fit()

print(results.summary())

data.plot(x="Date", y=["Exports", "Clarksons"])
# plt.savefig("./data/clarksons-exports.png")

fig, ax = plt.subplots()
ax.set_xlabel("Exports")
ax.set_ylabel("Clarksons")
ax.scatter(x, y, label="original data")
ax.plot(x, results.fittedvalues, label="predictions", color="r")
plt.legend()
plt.show()
fig.savefig("./fig/predict_exports.png")
