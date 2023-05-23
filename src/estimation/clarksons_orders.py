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

orders = pd.read_csv("./data/NewOrders.csv")
orders["DATE"] = orders["DATE"].astype("datetime64[ns]")
orders["DATE"] = orders["DATE"] + pd.offsets.DateOffset(years=2)
# plot_orders = orders.plot(x="DATE").get_figure()
# plot_orders.savefig("./fig/orders.png")

data = pd.merge(clarksons, orders, left_on="Date", right_on="DATE")
data.drop(columns=["DATE"], inplace=True)
data.rename(columns={"$/day": "Clarksons"}, inplace=True)

result = seasonal_decompose(data["Clarksons"], period=12)
trend, seasonal, resid = result.trend, result.seasonal, result.resid
data["Clarksons"] = pd.Series(trend)

result = seasonal_decompose(data["orders"], period=12)
trend, seasonal, resid = result.trend, result.seasonal, result.resid
data["orders"] = pd.Series(trend)

data["orders"] = data["orders"].pct_change(fill_method="ffill") * 100
data["Clarksons"] = data["Clarksons"].pct_change(fill_method="ffill") * 100
data = data[(data["Date"] >= "2020-01-01") & (data["Date"] <= "2023-01-01")]
# print(data.info)

# data = data[np.abs(data["Clarksons"])<5]
# data = data[np.abs(data["orders"])<20]

x = data["orders"]
X = sm.add_constant(x)
y = data["Clarksons"]

model = sm.OLS(y, X)
results = model.fit()

print(results.summary())


fig, ax = plt.subplots()
ax.set_xlabel("orders")
ax.set_ylabel("Clarksons")
ax.scatter(x, y, label="original data")
ax.plot(x, results.fittedvalues, label="predictions", color="r")
plt.legend()
plt.show()
# fig.savefig("./fig/predict_orders.png")
