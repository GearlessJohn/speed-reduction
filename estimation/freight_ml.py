import warnings

import numpy as np
import pandas as pd
import tqdm
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

# Pay attention to the difference between regressor and classifier!
warnings.filterwarnings("ignore")

data_eco = pd.read_excel("./data/projections Shipping Baseline.xlsx").dropna()
data_eco["Data"] = data_eco["Date"].astype("datetime64[ns]")
data_eco = data_eco.set_index("Data")

clarksons = pd.read_excel(
    "./data/Indices Clarksons.xlsx", skiprows=range(0, 5), sheet_name="Container"
)
clarksons["Date"] = clarksons["Date"].astype("datetime64[ns]")
clarksons = clarksons.set_index("Date")
clarksons = clarksons.resample("Q").mean()

data = pd.merge(clarksons, data_eco, left_index=True, right_index=True, how="left")
data.rename(columns={"$/day": "Clarksons"}, inplace=True)
y = data.pop("Clarksons")
X = data.drop(["Date"], axis=1)
X = X.drop(X.filter(regex='Bulker|Tanker').columns, axis=1)


# print(X.columns)

# standardise the columns of numeric values
def standardise(data, list_column):
    for name_column in list_column:
        # standardize
        if data[name_column].std() != 0:
            data[name_column] = (data[name_column] - data[name_column].mean()) / data[name_column].std()
        else:
            data[name_column] = (data[name_column] - data[name_column].mean())


standardise(X, X.columns.to_list())


def MAE(pred, real):
    N = len(pred)
    sum = 0
    for i in range(N):
        sum += abs(pred[i] - real[i]) / real[i]
    return sum / N


def model_result(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    loss = MAE(y_pred, y_test)
    return loss


alpha_list = [i + 1 for i in range(0, 5)]
loss_tot = []
# model = MLPRegressor(random_state=1, max_iter=200, hidden_layer_sizes=(300, 150, 10))
model = KNeighborsRegressor(n_neighbors=3)
for alpha in tqdm.tqdm(alpha_list):
    X_train, X_test, y_train, y_test = train_test_split(X.index, y, test_size=0.3)
    X_train = X.loc[X_train]
    X_test = X.loc[X_test]
    # model = Lasso(alpha=alpha)
    # model = RandomForestRegressor(max_depth=alpha)
    loss_model = model_result(model, X_train.values, y_train.values, X_test.values, y_test.values)
    loss_tot.append(loss_model)

print(f"Mean MAE score {100*np.mean(loss_tot):2.1f}%")

fig = plt.figure()
ax = plt.axes()
ax.scatter(alpha_list, loss_tot)
ax.plot(alpha_list, loss_tot)
plt.title("Loss")
plt.show()
