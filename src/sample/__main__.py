import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing custom classes
from vessel import Vessel
from global_market import GlobalMarket
from route import Route
from settlement import Settlement
from meanfield import MeanField

# Reading an Excel file using Pandas
df_vessels = pd.read_excel("./data/CACIB-SAMPLE.xlsx")

# Creating a list of Vessel objects
vessels = [Vessel(row) for _, row in df_vessels.iterrows()]

# Initializing GlobalMarket object
market = GlobalMarket(ifo380_price=494.0, vlsifo_price=631.5, carbon_tax_rates=0.0)

# Initializing Route object
shg_rtm = Route(
    name="Shanghai-Rotterdam",
    route_type="CONTAINER SHIPS",
    distance=11999,
    utilization_rate=0.95,
    freight_rate=1479.0,
)


# Create a virual sample of vessels with same information except CII score
vessels_virtual = [Vessel(df_vessels.iloc[1]) for i in range(1000)]

ciis = []
for vessel in vessels_virtual:
    vessel.cii_score_2021 = vessel.cii_score_2021 * (1 + (np.random.rand() - 0.5))
    ciis.append(vessel.cii_score_2021)

# plt.hist(ciis)
# plt.show()

# Launch Model
mf = MeanField(vessels_virtual, shg_rtm, market, q=0.15)
print("theta:\t", mf.theta_)
y = mf.x_ + mf.lam_ * mf.delta_
plt.plot(mf.simulate(tol=0.001, max_iter=100))
print(np.sum(y > mf.theta_ + 1e-3))

# plt.show()
