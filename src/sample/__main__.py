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


stm = Settlement(vessels[1], shg_rtm, market)
stm.fuel_cost_unit(pr=True)

# Create a virual sample of vessels with same information except CII score
vessels_virtual = [Vessel(df_vessels.iloc[1]) for i in range(100)]

ciis = []
for vessel in vessels_virtual:
    vessel.cii_score_2021 = vessel.cii_score_2021 * (1 + (np.random.rand() - 0.5))
    ciis.append(vessel.cii_score_2021)

# plt.hist(ciis)

# # Launch Model
mf = MeanField(vessels_virtual, shg_rtm, market, q=0.15)
errs, delta0, pis = mf.simulate(tol=0.01, max_iter=15)

fig, axs = plt.subplots(3)
axs[0].plot(errs)
axs[0].set_title("Proportion of vessels with y>theta")
axs[0].axline(xy1=(0, mf.q_), slope=0, c="red")

axs[1].plot(delta0)
axs[1].set_title("Speed variationof the first vessel's ")
axs[2].plot(pis)
axs[2].set_title("Profit of the first vessel")

plt.show()
fig.savefig("./fig/meanfield-100.png")
