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

# Launch Model
# Create a virual sample of vessels with same information

vessels_virtual = [Vessel(df_vessels.iloc[1]) for i in range(100)]

mf = MeanField(vessels_virtual, shg_rtm, market, q=0.15, value_exit=0.2)
mf.x_ = mf.x_ * (1 + 0.95 * 2 * (np.random.rand(len(mf.x_)) - 0.5))
# mf.x_ = mf.x_ * (1 + np.random.randn(len(mf.x_)))

# Simulate
errs, delta0, pis = mf.simulate(tol=0.01, max_iter=15)

fig, axs = plt.subplots(4, figsize=(8, 6))
plt.subplots_adjust(hspace=0.5)
axs[0].hist(mf.x_)
axs[0].set_title("Distribution of x")
axs[1].plot(errs)
axs[1].set_title("Proportion of vessels with y>theta")
axs[1].axline(xy1=(0, mf.q_), slope=0, c="red")

axs[2].plot(delta0)
axs[2].set_title("Speed variation of the first vessel's ")
axs[3].plot(pis)
axs[3].set_title("Profit of the first vessel")
fig.suptitle(
    f"{len(mf.x_):d} navires, q: {mf.q_:.2f}, exit value rate: {mf.value_exit_:.1f}"
)

plt.show()
fig.savefig(
    f"./fig/meanfield-{len(mf.x_):d} navires-exit value rate {mf.value_exit_:.1f}.png"
)
