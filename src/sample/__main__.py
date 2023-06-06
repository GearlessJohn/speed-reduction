import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from vessel import Vessel
from global_env import GlobalEnv
from route import Route
from settlement import settle
from meanfield import mf

df_vessels = pd.read_excel("./data/CACIB-SAMPLE.xlsx")

# Initializing GlobalEnv object
env = GlobalEnv(
    ifo380_price=494.0,
    vlsifo_price=594,
    mgo_price=781,
    lng_price=1500,
    carbon_tax_rates=0.0,
)

# Initializing Route objects
shg_rtm = Route(
    name="Shanghai-Rotterdam",
    route_type="CONTAINER SHIPS",
    distance=11999.0,
    freight_rate=1479.0,
    utilization_rate=0.95,
    fuel_ratio=0.46,
)

hst_shg = Route(
    name="Houston-Shanghai",
    route_type="BULKERS",
    distance=12324.0,
    freight_rate=35.0,
    utilization_rate=0.9,
    fuel_ratio=0.5,
)

s1 = settle(
    i=1,
    data_vessels=df_vessels,
    env=env,
    route=shg_rtm,
    power=3.0,
    retrofit=False,
    pr=True,
)


# s2 = settle(
#     i=5,
#     data_vessels=df_vessels,
#     env=env,
#     route=hst_shg,
#     power=2.0,
#     retrofit=False,
#     pr=True,
# )

# mf(
#     num=100,
#     data_vessels=df_vessels,
#     env=env,
#     route=shg_rtm,
#     value_exit=0.5,
#     binary=False,
# )
