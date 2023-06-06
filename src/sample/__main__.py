import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from vessel import Vessel
from global_env import GlobalEnv
from route import Route
from settlement import settle
from meanfield import mf


# settle(
#     i=1,
#     ifo380_price=494.0,
#     vlsifo_price=631.5,
#     # vlsifo_price=494.0,
#     carbon_tax_rates=0,
#     name="Shanghai-Rotterdam",
#     route_type="CONTAINER SHIPS",
#     distance=11999.0,
#     freight_rate=1479.0,
#     utilization_rate=0.95,
#     fuel_ratio=0.5,
#     retrofit=False,
#     pr=True,
# )


# s1 = settle(
#     i=5,
#     ifo380_price=494.0,
#     vlsifo_price=494.0,
#     carbon_tax_rates=0,
#     name="Houston-Shanghai",
#     route_type="BULKERS",
#     distance=12324.0,
#     freight_rate=35.0,
#     utilization_rate=0.9,
#     retrofit=False,
#     fuel_ratio=0.5,
# )

mf(num=100, value_exit=0.5, binary=False)
