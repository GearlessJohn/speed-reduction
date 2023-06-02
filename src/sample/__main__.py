import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from vessel import Vessel
from global_env import GlobalEnv
from route import Route
from settlement import settle
from meanfield import simulation

settle(
    i=1,
    ifo380_price=494.0,
    vlsifo_price=631.5,
    carbon_tax_rates=94.0,
    name="Shanghai-Rotterdam",
    route_type="CONTAINER SHIPS",
    distance=11999.0,
    freight_rate=1479.0,
    utilization_rate=0.95,
    retrofit=False,
)
# simulation()
