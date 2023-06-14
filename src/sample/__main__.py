import numpy as np
import pandas as pd
import sys
import time

from vessel import Vessel
from global_env import GlobalEnv
from route import Route
from settlement import settle
from meanfield import mf
from fleet import Fleet

df_vessels = pd.read_excel("./data/CACIB-SAMPLE.xlsx")
vessels = [Vessel(row) for _, row in df_vessels.iterrows()]

# Initializing GlobalEnv object
env = GlobalEnv(
    ifo380_prices=np.array([433.1, 404.9, 390.4, 375.0]),
    vlsifo_prices=np.array([569.0, 534.9, 523.5, 506.9]),
    mgo_prices=np.array([825.8, 805.5, 782.6, 766.5]),
    lng_prices=np.array([609.9, 744.6, 697.9, 608.4]),
    carbon_tax_rates=np.array([0.0, 94.0 * 0.4, 94.0 * 0.7, 94.0 * 1.0]),
)

# Initializing Route objects
shg_rtm = Route(
    name="Shanghai-Rotterdam",
    route_type="CONTAINER SHIPS",
    distance=11999.0,
    freight_rates=np.array([1000.0, 1000.0, 1000.0, 1000.0]),
    utilization_rate=0.95,
    fuel_ratio=0.5,
)

hst_shg = Route(
    name="Houston-Shanghai",
    route_type="BULKERS",
    distance=12324.0,
    freight_rates=np.array([35.0, 35.0, 35.0, 35.0]),
    utilization_rate=0.9,
    fuel_ratio=0.5,
)


def main(regime):
    if regime == 0:
        return settle(
            i=1,
            data_vessels=df_vessels,
            env=env,
            route=shg_rtm,
            power=3.0,
            retrofit=False,
            year=0,
            pr=True,
        )
    elif regime == 1:
        return settle(
            i=5,
            data_vessels=df_vessels,
            env=env,
            route=hst_shg,
            power=2.0,
            retrofit=False,
            year=0,
            pr=True,
        )
    elif regime == 2:
        return mf(
            num=100,
            data_vessels=df_vessels,
            env=env,
            route=shg_rtm,
            value_exit=0.5,
            binary=False,
        )
    elif regime == 3:
        return settle(
            i=1,
            data_vessels=df_vessels,
            env=env,
            route=shg_rtm,
            power=3.0,
            retrofit=False,
            year=0,
            pr=False,
        ).optimization(
            retrofit=False, power=3.0, years=np.arange(4), cii_limit=True, pr=True
        )
    elif regime == 4:
        return settle(
            i=5,
            data_vessels=df_vessels,
            env=env,
            route=hst_shg,
            power=2.0,
            retrofit=False,
            year=0,
            pr=False,
        ).optimization(retrofit=False, power=2.0, years=np.arange(4), pr=True)
    elif regime == 5:
        return Fleet(
            vessels=[vessels[1], vessels[5]], routes=[shg_rtm, hst_shg], global_env=env
        ).global_optimization(retrofit=False, cii_limit=True, pr=True)
    else:
        return


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
