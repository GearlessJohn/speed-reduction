import sys

import numpy as np
import pandas as pd


import fleet
import globalenv
import meanfield
import route
import settlement
import vessel

df_vessels = pd.read_excel("./data/CACIB-SAMPLE.xlsx")
vessels = [vessel.Vessel(row) for _, row in df_vessels.iterrows()]

# Initializing GlobalEnv object
global_market = globalenv.GlobalEnv(
    ifo380_prices=np.array([433.1, 404.9, 390.4, 375.0]),
    vlsifo_prices=np.array([569.0, 534.9, 523.5, 506.9]),
    mgo_prices=np.array([825.8, 805.5, 782.6, 766.5]),
    lng_prices=np.array([609.9, 744.6, 697.9, 608.4]),
    carbon_tax_rates=np.array([0.0, 94.0 * 0.4, 94.0 * 0.7, 94.0 * 1.0]),
    # carbon_tax_rates=np.zeros(4),
)

# Initializing Route objects
rt_container_0 = route.Route(
    name="Shanghai-Rotterdam",
    route_type="CONTAINER SHIPS",
    distance=11999.0,
    freight_rates=np.array([1000.0, 1000.0, 1000.0, 1000.0]),
    utilization_rate=0.95,
    fuel_ratio=0.5,
)

rt_bulker_0 = route.Route(
    name="Houston-Shanghai",
    route_type="BULKERS",
    distance=12324.0,
    freight_rates=np.array([35.0, 35.0, 35.0, 35.0]),
    utilization_rate=0.9,
    fuel_ratio=0.5,
)


def main(regime: object) -> object:
    if regime == 0:
        return settlement.settle(
            i=1,
            data_vessels=df_vessels,
            env=global_market,
            route=rt_container_0,
            power=3.0,
            retrofit=False,
            acc=True,
            year=0,
            pr=True,
        )
    elif regime == 1:
        return settlement.settle(
            i=5,
            data_vessels=df_vessels,
            env=global_market,
            route=rt_bulker_0,
            power=2.0,
            retrofit=False,
            acc=True,
            year=0,
            pr=True,
        )
    elif regime == 2:
        return meanfield.mf(
            num=100,
            data_vessels=df_vessels,
            env=global_market,
            route=rt_bulker_0,
            value_exit=0.5,
            q=0.15,
            binary=False,
        )
    elif regime == 3:
        return settlement.settle(
            i=1,
            data_vessels=df_vessels,
            env=global_market,
            route=rt_container_0,
            power=3.0,
            retrofit=False,
            year=0,
            pr=False,
        ).optimization(
            retrofit=False,
            power=3.0,
            years=np.arange(4),
            cii_limit=True,
            acc=True,
            pr=True,
        )
    elif regime == 4:
        return settlement.settle(
            i=5,
            data_vessels=df_vessels,
            env=global_market,
            route=rt_bulker_0,
            power=2.0,
            retrofit=False,
            year=0,
            acc=True,
            pr=False,
        ).optimization(
            retrofit=False,
            power=2.0,
            years=np.arange(4),
            cii_limit=True,
            acc=True,
            pr=True,
        )
    elif regime == 5:
        return fleet.Fleet(
            vessels=[vessels[1], vessels[5], vessels[0], vessels[7]],
            routes=[rt_container_0, rt_bulker_0, rt_bulker_0, rt_bulker_0],
            global_env=global_market,
        ).global_optimization(
            retrofit=False, cii_limit=True, construction=True, pr=True
        )
    elif regime == 6:
        flt = fleet.Fleet(
            vessels=[vessels[1], vessels[5], vessels[0], vessels[7]],
            routes=[rt_container_0, rt_bulker_0, rt_bulker_0, rt_bulker_0],
            global_env=global_market,
        )
        flt.mean_field(max_iter=30, elas=1.3807, tol=1e-3)
        return
    else:
        return


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
