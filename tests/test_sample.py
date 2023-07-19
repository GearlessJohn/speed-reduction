import numpy as np
import pandas as pd

import sample.fleet
import sample.globalenv
import sample.meanfield
import sample.route
import sample.settlement
import sample.vessel

df_vessels = pd.read_excel("./data/CACIB-SAMPLE.xlsx")
vessels = [sample.vessel.Vessel(row) for _, row in df_vessels.iterrows()]

# Initializing GlobalEnv object
global_market = sample.globalenv.GlobalEnv(
    ifo380_prices=np.array([433.1, 404.9, 390.4, 375.0]),
    vlsifo_prices=np.array([569.0, 534.9, 523.5, 506.9]),
    mgo_prices=np.array([825.8, 805.5, 782.6, 766.5]),
    lng_prices=np.array([609.9, 744.6, 697.9, 608.4]),
    carbon_tax_rates=np.array([0.0, 94.0 * 0.4, 94.0 * 0.7, 94.0 * 1.0]),
    # carbon_tax_rates=np.zeros(4),
)

# Initializing Route objects
rt_container_0 = sample.route.Route(
    name="Shanghai-Rotterdam",
    route_type="CONTAINER SHIPS",
    distance=11999.0,
    freight_rates=np.array([1000.0, 1000.0, 1000.0, 1000.0]),
    utilization_rate=0.95,
    fuel_ratio=0.5,
)

rt_bulker_0 = sample.route.Route(
    name="Houston-Shanghai",
    route_type="BULKERS",
    distance=12324.0,
    freight_rates=np.array([35.0, 35.0, 35.0, 35.0]),
    utilization_rate=0.9,
    fuel_ratio=0.5,
)


def main(regime: object) -> object:
    if regime == 0:
        return sample.settlement.settle(
            i=1,
            data_vessels=df_vessels,
            env=global_market,
            route=rt_container_0,
            power=3.0,
            retrofit=False,
            acc=True,
            year=0,
            pr=True,
            plot=False
        )
    elif regime == 1:
        return sample.settlement.settle(
            i=5,
            data_vessels=df_vessels,
            env=global_market,
            route=rt_bulker_0,
            power=2.0,
            retrofit=False,
            acc=True,
            year=0,
            pr=True,
            plot=False,
        )
    elif regime == 2:
        return sample.meanfield.mf(
            num=100,
            data_vessels=df_vessels,
            env=global_market,
            route=rt_bulker_0,
            value_exit=0.5,
            q=0.15,
            binary=False,
        )
    elif regime == 3:
        return sample.settlement.settle(
            i=1,
            data_vessels=df_vessels,
            env=global_market,
            route=rt_container_0,
            power=3.0,
            retrofit=False,
            year=0,
            pr=False,
            plot=False
        )[0].optimization(
            retrofit=False,
            power=3.0,
            years=np.arange(4),
            cii_limit=True,
            acc=True,
            pr=True,
            plot=False
        )
    elif regime == 4:
        return sample.settlement.settle(
            i=5,
            data_vessels=df_vessels,
            env=global_market,
            route=rt_bulker_0,
            power=2.0,
            retrofit=False,
            year=0,
            acc=True,
            pr=False,
        )[0].optimization(
            retrofit=False,
            power=2.0,
            years=np.arange(4),
            cii_limit=True,
            acc=True,
            pr=True,
        )
    elif regime == 5:
        return sample.fleet.Fleet(
            vessels=[vessels[1], vessels[5], vessels[0], vessels[7]],
            routes=[rt_container_0, rt_bulker_0, rt_bulker_0, rt_bulker_0],
            global_env=global_market,
        ).global_optimization(
            retrofit=False, cii_limit=True, construction=True, pr=True, plot=False
        )
    elif regime == 6:
        flt = sample.fleet.Fleet(
            vessels=[vessels[1], vessels[5], vessels[0], vessels[7]],
            routes=[rt_container_0, rt_bulker_0, rt_bulker_0, rt_bulker_0],
            global_env=global_market,
        )
        flt.mean_field(max_iter=30, elas=1.3807, tol=1e-3, plot=False)
        return flt
    else:
        return


def test_step_1():
    assert (main(0)[1] - 15.29999) < 1e-4


def test_step_2():
    assert np.mean(np.abs((main(3)[0] - np.array([15.30312018, 14.54312018, 14.03312018, 13.59312018])))) < 1e-4


def test_step_3():
    assert np.mean(np.abs((main(5) - np.array([[39939010.610, 38546933.194, 39239818.697, 39597973.992],
                                               [20739001.434, 19538076.097, 16947019.644, 14618713.554],
                                               [508964.457, 3955240.708, 4958535.028, 4799871.136],
                                               [-842785.239, 441263.483, 579224.500, 434726.485]])))) < 1e-3


def test_step_4():
    assert np.mean(np.abs((main(6).routes[1].freight_rates - np.array([35.721, 36.753, 36.181, 36.629])))) < 1e-3
