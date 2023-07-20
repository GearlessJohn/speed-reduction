import numpy as np

import sample.globalenv
import sample.route
import sample.settlement
import sample.vessel

vessel = sample.vessel.Vessel(name="Vessel Test", vessel_type="CONTAINER SHIPS", sub_type="PANAMAX / POST PANAMAX",
                              dwt=100000, capacity=9000, unit="TEU", built=2015, distance_2021=80000, hours_2021=5000,
                              hfo_2021=15000, lfo_2021=2000, lng_2021=0, diesel_2021=2000, co2_2021=65000,
                              cii_score_2021=7,
                              cii_class_2021="C")

global_market = sample.globalenv.GlobalEnv(
    ifo380_prices=np.array([433.1, 404.9, 390.4, 375.0]),
    vlsifo_prices=np.array([569.0, 534.9, 523.5, 506.9]),
    mgo_prices=np.array([825.8, 805.5, 782.6, 766.5]),
    lng_prices=np.array([609.9, 744.6, 697.9, 608.4]),
    carbon_tax_rates=np.array([0.0, 94.0 * 0.4, 94.0 * 0.7, 94.0 * 1.0]),
)

rt_container_0 = sample.route.Route(
    name="Shanghai-Rotterdam",
    route_type="CONTAINER SHIPS",
    distance=11999.0,
    freight_rates=np.array([1000.0, 1000.0, 1000.0, 1000.0]),
    utilization_rate=0.95,
    fuel_ratio=0.5,
)

stm = sample.settlement.Settlement(vessel, rt_container_0, global_market)


def test_cii_scheme():
    assert stm.cii_class(speed=vessel.speed_2021, power=3, year=0) == vessel.cii_class_2021


def test_correct_voyage_hours():
    hours = [stm.voyage_hours(v, acc=True) for v in [0, 10, 1000]]
    assert min(hours) > 0 and max(hours) < 365 * 24


def test_signs_of_financial_results():
    speed = vessel.speed_2021
    fc = stm.fuel_cost(speed, saving=0, power=3, year=0)
    retrofit = stm.retrofitting(speed, power=3, year=0)[0]
    carbon_tax = stm.carbon_tax(speed, saving=0, power=3, year=0)
    op = stm.operation_cost()
    freight = stm.freight(year=0)
    assert min([fc, retrofit, carbon_tax, op, freight]) >= 0


def test_cii_profit_reverse_optimization():
    profits = np.array([[2.5, 3, 3.2], [1.2, 1.5, 1.6], [1.1, 1.5, 1.3], [1, 0.9, 0.8]])
    cii_class = np.array(
        [["C", "D", "E"], ["C", "D", "E"], ["C", "D", "E"], ["C", "D", "E"]]
    )
    res, total_profit = stm.cii_profits_reverse(profits, cii_class)
    assert np.array_equal(res, [1, 0, 1, 0]) and total_profit == 6.7
