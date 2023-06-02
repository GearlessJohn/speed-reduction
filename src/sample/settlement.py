import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from vessel import Vessel
from global_env import GlobalEnv
from route import Route


class Settlement:
    def __init__(self, vessel, route, global_env, utilization_rate=0.95):
        self.vessel = vessel
        self.route = route
        self.global_env = global_env
        self.utilization_rate = utilization_rate

    def cost_fuel(self, speed=None, saving=0.0):
        speed = speed or self.vessel.speed_2021
        # Calculate the expenditure on fuel
        return -(
            self.vessel.fuel_consumption_rate(speed)
            * self.route.distance
            * self.global_env.fuel_price(self.vessel.main_engine_fuel_type)
        ) * (1 - saving)

    def cost_retrofit(self, speed=None):
        speed = speed or self.vessel.speed_2021
        fuel_cost = self.cost_fuel(speed=speed)
        # Consider now the possible retrofitting measures
        years_left = 25 - self.vessel.age
        engine = 10000.0 / years_left  # 2.5%
        propeller = 450000.0 / years_left  # 7.0%
        bulbous = 550000.0 / years_left  # 4.0%

        cost_retrofit = 0.0
        saving = 0.0
        if -fuel_cost * 0.025 > engine:
            cost_retrofit += -engine
            saving += 0.025
        if -fuel_cost * 0.07 > propeller:
            cost_retrofit += -propeller
            saving += 0.07
        if -fuel_cost * 0.04 > bulbous:
            cost_retrofit += -bulbous
            saving += 0.04
        return cost_retrofit, saving

    def cost_fuel_unit(self, speed=None, pr=False):
        speed = speed or self.vessel.speed_2021
        cost = self.cost_fuel(speed) / (self.vessel.capacity * self.utilization_rate)
        if pr:
            print(
                f"Fuel cost of {self.vessel.name} for route {self.route.name}: {cost:.1f} dollars/{self.vessel.unit}"
            )
        return cost

    def cii_class(self, speed=None, year=2023):
        speed = speed or self.vessel.speed_2021
        cla = self.global_env.cii_class(
            self.vessel.cii_score_2021,
            self.vessel.vessel_type,
            self.vessel.sub_type,
            self.vessel.dwt,
            year=year,
        )
        return cla

    def ghg_operation(self, speed=None, saving=0.0):
        speed = speed or self.vessel.speed_2021
        return self.vessel.co2_emission(speed) * (1 - saving)

    def ghg_construction(self, ratio=None):
        return self.vessel.co2_emission(self.vessel.speed_2021) * ratio

    def cost_carbon_tax(self, speed=None, saving=0.0):
        speed = speed or self.vessel.speed_2021
        # Assess the financial implications of carbon emissions
        return -(
            self.ghg_operation(speed, saving=saving)
            * self.route.distance
            * self.global_env.carbon_tax_rates
        )

    def cost_operation(self, ratio=1.0):
        # Determine the expenses associated with vessel operations
        return self.cost_fuel(speed=self.vessel.speed_2021, saving=0.0) * ratio

    def cost_route(self):
        # Evaluate costs associated with traversing specific routes or canals
        return 0.0

    def income_freight(self):
        # Estimate the revenue generated from the transportation of goods
        return self.vessel.capacity * self.utilization_rate * self.route.freight_rate

    def hours_voyage(self, speed=None, acc=False):
        if acc:
            h0 = self.vessel.hours_2021
            p0 = (365 * 24 - h0) * 0.5
            return (h0 + p0) / (1 + p0 * speed / (h0 * self.vessel.speed_2021))
        else:
            return self.vessel.hours_2021

    def nmb_trip(self, speed=None, acc=False):
        speed = speed or self.vessel.speed_2021
        return self.hours_voyage(speed, acc) * speed / self.route.distance

    def profit_trip(self, speed=None):
        speed = speed or self.vessel.speed_2021

        res = 0.0
        saving = 0.0

        cost_retrofit, saving = self.cost_retrofit(speed)
        res += cost_retrofit
        res += self.cost_fuel(speed, saving=saving)
        res += self.cost_carbon_tax(speed, saving=saving)
        res += self.cost_operation()
        res += self.cost_route()
        res += self.income_freight()

        return res

    def profit_year(self, pr=False, speed=None):
        speed = speed or self.vessel.speed_2021

        res = self.profit_trip(speed) * self.nmb_trip(speed, acc=True)

        if pr:
            print(
                f"Profit of {self.vessel.name} in one year at speed {speed:.2f} knots: {res/1e6:.2f} million dollars"
            )

        return res

    def plot_profit_year(self, pr=False):
        vs = np.arange(10, 24, 0.01)
        profits = np.array([self.profit_year(speed=vs[i]) for i in range(len(vs))])
        v_best = vs[np.argmax(profits)]
        profit_best = np.max(profits)
        if pr:
            print(
                f"Profit of {self.vessel.name} in one year at speed {v_best:.2f} knots: {profit_best/1e6:.2f} million dollars"
            )
        plt.plot(vs, profits)
        plt.annotate(
            f"Best speed={v_best:0.2f} knots",
            xy=(v_best, profit_best),
            xytext=(v_best, profit_best + 5),
            arrowprops=dict(facecolor="red"),
        )
        plt.xlabel("Vessel Speed (knot)")
        plt.ylabel("Profit (dollar)")
        plt.show()

        return v_best


def settle():
    # Reading an Excel file using Pandas
    df_vessels = pd.read_excel("./data/CACIB-SAMPLE.xlsx")

    # Creating a list of Vessel objects
    vessels = [Vessel(row) for _, row in df_vessels.iterrows()]

    # Initializing GlobalEnv object
    env = GlobalEnv(ifo380_price=494.0, vlsifo_price=631.5, carbon_tax_rates=94.0)

    # Initializing Route object
    shg_rtm = Route(
        name="Shanghai-Rotterdam",
        route_type="CONTAINER SHIPS",
        distance=11999.0,
        freight_rate=1479.0,
    )
    stm = Settlement(
        vessel=vessels[1], route=shg_rtm, global_env=env, utilization_rate=0.95
    )
    stm.profit_year(pr=True)
    stm.plot_profit_year(pr=True)
