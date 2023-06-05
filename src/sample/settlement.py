import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from vessel import Vessel
from global_env import GlobalEnv
from route import Route


class Settlement:
    def __init__(self, vessels, route, global_env):
        self.vessels = vessels
        self.route = route
        self.global_env = global_env

    def cost_fuel(self, i, speed, saving):
        # Calculate the expenditure on fuel
        return -(
            self.vessels[i].fuel_consumption_rate(speed)
            * self.route.distance
            * self.global_env.fuel_price(self.vessels[i].main_engine_fuel_type)
        ) * (1 - saving)

    def cost_retrofit(self, i, speed):
        fuel_cost = self.cost_fuel(i=i, speed=speed, saving=0.0)
        # Consider now the possible retrofitting measures
        years_left = 25 - self.vessels[i].age
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

    def cost_fuel_unit(self, i, speed, saving, pr=False):
        cost = self.cost_fuel(i=i, speed=speed, saving=saving) / (
            self.vessels[i].capacity * self.route.utilization_rate
        )
        if pr:
            print(f"\tFuel cost:\t {cost:.1f} $/{self.vessels[i].unit}")
        return cost

    def cii_class(self, i, speed, year):
        class_abcde = self.global_env.cii_class(
            self.vessels[i].cii_score_2021 * (speed / self.vessels[i].speed_2021) ** 3,
            self.vessels[i].vessel_type,
            self.vessels[i].sub_type,
            self.vessels[i].dwt,
            year=year,
        )
        return class_abcde

    def ghg_operation(self, i, speed, saving):
        return self.vessels[i].co2_emission(speed) * (1 - saving)

    def ghg_construction(self, i, ratio):
        return self.vessels[i].co2_emission(self.vessels[i].speed_2021) * ratio

    def cost_carbon_tax(self, i, speed, saving):
        # Assess the financial implications of carbon emissions
        return -(
            self.ghg_operation(i=i, speed=speed, saving=saving)
            * self.route.distance
            * self.global_env.carbon_tax_rates
        )

    def cost_operation(self, i):
        # Determine the expenses associated with vessel operations
        return (
            self.cost_fuel(i=i, speed=self.vessels[i].speed_2021, saving=0.0)
            * (1 - self.route.fuel_ratio)
            / self.route.fuel_ratio
        )

    def cost_route(self, i):
        # Evaluate costs associated with traversing specific routes or canals
        return 0.0

    def income_freight(self, i):
        # Estimate the revenue generated from the transportation of goods
        return (
            self.vessels[i].capacity
            * self.route.utilization_rate
            * self.route.freight_rate
        )

    def hours_voyage(self, i, speed, acc=True):
        if acc:
            h0 = self.vessels[i].hours_2021
            p0 = (365 * 24 - h0) * 0.5
            return (h0 + p0) / (1 + p0 * speed / (h0 * self.vessels[i].speed_2021))
        else:
            return self.vessels[i].hours_2021

    def nmb_trip(self, i, speed, acc=True):
        return (
            self.hours_voyage(i=i, speed=speed, acc=acc) * speed / self.route.distance
        )

    def profit_trip(self, i, speed, retrofit):
        res = 0.0
        saving = 0.0
        if retrofit:
            cost_retrofit, saving = self.cost_retrofit(i=i, speed=speed)
            res += cost_retrofit
        res += self.cost_fuel(i=i, speed=speed, saving=saving)
        res += self.cost_carbon_tax(i=i, speed=speed, saving=saving)
        res += self.cost_operation(i=i)
        res += self.cost_route(i=i)
        res += self.income_freight(i=i)

        return res

    def profit_year(self, i, speed, retrofit, pr=False):
        res = self.profit_trip(i=i, speed=speed, retrofit=retrofit) * self.nmb_trip(
            i=i, speed=speed, acc=True
        )

        if pr:
            print(
                f"Profit of {self.vessels[i].name} in one year at speed {speed:.2f} knots: {res/1e6:.2f} million dollars"
            )

        return res

    def emission_year(self, i, speed, saving):
        return self.ghg_operation(i=i, speed=speed, saving=saving) * self.nmb_trip(
            i=i, speed=speed, acc=True
        )

    def plot_profit_year(self, i, retrofit, pr=False):
        vs = np.arange(7, 24, 0.01)
        profits = (
            np.array(
                [
                    self.profit_year(i=i, speed=vs[j], retrofit=retrofit)
                    for j in range(len(vs))
                ]
            )
            / 1e6
        )
        emissions = np.array(
            [
                self.emission_year(
                    i=i,
                    speed=vs[j],
                    saving=self.cost_retrofit(i=i, speed=vs[j])[1] if retrofit else 0.0,
                )
                for j in range(len(vs))
            ]
        )
        v_best = vs[np.argmax(profits)]
        profit_best = np.max(profits)
        saving_best = self.cost_retrofit(i=i, speed=v_best)[1]

        if pr:
            print("-" * 60)
            print("\tVessel Name:\t", self.vessels[i].name)
            print("\tType:\t\t", self.vessels[i].vessel_type)
            print("\tSub-type:\t", self.vessels[i].sub_type)
            print("\tCapacity:\t", self.vessels[i].capacity, self.vessels[i].unit)
            print()

            print("\tRoute:\t\t", self.route.name)
            print("\tDistance:\t", self.route.distance, "knots")
            print(
                "\tFreight Rate:\t",
                self.route.freight_rate,
                f"$/{self.vessels[i].unit}",
            )
            print(
                "\tFuel Price:\t",
                self.global_env.fuel_price(self.vessels[i].main_engine_fuel_type),
                "$/ton",
            )
            print("\tCarbon Tax:\t", self.global_env.carbon_tax_rates, "$/ton")
            print("\tUtilization:\t", self.route.utilization_rate * 100, "%")
            print("\tRetrofit:\t", retrofit)
            print()

            print("\tInitial Speed:\t", f"{self.vessels[i].speed_2021:.2f} knots")
            print("\tOptimal Speed:\t", f"{v_best:.2f} knots")
            print(
                "\tFuel cost:\t",
                f"{self.cost_fuel_unit(i=i,speed=v_best, saving=saving_best):.2f} $/{self.vessels[i].unit}",
            )
            print("\tAnnual Profit:\t", f"{profit_best:.2f} M $")
            print(
                "\tSpeed Reduction:\t",
                f"{(v_best-self.vessels[i].speed_2021)/self.vessels[i].speed_2021*100:.2f} %",
            )
            print(
                "\tEmission Reduction:\t",
                f"{(emissions[np.argmax(profits)]-self.emission_year(i=i, speed=self.vessels[i].speed_2021,saving=0.0))/self.emission_year(i=i, speed=self.vessels[i].speed_2021, saving=0.0)*100:.2f} %",
            )
            print(
                "\tCurrennt CII class:\t",
                f"{self.cii_class(i=i,speed=v_best, year=2023)}",
            )
            print("-" * 60)
        fig, ax = plt.subplots()
        ax.plot(vs, profits, label="profit", color="blue")
        ax.annotate(
            f"Optimal Speed={v_best:0.2f} knots",
            xy=(v_best, profit_best),
            xytext=(v_best, profit_best),
            arrowprops=dict(facecolor="red"),
        )
        ax.set_xlabel("Vessel Speed (knot)")
        ax.set_ylabel("Profit (dollar)", color="blue")
        ax.axvline(x=v_best, ymax=profit_best, c="red", linestyle="--")
        ax.legend(loc="upper left")

        ax1 = ax.twinx()
        ax1.plot(vs, emissions, label="emission", color="green")
        ax1.set_ylabel("CO2 Emission (ton)", color="green")
        ax1.legend(loc="upper right")

        fig.suptitle(
            f"Annual result, Carbon Tax: {self.global_env.carbon_tax_rates}, Retrofit: {retrofit} "
        )
        plt.show()

        return v_best


def settle(
    i,
    ifo380_price,
    vlsifo_price,
    carbon_tax_rates,
    name,
    route_type,
    distance,
    freight_rate,
    utilization_rate,
    fuel_ratio,
    retrofit,
):
    # Reading an Excel file using Pandas
    df_vessels = pd.read_excel("./data/CACIB-SAMPLE.xlsx")

    # Creating a list of Vessel objects
    vessels = [Vessel(row) for _, row in df_vessels.iterrows()]

    # Initializing GlobalEnv object
    env = GlobalEnv(
        ifo380_price=ifo380_price,
        vlsifo_price=vlsifo_price,
        carbon_tax_rates=carbon_tax_rates,
    )

    # Initializing Route object
    shg_rtm = Route(
        name=name,
        route_type=route_type,
        distance=distance,
        freight_rate=freight_rate,
        utilization_rate=utilization_rate,
        fuel_ratio=fuel_ratio,
    )
    stm = Settlement(
        vessels=vessels,
        route=shg_rtm,
        global_env=env,
    )
    stm.plot_profit_year(i=i, retrofit=retrofit, pr=True)
