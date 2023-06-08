import numpy as np
import matplotlib.pyplot as plt
from vessel import Vessel


class Settlement:
    def __init__(self, vessel, route, global_env):
        self.vessel = vessel
        self.route = route
        self.global_env = global_env

    def cost_fuel(self, speed, saving, power):
        # Calculate the expenditure on fuel
        return (
            -(
                self.vessel.hfo_quantity_2021 * self.global_env.ifo380_price
                + (self.vessel.lfo_quantity_2021 + self.vessel.diesel_quantity_2021)
                * self.global_env.mgo_price
                + self.vessel.lng_quantity_2021 * self.global_env.lng_price
            )
            / self.vessel.distance_2021
            * (speed / self.vessel.speed_2021) ** power
            * self.route.distance
            * (1 - saving)
        )

    def cost_retrofit(self, speed, power):
        fuel_cost = self.cost_fuel(speed=speed, saving=0.0, power=power)
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

    def cost_fuel_unit(self, speed, saving, power, pr=False):
        cost = self.cost_fuel(speed=speed, saving=saving, power=power) / (
            self.vessel.capacity * self.route.utilization_rate
        )
        if pr:
            print(f"\tFuel cost:\t {cost:.1f} $/{self.vessel.unit}")
        return cost

    def cii_class(self, speed, year, power):
        class_abcde = self.global_env.cii_class(
            self.vessel.cii_score_2021 * (speed / self.vessel.speed_2021) ** power,
            self.vessel.vessel_type,
            self.vessel.sub_type,
            self.vessel.dwt,
            year=year,
        )
        return class_abcde

    def ghg_operation(self, speed, saving, power):
        return (
            self.vessel.co2_emission_2021
            / self.vessel.distance_2021
            * self.route.distance
            * (speed / self.vessel.speed_2021) ** power
            * (1 - saving)
        )

    def ghg_construction(self, ratio):
        return (
            self.vessel.co2_emission_2021
            / self.vessel.distance_2021
            * self.route.distance
            * ratio
        )

    def cost_carbon_tax(self, speed, saving, power):
        # Assess the financial implications of carbon emissions
        return -(
            self.ghg_operation(speed=speed, saving=saving, power=power)
            * self.global_env.carbon_tax_rates
        )

    def cost_operation(self):
        # Determine the expenses associated with vessel operations
        return (
            self.cost_fuel(speed=self.vessel.speed_2021, saving=0.0, power=3.0)
            * (1 - self.route.fuel_ratio)
            / self.route.fuel_ratio
        )

    def cost_route(self):
        # Evaluate costs associated with traversing specific routes or canals
        return 0.0

    def income_freight(self):
        # Estimate the revenue generated from the transportation of goods
        return (
            self.vessel.capacity * self.route.utilization_rate * self.route.freight_rate
        )

    def hours_voyage(self, speed, acc=True):
        if acc:
            h0 = self.vessel.hours_2021
            p0 = (365 * 24 - h0) * 0.5
            return (h0 + p0) / (1 + p0 * speed / (h0 * self.vessel.speed_2021))
        else:
            return self.vessel.hours_2021

    def nmb_trip(self, speed, acc=True):
        return self.hours_voyage(speed=speed, acc=acc) * speed / self.route.distance

    def profit_trip(self, speed, power, retrofit):
        res = 0.0
        saving = 0.0
        if retrofit:
            cost_retrofit, saving = self.cost_retrofit(speed=speed, power=power)
            res += cost_retrofit
        res += self.cost_fuel(speed=speed, saving=saving, power=power)
        res += self.cost_carbon_tax(speed=speed, saving=saving, power=power)
        res += self.cost_operation()
        res += self.cost_route()
        res += self.income_freight()

        return res

    def emission_year(self, speed, saving, power):
        return self.ghg_operation(
            speed=speed, saving=saving, power=power
        ) * self.nmb_trip(speed=speed, acc=True)

    def profit_year(self, speed, power, retrofit, pr=False):
        res = self.profit_trip(
            speed=speed, power=power, retrofit=retrofit
        ) * self.nmb_trip(speed=speed, acc=True)

        if pr:
            print(
                f"Profit of {self.vessel.name} in one year at speed {speed:.2f} knots: {res/1e6:.2f} million dollars"
            )

        return res

    def plot_profit_year(self, retrofit, power, pr=False):
        vs = np.arange(7, 24, 0.01)
        profits = (
            np.array(
                [
                    self.profit_year(speed=vs[j], power=power, retrofit=retrofit)
                    for j in range(len(vs))
                ]
            )
            / 1e6
        )
        emissions = np.array(
            [
                self.emission_year(
                    speed=vs[j],
                    saving=self.cost_retrofit(speed=vs[j])[1] if retrofit else 0.0,
                    power=power,
                )
                for j in range(len(vs))
            ]
        )
        v_best = vs[np.argmax(profits)]
        profit_best = np.max(profits)
        saving_best = (
            self.cost_retrofit(speed=v_best, power=power)[1] if retrofit else 0.0
        )

        if pr:
            print("-" * 60)
            print("\tVessel Name:\t", self.vessel.name)
            print("\tType:\t\t", self.vessel.vessel_type)
            print("\tSub-type:\t", self.vessel.sub_type)
            print("\tCapacity:\t", self.vessel.capacity, self.vessel.unit)
            print()

            print("\tRoute:\t\t", self.route.name)
            print("\tDistance:\t", self.route.distance, "knots")
            print(
                "\tFreight Rate:\t",
                self.route.freight_rate,
                f"$/{self.vessel.unit}",
            )
            # print(
            #     "\tFuel Price:\t",
            #     self.global_env.fuel_price(self.vessel.main_engine_fuel_type),
            #     "$/ton",
            # )
            print("\tCarbon Tax:\t", self.global_env.carbon_tax_rates, "$/ton")
            print("\tUtilization:\t", self.route.utilization_rate * 100, "%")
            print("\tRetrofit:\t", retrofit)
            print()

            print("\t2021 Speed:\t", f"{self.vessel.speed_2021:.2f} knots")
            print("\tOptimal Speed:\t", f"{v_best:.2f} knots")
            fc = self.cost_fuel_unit(speed=v_best, saving=saving_best, power=power)
            print(
                "\tFuel cost:\t",
                f"{fc:.2f} $/{self.vessel.unit}",
            )
            oc = self.cost_operation() / (
                self.vessel.capacity * self.route.utilization_rate
            )
            print(
                "\tOperation cost:\t",
                f"{oc:.2f} $/{self.vessel.unit}",
            )
            print(
                "\tProfitability:\t",
                f"{(self.route.freight_rate+oc + fc)/self.route.freight_rate*100:.2f} %",
            )
            print("\tAnnual Profit:\t", f"{profit_best:.2f} M $")
            print()
            print(
                "\tSpeed Variation:\t",
                f"{(v_best-self.vessel.speed_2021)/self.vessel.speed_2021*100:+.2f} %",
            )
            print(
                "\tEmission Variation:\t",
                f"{(emissions[np.argmax(profits)]-self.emission_year( speed=self.vessel.speed_2021,saving=0.0, power=power))/self.emission_year( speed=self.vessel.speed_2021, saving=0.0, power=power)*100:+.2f} %",
            )
            print(
                "\t2021 CII class:\t\t",
                f"{self.vessel.cii_class_2021}",
            )
            print(
                "\tCurrent CII class:\t",
                f"{self.cii_class(speed=v_best,power=power, year=2023)}",
            )
            print("-" * 60)
            fig, ax = plt.subplots()
            ax.plot(vs, profits, label="profit", color="blue")
            ax.set_xlabel("Vessel Speed (knot)")
            ax.set_ylabel("Profit (dollar)", color="blue")
            ax.axvline(x=v_best, ymax=profit_best, c="red", linestyle="--")
            ax.axvline(
                x=self.vessel.speed_2021,
                ymax=profit_best,
                c="grey",
                linestyle="-.",
            )
            ax.annotate(
                f"Optimal Speed={v_best:0.2f} knots",
                xy=(v_best, profit_best),
                xytext=(v_best, profit_best * 0.95),
                arrowprops=dict(facecolor="red"),
            )
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

    def global_optimization(
        self, power, retrofit, lim=0.2, years=[2023, 2024, 2025, 2026], pr=False
    ):
        speed_ini = self.vessel.speed_2021
        vs = np.tile(speed_ini + np.linspace(-3, 3, 0.1), (4, 1))

        return


def settle(i, data_vessels, env, route, power, retrofit, pr):
    # Creating a list of Vessel objects
    vessels = [Vessel(row) for _, row in data_vessels.iterrows()]

    stm = Settlement(
        vessel=vessels[i],
        route=route,
        global_env=env,
    )
    stm.plot_profit_year(retrofit=retrofit, power=power, pr=pr)
    return stm
