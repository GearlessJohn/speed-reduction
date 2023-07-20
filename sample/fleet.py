import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from settlement import Settlement


class Fleet:
    """Fleet and its decision-making.

    For a fleet, it has a number of vessels of different categories.
    The goal of the fleet is to maximize total profit.

    Attributes:
        vessels (List[Vessel]): List of vessel types in the fleet.
        nmb (numpy.ndarray): List of number of every type of vessel.
        speeds (numpy.ndarray): List of speed choice of every vessel in each year.
        routes (Route): List of routes for every type of vessel.
        global_env (GlobalEnv): Current global markets and policies.
        years (List[int]): List of years for which the optimization is performed.

    """

    def __init__(self, vessels, routes, global_env, years=np.arange(4)):
        self.vessels = vessels
        self.nmb = np.ones((len(vessels), len(years)))
        self.speeds = np.zeros((len(vessels), len(years)))
        self.routes = routes
        self.global_env = global_env
        self.years = years

        # Create Settlements for every vessel
        self.stms = [
            Settlement(
                vessel=vessels[j], route=self.routes[j], global_env=self.global_env
            )
            for j in range(len(vessels))
        ]

    def construction(self, i, j, stm, speed, acc):
        """Estimate the excess demand of maritime transport due to speed reduction
        and return the cost and ghg emission of construction of new vessel.

        Based on the speed of model j ship in year i, the difference in capacity
        is calculated using the value in 2021 as a reference.
        The number of new ships to be built is also determined based on the difference.

        Args:
            i (int): The year of construction.
            j (int): Index of the related vessel.
            stm (Settlement): The pre-created Settlement instance for vessel j.
            speed (float): Speed choice of vessel j in year i.
            acc (bool): Whether consider the change of voyages hours.

        Returns:
            tuple: A tuple of three elements.
            The first element is number of orders of vessel j in year i.
            The second element is the corresponding cost.
            The third element is the corresponding CO2 emission.

            The three elements are always positive or zero, which means there is only construction but no destruction.

        """
        # Calculate the percentage difference in transportation capability relative to 2021 for the given speed
        diff = 1 - (
                speed
                * stm.voyage_hours(speed=speed, acc=acc)
                * (self.nmb[j][i + 1] if i + 1 <= self.years[-1] else self.nmb[j][i])
                / (stm.vessel.speed_2021 * stm.vessel.hours_2021)
        )

        # If there is more supply than demand, the fleet will not order new vessels.
        if diff <= 0:
            return 0, 0, 0

        # We have two options for CO2 emission:
        # The larger ones take into account the carbon emissions of maintenance and ship dismantling.
        # The smaller ones only consider construction.
        # co2_0 = 4.103e4
        co2_0 = 2.29e4

        dwt0 = 74296

        # Categorizing the construction costs of different vessels
        if stm.vessel.vessel_type == "CONTAINERS":
            cost_construction = (95e6 * stm.vessel.capacity / 8000) * diff
        elif stm.vessel.vessel_type == "BULKERS":
            cost_construction = 25e6 * stm.vessel.dwt / 7e4 * diff
        else:
            cost_construction = 0.0

        # Assuming a linear relationship between CO2 emissions and ship's DWT
        emission_construction = co2_0 * (stm.vessel.dwt / dwt0) * diff
        return diff, cost_construction, emission_construction

    def global_optimization(
            self,
            retrofit,
            acc=True,
            cii_limit=True,
            construction=True,
            pr=False,
            plot=True
    ):
        """Perform optimization on vessel speed for maximum 4-year profit with construction,
        update the speeds attribute, and calculates related metrics.

        The function calculates maximum 4-year profit for a range of speeds for each year and
        identifies the optimal speed combination that maximizes profit. And it updates the orders of new vessels based
        on the difference between demand and supply of maritime transportation. Can optionally plot the results.

        Args:
            retrofit (bool): Whether consider the retrofitting, else not.
            acc (bool, optional): Whether consider the change of voyages hours.
            cii_limit (bool, optional): Whether apply the Carbon Intensity Indicator (CII) limit. Default is True.
            construction (bool, optional): Whether construct new wells. Default is True.
            pr (bool, optional): If True, prints and plots the optimization results. Default is False.

        Returns:
           profits (np.ndarray): the list of best profits in each year for every vessel

        """
        self.speeds = []
        self.nmb = np.ones((len(self.vessels), len(self.years)))
        profits = []
        emissions = []
        ciis = []
        for j in tqdm(range(len(self.vessels))):
            vessel = self.vessels[j]
            power = 2.0 if vessel.speed_2021 < 13.0 else 3.0

            stm = self.stms[j]
            v_best, profits_best, emissions_best, cii_best = stm.optimization(
                retrofit=retrofit,
                power=power,
                years=self.years,
                cii_limit=cii_limit,
                acc=acc,
                pr=False,
                plot=False
            )
            self.speeds.append(v_best)
            if construction:
                for i in self.years:
                    diff, cost_construction, emission_construction = self.construction(
                        i=i, j=j, stm=stm, speed=v_best[i], acc=acc
                    )
                    emissions_best[i] *= self.nmb[j, i]
                    profits_best[i] *= self.nmb[j, i]

                    emissions_best[i] += emission_construction
                    profits_best[i] -= cost_construction
                    if i + 2 <= self.years[-1]:
                        self.nmb[j, i + 2:] += diff

            profits.append(profits_best)
            emissions.append(emissions_best)
            ciis.append(cii_best)

        self.speeds = np.array(self.speeds)
        profits = np.array(profits)
        emissions = np.array(emissions)

        np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})
        nmb_vessels = len(self.vessels)
        if pr:
            print("Speed of vessels by type:")
            for j in range(nmb_vessels):
                print(
                    f"\t{self.vessels[j].name}: (2021: {self.vessels[j].speed_2021:0.3f}) \t",
                    self.speeds[j],
                )

            print("Profits of vessels by type (M $):")
            # The result could still be negative because of the cost of construction
            for j in range(nmb_vessels):
                print(
                    f"\t{self.vessels[j].name}:\t",
                    profits[j] / 1e6,
                )

            print("Emission of vessels by type:")
            for j in range(nmb_vessels):
                print(
                    f"\t{self.vessels[j].name}: (2021: {self.vessels[j].co2_2021:0.3f}) \t",
                    emissions[j],
                )

            print("Number of vessels by type:")
            for j in range(nmb_vessels):
                print(f"\t{self.vessels[j].name}: \t", self.nmb[j, :])

            print("CII class of vessels by type:")
            for j in range(nmb_vessels):
                print(f"\t{self.vessels[j].name}: \t", ciis[j])
        if plot:
            fig, axs = plt.subplots(nrows=nmb_vessels, ncols=2, figsize=(10, 6))
            for j in range(nmb_vessels):
                axs[j][0].plot(2023 + self.years, self.speeds[j])
                axs[j][0].axline(
                    xy1=(2023, self.vessels[j].speed_2021),
                    slope=0,
                    c="red",
                    ls="-.",
                    label="2021 speed",
                )
                axs[j][0].set_title(f"Speed of {self.vessels[j].name}")
                axs[j][0].set_xticks(2023 + self.years)
                axs[j][0].legend()

                axs[j][1].plot(2023 + self.years, emissions[j])
                axs[j][1].axline(
                    xy1=(2023, self.vessels[j].co2_2021),
                    slope=0,
                    c="red",
                    ls="-.",
                    label="2021 emission",
                )
                axs[j][1].set_title(f"Emission of {self.vessels[j].name}")
                axs[j][1].set_xticks(2023 + self.years)
                axs[j][1].legend()
            plt.subplots_adjust(hspace=0.584)
            plt.show()

        return profits

    def freight_estimator(self, capacity_by_type_ini, freight_rates_ini, elas, acc):
        capacity_by_type = {}
        for j in range(len(self.vessels)):
            vessel = self.vessels[j]
            if vessel.vessel_type not in capacity_by_type:
                capacity_by_type[vessel.vessel_type] = np.zeros(len(self.years))

            capacity_by_type[vessel.vessel_type] += (
                    vessel.capacity
                    * self.nmb[j]
                    * self.speeds[j]
                    * self.stms[j].voyage_hours(speed=self.speeds[j], acc=acc)
            )

        for j in range(len(self.routes)):
            self.routes[j].freight_rates = freight_rates_ini[j] * (
                    elas
                    + 1
                    - elas
                    * capacity_by_type[self.routes[j].route_type]
                    / capacity_by_type_ini[self.routes[j].route_type]
            )
        return

    def one_step(
            self,
            capacity_by_type_ini,
            freight_rates_ini,
            elas,
            retrofit,
            acc,
            cii_limit,
            construction,
    ):
        profits_best = self.global_optimization(
            retrofit=retrofit,
            acc=acc,
            cii_limit=cii_limit,
            construction=construction,
            pr=False,
            plot=False
        )
        self.freight_estimator(
            capacity_by_type_ini=capacity_by_type_ini,
            freight_rates_ini=freight_rates_ini,
            elas=elas,
            acc=acc,
        )
        return profits_best

    def mean_field(
            self,
            tol=25e-3,
            max_iter=20,
            elas=1.9321,
            retrofit=False,
            acc=True,
            cii_limit=True,
            construction=True,
            plot=True
    ):
        capacity_by_type_ini = {}
        freight_rates_ini = np.array(
            [self.routes[j].freight_rates for j in range(len(self.routes))]
        )
        speeds_plot = []
        profits_plot = []
        for j in range(len(self.vessels)):
            vessel = self.vessels[j]
            if vessel.vessel_type not in capacity_by_type_ini:
                capacity_by_type_ini[vessel.vessel_type] = np.zeros(len(self.years))

            capacity_by_type_ini[vessel.vessel_type] += (
                    vessel.capacity * vessel.speed_2021 * vessel.hours_2021
            )

        for i in range(1, max_iter + 1):
            speeds_previous = self.speeds
            print(f"iteration {i}:")
            profits_best = self.one_step(
                capacity_by_type_ini=capacity_by_type_ini,
                freight_rates_ini=freight_rates_ini,
                elas=elas,
                retrofit=retrofit,
                acc=acc,
                cii_limit=cii_limit,
                construction=construction,
            )
            speeds_plot.append(self.speeds[1, 2])
            profits_plot.append(profits_best[1, 2])
            print("Speed of vessels by type:")
            for j in range(len(self.vessels)):
                print(
                    f"\t{self.vessels[j].name}:\t",
                    self.speeds[j] - speeds_previous[j],
                )
            print("Freight rates:")
            routes = set()
            for j in range(len(self.vessels)):
                if self.routes[j].name not in routes:
                    print(
                        f"\t{self.routes[j].name}:\t",
                        self.routes[j].freight_rates,
                    )
                    routes.add(self.routes[j].name)
            print()
            if np.mean(np.abs(self.speeds - speeds_previous)) < tol:
                print(f"Tolerance satisfied at iteration {i}!")
                break

        print("Final speed of vessels:")
        for j in range(len(self.vessels)):
            print(
                f"\t{self.vessels[j].name}:\t",
                self.speeds[j],
            )
        print("Final freight rates:")
        routes = set()
        for j in range(len(self.vessels)):
            if self.routes[j].name not in routes:
                print(
                    f"\t{self.routes[j].name}:\t",
                    self.routes[j].freight_rates,
                )
                routes.add(self.routes[j].name)

        if plot:
            fig, ax = plt.subplots()
            ax.plot(np.arange(len(speeds_plot)), speeds_plot, label="speed", color="blue")
            ax.set_xlabel("iteration")
            ax.set_ylabel("Speed (knot)", color="blue")

            ax.legend(loc="upper left")

            ax1 = ax.twinx()
            ax1.plot(np.arange(len(speeds_plot)), np.array(profits_plot) / 1e6, label="profit", color="green")
            ax1.set_ylabel("Profit (M$)", color="green")
            ax1.legend(loc="upper right")

            fig.suptitle(
                "Iteration Trace of Bulker 01 in 2025"
            )
            plt.show()

        return
