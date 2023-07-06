import numpy as np
from settlement import Settlement
import matplotlib.pyplot as plt
from tqdm import tqdm


class Fleet:
    def __init__(self, vessels, routes, global_env, years=np.arange(4)):
        self.vessels = vessels
        self.nmb = np.ones((len(vessels), len(years)))
        self.speeds = np.zeros((len(vessels), len(years)))
        self.routes = routes
        self.global_env = global_env
        self.years = years
        self.stms = [
            Settlement(
                vessel=vessels[j], route=self.routes[j], global_env=self.global_env
            )
            for j in range(len(vessels))
        ]

    def construction(self, i, j, stm, speed, acc):
        """
        This function estimates the excess demand of maritime transport due to speed reduction 
        and returns the cost and ghg emission of construction of new vessel.
        """
        diff = 1 - (
            speed
            * stm.hours_voyage(speed=speed, acc=acc)
            * (self.nmb[j][i + 1] if i + 1 <= self.years[-1] else self.nmb[j][i])
            / (stm.vessel.speed_2021 * stm.vessel.hours_2021)
        )

        # If there if more supply than demand, the fleet will not order new vessels.
        if diff <= 0:
            return 0, 0, 0

        # CO20 = 4.103e4
        CO20 = 2.29e4
        dwt0 = 74296
        if stm.vessel.vessel_type == "CONTAINERS":
            cost_construction = (95e6 * stm.vessel.capacity / 8000) * diff
        elif stm.vessel.vessel_type == "BULKERS":
            cost_construction = 25e6 * stm.vessel.dwt / 7e4 * diff
        else:
            cost_construction = 0.0
        emission_construction = CO20 * (stm.vessel.dwt / dwt0) * diff
        return (diff, cost_construction, emission_construction)

    def global_optimization(
        self,
        retrofit,
        acc=True,
        cii_limit=True,
        construction=True,
        pr=False,
    ):
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
                        self.nmb[j, i + 2 :] += diff

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
                    f"\t{self.vessels[j].name}: (2021: {self.vessels[j].co2_emission_2021:0.3f}) \t",
                    emissions[j],
                )

            print("Number of vessels by type:")
            for j in range(nmb_vessels):
                print(f"\t{self.vessels[j].name}: \t", self.nmb[j, :])

            print("CII class of vessels by type:")
            for j in range(nmb_vessels):
                print(f"\t{self.vessels[j].name}: \t", ciis[j])

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
                    xy1=(2023, self.vessels[j].co2_emission_2021),
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
                * self.stms[j].hours_voyage(speed=self.speeds[j], acc=acc)
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

        fig, ax = plt.subplots()
        ax.plot(np.arange(len(speeds_plot)), speeds_plot, label="speed", color="blue")
        ax.set_xlabel("iteration")
        ax.set_ylabel("Speed (knot)", color="blue")
        # ax.axvline(x=v_best, ymax=profit_best, c="red", linestyle="--")
        # ax.axvline(
        #     x=self.vessel.speed_2021,
        #     ymax=profit_best,
        #     c="grey",
        #     linestyle="-.",
        # )
        # ax.annotate(
        #     f"Optimal Speed={v_best:0.2f} knots",
        #     xy=(v_best, profit_best),
        #     xytext=(v_best, profit_best * 0.95),
        #     arrowprops=dict(facecolor="red"),
        # )
        ax.legend(loc="upper left")

        ax1 = ax.twinx()
        ax1.plot(np.arange(len(speeds_plot)), np.array(profits_plot)/1e6, label="profit", color="green")
        ax1.set_ylabel("Profit (M$)", color="green")
        ax1.legend(loc="upper right")

        fig.suptitle(
            "Iteration Trace of Bukler 01 in 2025"
        )
        plt.show()

        return
