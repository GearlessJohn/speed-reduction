import numpy as np
from settlement import Settlement
import matplotlib.pyplot as plt
from tqdm import tqdm


class Fleet:
    def __init__(self, vessels, routes, global_env, years=np.arange(4)):
        self.vessels = vessels
        self.nmb = np.ones((len(years), len(vessels)))
        self.routes = routes
        self.global_env = global_env
        self.years = years

    def construction(self, i, j, stm, speed):
        """
        This function estimates the excess demand of maritime transport due to speed reduction and returns the cost and ghg emission of construction of new vessel.
        """
        diff = 1 - (
            speed
            * stm.hours_voyage(speed=speed, acc=True)
            * (self.nmb[i + 1][j] if i + 1 <= self.years[-1] else self.nmb[i][j])
            / (stm.vessel.speed_2021 * stm.vessel.hours_2021)
        )

        # If there if more supply than demand, the fleet will not order new vessels.
        if diff < 0:
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
        speeds = []
        profits = []
        emissions = []
        ciis = []
        for j in tqdm(range(len(self.vessels))):
            vessel = self.vessels[j]
            power = 2.0 if vessel.speed_2021 < 13.0 else 3.0

            stm = Settlement(
                vessel=vessel, route=self.routes[j], global_env=self.global_env
            )
            v_best, profits_best, emissions_best, cii_best = stm.optimization(
                retrofit=retrofit,
                power=power,
                years=self.years,
                cii_limit=cii_limit,
                acc=acc,
                pr=False,
            )
            speeds.append(v_best)
            if construction:
                for i in self.years:
                    diff, cost_construction, emission_construction = self.construction(
                        i=i, j=j, stm=stm, speed=v_best[i]
                    )
                    emissions_best[i] += emission_construction
                    profits_best[i] -= cost_construction
                    if i + 2 <= self.years[-1]:
                        self.nmb[i + 2 :, j] += diff
                        emissions_best[i + 2] *= self.nmb[i + 2, j]
                        profits_best[i + 2] *= self.nmb[i + 2, j]

            profits.append(profits_best)
            emissions.append(emissions_best)
            ciis.append(cii_best)

        speeds = np.array(speeds)
        profits = np.array(profits)
        emissions = np.array(emissions)

        np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})
        nmb_vessels = len(self.vessels)
        if pr:
            print("Speed of vessels by type:")
            for j in range(nmb_vessels):
                print(
                    f"\t{self.vessels[j].name}: (2021: {self.vessels[j].speed_2021:0.3f}) \t",
                    speeds[j],
                )

            print("Emission of vessels by type:")
            for j in range(nmb_vessels):
                print(
                    f"\t{self.vessels[j].name}: (2021: {self.vessels[j].co2_emission_2021:0.3f}) \t",
                    emissions[j],
                )

            print("Number of vessels by type:")
            for j in range(nmb_vessels):
                print(f"\t{self.vessels[j].name}: \t", self.nmb[:, j])

            print("CII class of vessels by type:")
            for j in range(nmb_vessels):
                print(f"\t{self.vessels[j].name}: \t", ciis[j])

            fig, axs = plt.subplots(nrows=nmb_vessels, ncols=2, figsize=(10, 6))
            for j in range(nmb_vessels):
                axs[j][0].plot(2023 + self.years, speeds[j])
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

        return
