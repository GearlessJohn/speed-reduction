import numpy as np
from settlement import Settlement


class Fleet:
    def __init__(self, vessels, routes, global_env):
        self.vessels = vessels
        self.routes = routes
        self.global_env = global_env

    def construction(self, vessel, speed):
        """
        This function estimates the excess demand of maritime transport due to speed reduction and returns the cost and ghg emission of construction of new vessel.
        """
        cost = 0.0
        emission = 0.0
        return

    def global_optimization(self, retrofit, years=range(4), cii_limit=True, pr=False):
        speeds = []
        profits = []
        emissions = []
        for i in range(len(self.vessels)):
            vessel = self.vessels[i]
            power = 2.0 if vessel.speed_2021 < 13.0 else 3.0

            stm = Settlement(
                vessel=vessel, route=self.routes[i], global_env=self.global_env
            )
            v_best, profits_best, emissions_best = stm.optimization(
                retrofit=retrofit,
                power=power,
                years=years,
                cii_limit=cii_limit,
                pr=False,
            )
            speeds.append(v_best)
            profits.append(profits_best)
            emissions.append(emissions_best)
        speeds = np.array(speeds)
        profits = np.array(profits)
        emissions = np.array(emissions)
        np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})
        if pr:
            print(f"Average Speed:\t{np.average(speeds, axis=1)}")
            print(f"Average Profits:\t{np.average(profits, axis=1)}")
            print(f"Average Emissions:\t{np.average(emissions, axis=1)}")

        return
