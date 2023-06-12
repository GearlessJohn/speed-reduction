import numpy as np


class fleet:
    def __init__(self, vessels, routes, global_env, years=[2023, 2024, 2025, 2026]):
        self.vessels = vessels
        self.routes = routes
        self.global_env = global_env

    def global_optimization(
        self, power, retrofit, lim=0.2, years=[2023, 2024, 2025, 2026], pr=False
    ):
        for i in range(len(self.vessels)):
            vessel = self.vessels[i]
            speed_ini = vessel.speed_2021
            vs = np.tile(speed_ini + np.linspace(-3, 3, 0.1), (4, 1))

        return
