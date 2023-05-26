import numpy as np


class Settlement:
    def __init__(self, vessel, route, global_market, utilization_rate):
        self.vessel = vessel
        self.route = route
        self.global_market = global_market
        self.utilization_rate = utilization_rate

    def cost_fuel(self, speed=None):
        # Calculate the expenditure on fuel
        return -(
            self.vessel.fuel_consumption_rate(speed)
            * self.route.distance
            * self.global_market.fuel_price(self.vessel.main_engine_fuel_type)
        )

    def cost_fuel_unit(self, speed=None, pr=False):
        cost = self.cost_fuel(speed) / (self.vessel.capacity * self.utilization_rate)
        if pr:
            print(
                f"Fuel cost of {self.vessel.name} for route {self.route.name}: {cost:.1f} dollars/{self.vessel.unit}"
            )
        return cost

    def ghg_operation(self, speed=None):
        return self.vessel.co2_emission(speed)

    def ghg_construction(self, ratio=None):
        return self.vessel.co2_emission(self.vessel.speed_2021) * ratio

    def cost_carbon_tax(self, speed=None):
        # Assess the financial implications of carbon emissions
        return -(
            self.ghg_operation(speed)
            * self.route.distance
            * self.global_market.carbon_tax_rates
        )

    def cost_operation(self, ratio=1.0):
        # Determine the expenses associated with vessel operations
        return self.cost_fuel(speed=self.vessel.speed_2021) * ratio

    def cost_route(self):
        # Evaluate costs associated with traversing specific routes or canals
        return 0.0

    def income_freight(self):
        # Estimate the revenue generated from the transportation of goods
        return self.vessel.capacity * self.utilization_rate * self.route.freight_rate

    def hours_voyage(self, speed=None):
        return self.vessel.hours_2021

    def nmb_trip(self, speed=None):
        if speed is None:
            speed = self.vessel.speed_2021

        return self.hours_voyage(speed) * speed / self.route.distance

    def profit_trip(self, speed=None):
        res = 0.0
        res += self.cost_fuel(speed)
        res += self.cost_carbon_tax(speed)
        res += self.cost_operation()
        res += self.cost_route()

        res += self.income_freight()

        return res

    def profit_year(self, pr=False, speed=None):
        if speed is None:
            speed = self.vessel.speed_2021

        res = self.profit_trip(speed) * self.nmb_trip(speed)

        if pr:
            print(
                f"Profit of {self.vessel.name} in one year at speed {speed:.2f} knots: {res/1e6:.2f} million dollars"
            )

        return res

    # def plot_profit_year(self):
    #     vs = np.linspace(10, 24, 0.5)

    #     for v in vs:
    #         profit = self.profit_year()
