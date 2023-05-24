import numpy as np


class Settlement:
    def __init__(self, vessel, route, global_market):
        self.vessel = vessel
        self.route = route
        self.global_market = global_market

    def fuel_cost(self, speed=None):
        # Calculate the expenditure on fuel
        return -(
            self.vessel.fuel_consumption(speed)
            * self.route.distance
            * self.global_market.fuel_price(self.vessel.main_engine_fuel_type)
        )

    def carbon_tax(self, speed=None):
        # Assess the financial implications of carbon emissions
        return -(
            self.vessel.ges_emission(speed)
            * self.route.distance
            * self.global_market.carbon_tax_rates
        )

    def operational_cost(self, cost=None):
        # Determine the expenses associated with vessel operations
        return self.fuel_cost()

    def route_cost(self):
        # Evaluate costs associated with traversing specific routes or canals
        return 0

    def freight_rate(
        self,
    ):
        # Estimate the revenue generated from the transportation of goods
        return (
            self.vessel.capacity * self.route.utilization_rate * self.route.freight_rate
        )

    def fuel_cost_unit(self, speed=None, pr=False):
        cost = self.fuel_cost(speed) / (
            self.vessel.capacity * self.route.utilization_rate
        )
        if pr:
            print(
                f"Fuel cost of {self.vessel.name} for route {self.route.name}: {cost:.1f} dollars/{self.vessel.unit}"
            )
        return

    def profit(self, speed=None):
        return (
            self.fuel_cost(speed)
            + self.carbon_tax(speed)
            + self.operational_cost()
            + self.route_cost()
            + self.freight_rate()
        )
