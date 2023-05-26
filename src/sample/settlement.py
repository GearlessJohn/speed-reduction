import numpy as np


class Settlement:
    def __init__(self, vessel, route, global_market, utilization_rate, speed=None):
        self.vessel = vessel
        self.route = route
        self.global_market = global_market
        self.utilization_rate = utilization_rate
        if speed is None:
            self.speed = self.vessel.speed_2021
        else:
            self.speed = speed

    def cost_fuel(self):
        # Calculate the expenditure on fuel
        return -(
            self.vessel.fuel_consumption_rate(self.speed)
            * self.route.distance
            * self.global_market.fuel_price(self.vessel.main_engine_fuel_type)
        )

    def cost_fuel_unit(self, pr=False):
        cost = self.cost_fuel() / (self.vessel.capacity * self.utilization_rate)
        if pr:
            print(
                f"Fuel cost of {self.vessel.name} for route {self.route.name}: {cost:.1f} dollars/{self.vessel.unit}"
            )
        return cost

    def ghg_operation(self):
        return self.vessel.co2_emission(self.speed)

    def ghg_construction(self, ratio):
        return self.vessel.co2_emission(self.vessel.service_speed) * ratio

    def cost_carbon_tax(self):
        # Assess the financial implications of carbon emissions
        return -(
            self.ghg_operation()
            * self.route.distance
            * self.global_market.carbon_tax_rates
        )

    def cost_operation(self, ratio=1.0):
        # Determine the expenses associated with vessel operations
        return self.cost_fuel() * ratio

    def cost_route(self):
        # Evaluate costs associated with traversing specific routes or canals
        return 0

    def income_freight(self):
        # Estimate the revenue generated from the transportation of goods
        return self.vessel.capacity * self.utilization_rate * self.route.freight_rate

    def nmb_trip(self, hours_voyage=None):
        if hours_voyage is None:
            hours_voyage = self.vessel.hours_2021
        return hours_voyage * self.speed / self.route.distance

    def profit_trip(self):
        res = 0.0
        res += self.cost_fuel()
        res += self.cost_carbon_tax()
        res += self.cost_operation()
        res += self.cost_route()

        res += self.income_freight()

        return res

    def profit_year(self, pr=False):
        res = self.profit_trip() * self.nmb_trip()
        if pr:
            print(
                f"Profit of {self.vessel.name} in one year at speed {self.speed:.2f} knots: {res*1e-6:2f} million dollars"
            )

        return res
