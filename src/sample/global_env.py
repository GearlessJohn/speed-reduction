import numpy as np
import bisect


class GlobalEnv:
    def __init__(self, ifo380_price, vlsifo_price, carbon_tax_rates, year_actual=2023):
        # Initializing the attributes of the GlobalMarket object
        self.ifo380_price = ifo380_price
        self.vlsifo_price = vlsifo_price
        self.carbon_tax_rates = carbon_tax_rates
        self.year_actual = year_actual

    def fuel_price(self, fuel_type):
        # Returning the fuel price based on the fuel type
        # Using the match statement available since Python 3.10
        match fuel_type:
            case "IFO380":
                return self.ifo380_price

            case "VLS IFO":
                return self.vlsifo_price

            case _:
                return self.vlsifo_price

    def cii_reduction(self, year):
        reductions = {
            2021: 0.01,
            2022: 0.03,
            2023: 0.05,
            2024: 0.07,
            2025: 0.09,
            2026: 0.11,
        }
        return reductions[year]

    def cii_ac(self, vessel_type, sub_type, dwt):
        match vessel_type:
            case "CONTAINER SHIPS":
                return 1984, 0.489
            case "BULKERS":
                return 4745, 0.622
            case "TANKERS":
                return 5247, 0.610
            case "GAS CARRIERS":
                match sub_type:
                    case "LNG":
                        if dwt >= 100000:
                            return 9.287, 0.0
                        elif dwt >= 65000:
                            return 14479e10, 2.673
                        else:
                            return 14779e10, 2.673
                    case _:
                        if dwt >= 65000:
                            return 14405e7, 2.071
                        else:
                            return 8104, 0.639
            case _:
                raise ValueError("CII Calculation: Unknow vessel type to get a and c")

    def cii_ref(self, vessel_type, sub_type, dwt, year=None):
        year = year or self.year_actual

        a, c = self.cii_ac(vessel_type, sub_type, dwt)
        cii_ref = (a * dwt**-c) * (1 - self.cii_reduction(year))
        return cii_ref

    def cii_expd(self, vessel_type, sub_type, dwt):
        match vessel_type:
            case "CONTAINER SHIPS":
                return [0.83, 0.94, 1.07, 1.19]
            case "BULKERS":
                return [0.86, 0.94, 1.06, 1.18]
            case "TANKERS":
                return [0.82, 0.93, 1.08, 1.28]
            case "GAS CARRIERS":
                match sub_type:
                    case "LNG":
                        if dwt >= 100000:
                            return [0.89, 0.98, 1.06, 1.13]
                        else:
                            return [0.78, 0.92, 1.10, 1.37]
                    case _:
                        if dwt >= 65000:
                            return [0.81, 0.91, 1.12, 1.44]
                        else:
                            return [0.85, 0.95, 1.06, 1.25]
            case _:
                raise ValueError("CII Calculation: Unknow vessel type to get exp(d)")

    def cii_fronts(self, vessel_type, sub_type, dwt, year=None):
        year = year or self.year_actual

        expd = np.array(self.cii_expd(vessel_type, sub_type, dwt))
        return expd * self.cii_ref(vessel_type, sub_type, dwt, year)

    def cii_class(self, cii_atteined, vessel_type, sub_type, dwt, year=None):
        year = year or self.year_actual

        cii_classes = ["A", "B", "C", "D", "E"]

        return cii_classes[
            bisect.bisect(
                self.cii_fronts(vessel_type, sub_type, dwt, year), cii_atteined
            )
        ]
