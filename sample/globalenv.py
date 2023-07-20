import bisect

import numpy as np


class GlobalEnv:
    """Global market and policy information.

    Includes global average fuel prices and carbon tax rates, as well as a CII rating calculator.
    The rating calculations for CII are all derived from official IMO documents and use the same notation.

    Attributes:
        *_prices (List[float]): Average global price of * fuels, 2023-2026, in USD.
        carbon_tax_rates (List[float]):  Carbon tax price from 2023 to 2026, USD per metric tonne CO2.
    """

    def __init__(self, ifo380_prices, vlsifo_prices, mgo_prices, lng_prices, carbon_tax_rates):
        # Initializing the attributes of the GlobalMarket object
        self.ifo380_prices = ifo380_prices
        self.vlsifo_prices = vlsifo_prices
        self.mgo_prices = mgo_prices
        self.lng_prices = lng_prices
        self.carbon_tax_rates = carbon_tax_rates

    def cii_reduction(self, year):
        """Return the CII reduction factor for a specific year."""
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
        """Return the a and c parameters for calculation of CII reference line"""
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
                raise ValueError("CII Calculation: Unknown vessel type to get a and c")

    def cii_ref(self, vessel_type, sub_type, dwt, year):
        """"Return the CII reference line"""
        a, c = self.cii_ac(vessel_type, sub_type, dwt)
        cii_ref = (a * dwt ** -c) * (1 - self.cii_reduction(year))
        return cii_ref

    def cii_expd(self, vessel_type, sub_type, dwt):
        """Return the distances of different frontiers from the reference"""
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
                raise ValueError("CII Calculation: Unknown vessel type to get exp(d)")

    def cii_fronts(self, vessel_type, sub_type, dwt, year):
        """Return the frontiers of different classes"""
        expd = np.array(self.cii_expd(vessel_type, sub_type, dwt))
        return expd * self.cii_ref(vessel_type, sub_type, dwt, year)

    def cii_class(self, cii_attained, vessel_type, sub_type, dwt, year):
        """"Return the CII class for a given vessel with its attained CII"""
        assert year > 2020, "The year should start from 2021."
        cii_classes = ["A", "B", "C", "D", "E"]

        return cii_classes[
            bisect.bisect(
                self.cii_fronts(vessel_type, sub_type, dwt, year), cii_attained
            )
        ]
