class GlobalMarket:
    def __init__(self, ifo380_price, vlsifo_price, carbon_tax_rates):
        # Initializing the attributes of the GlobalMarket object
        self.ifo380_price = ifo380_price
        self.vlsifo_price = vlsifo_price
        self.carbon_tax_rates = carbon_tax_rates

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

