class Route:
    """Route of vessel with freight rates.

    Route for a particular type of vessel , including information on sailing distances and freight rates.

    Attributes:
        name (str): Name of the route
        route_type (str): Types of ships consumed on the route, such as CONTAINER SHIPS, BULKERS, etc.
        distance (float): Total voyage distance of the route, in nautical miles.
        freight_rates (float):  Freight, in consistent the `unit` attribute of the vessel corresponding to route_type.
        utilization_rate (float): Loading rate of the vessel, in percentage.
        fuel_ratio (float): Proportion of fuel cost to total cost for the route, in percentage.
    """

    def __init__(self, name, route_type, distance, freight_rates, utilization_rate, fuel_ratio):
        self.name = name
        self.route_type = route_type
        self.distance = distance
        self.freight_rates = freight_rates  # dollar/unit
        self.utilization_rate = utilization_rate
        self.fuel_ratio = fuel_ratio
