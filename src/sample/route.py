class Route:
    def __init__(
        self,
        name,
        route_type,
        distance,
        freight_rate,
        utilization_rate,
        fuel_ratio,
    ):
        self.name = name
        self.route_type = route_type
        self.distance = distance
        self.freight_rate = freight_rate  # dollar/unit
        self.utilization_rate = utilization_rate
        self.fuel_ratio = fuel_ratio
