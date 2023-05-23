class Route:
    def __init__(self, name, route_type, distance, utilization_rate, freight_rate):
        self.name = name
        self.route_type = route_type
        self.distance = distance
        self.utilization_rate = utilization_rate
        self.freight_rate = freight_rate  # dollar/unit
