class Vessel:
    """The instance of vessel with all its information.

    Vessel information including ID, capacity, construction information, engine and voyage data.

    Attributes:
            name (str): Name of the vessel.
            vessel_type (str): Main type of the vessel, including CONTAINER SHIPS, BULKERS, GAS CARRIERS, TANKERS.
            sub_type (str): Subtype of the vessel, related with its size.
            dwt (float): Dead-weight tonnage of the vessel, in metric tonnes.
            capacity (float): Capacity of the vessel, number of TEU for CONTAINER SHIPS and dwt for other types.
            unit (str): Unit of capacity, TEU for CONTAINER SHIPS and metric ton for others.
            built (float): Year of construction of the vessel.
            age (int): Age of the vessel, by calculating the difference between 2023 and the year of construction.
            distance_2021 (float): Distance sailed in 2021，in nautical miles.
            hours_2021 (float): Sailing time in 2021, in hours.
            speed_2021 (float): Average speed in 2021, in nautical miles per hour.
            *_quantity_2021 (float): Fuel consumption in * category in 2021.
            co2_emission_2021 (float): CO2 emission in 2021.
            cii_score_2021 (float): Attained CII Score in 2021 .
            cii_class_2021 (str): Attained CII class in 2021, from A to E.
    """

    def __init__(self, row=None, name=None, vessel_type=None, sub_type=None, dwt=None, capacity=None, unit=None, built=None, distance_2021=None, hours_2021=None,
                 hfo_2021=None, lfo_2021=None, diesel_2021=None, lng_2021=None, co2_2021=None, cii_score_2021=None, cii_class_2021=None):
        if row is not None:
            # Initializing the attributes of the Vessel object
            # ID
            self.imo_number = row["IMO Number"]
            self.name = row["Name"]
            self.vessel_type = row["Type"]
            self.sub_type = row["Sub_type"]

            # Capacity
            self.gt = row["GT"]
            self.dwt = row["Dwt"]
            self.capacity = row["Capacity"]
            self.unit = row["Unit"]

            # Build
            self.built = row["Built"]
            self.age = int(2023 - self.built)
            self.builder = row["Builder"]
            self.length = row["Length overall (m)"]
            self.beam = row["Beam (m)"]
            self.draught = row["Draught (m)"]

            # Engine
            self.main_engine = row["Main engine"]
            self.main_engine_fuel_type = str.split(row["Main Engine Fuel Type"], sep=",")[0]
            self.output = row["Output (kW)"]
            self.service_speed = row["Service Speed Design (kn)"]

            self.sfoc = row["SFOC"]

            # Travel
            self.distance_2021 = row["Distance Travelled 2021"]
            self.hours_2021 = row["Hours Under way 2021"]
            self.speed_2021 = self.distance_2021 / self.hours_2021
            self.hfo_2021 = row["HFO quantity 2021"]
            self.lfo_2021 = row["LFO quantity 2021"]
            self.diesel_2021 = row["Diesel/Gas Oil quantity 2021"]
            self.lng_2021 = row["LNG 2021"]
            self.co2_2021 = row["CO2 Emissions TtW 2021"]
            self.cii_score_2021 = row["Actual CII 2021"]
            self.cii_class_2021 = row["2021 CII rating"]

        else:
            self.name = name
            self.vessel_type = vessel_type
            self.sub_type = sub_type
            self.dwt = dwt
            self.capacity = capacity
            self.unit = unit
            self.built = built
            self.age = int(2023 - self.built)

            self.distance_2021 = distance_2021
            self.hours_2021 = hours_2021
            self.speed_2021 = self.distance_2021 / self.hours_2021

            self.hfo_2021 = hfo_2021
            self.lfo_2021 = lfo_2021
            self.diesel_2021 = diesel_2021
            self.lng_2021 = lng_2021

            self.co2_2021 = co2_2021
            self.cii_score_2021 = cii_score_2021
            self.cii_class_2021 = cii_class_2021
