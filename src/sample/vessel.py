class Vessel:
    def __init__(self, row):
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
        self.hfo_quantity_2021 = row["HFO quantity 2021"]
        self.lfo_quantity_2021 = row["LFO quantity 2021"]
        self.diesel_quantity_2021 = row["Diesel/Gas Oil quantity 2021"]
        self.lng_quantity_2021 = row["LNG 2021"]
        self.co2_emission_2021 = row["CO2 Emissions TtW 2021"]
        self.cii_score_2021 = row["Actual CII 2021"]
        self.cii_class_2021 = row["2021 CII rating"]
