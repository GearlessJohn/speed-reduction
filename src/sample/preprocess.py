import pandas as pd
import numpy as np

# Reading an Excel file using Pandas and skipping the second row
# Also, using only columns from A to AN
data_cacib = pd.read_excel(
    "./data/CACIB fleet sample V2.xlsx", skiprows=[1], usecols="A:AN"
).dropna(subset=["Hours Under way 2021"])

print(data_cacib.columns)
# Converting the "Name" column to uppercase
data_cacib["Name"] = data_cacib["Name"].str.upper()

# Reading another Excel file and skipping the first and third rows
data_csb = pd.read_excel("./data/Sample.xlsx", skiprows=[0, 2])

# Converting the "Vessel Name" column to uppercase
data_csb["Vessel Name"] = data_csb["Vessel Name"].str.upper()

# Merging the two data frames on the "Name" and "Vessel Name" columns, respectively
# Using "right" join to keep all rows from data_csb
data = pd.merge(
    data_cacib, data_csb, left_on="Name", right_on="Vessel Name", how="left"
)

# Dropping unnecessary columns from the merged data frame
data.drop(columns=["Sub-type_x", "Vessel Name", "Fuel type"], inplace=True)

# Renaming the "Sub-type_y" column to "Sub_type"
data.rename(columns={"Sub-type_y": "Sub_type"}, inplace=True)

# Using a lambda function to create a new "Capacity" column
# Using the "TEU" column for container ships, "Capacity" column for gas carriers,
# and "Dwt" column for other types of vessels
data["Capacity"] = data.apply(
    lambda x: x.TEU
    if x.Type == "CONTAINER SHIPS"
    else x.Capacity
    if x.Type == "GAS CARRIERS"
    else x.Dwt,
    axis=1,
)

# Reordering the columns of the data frame
data = data[
    [
        "IMO Number",
        "Name",
        "Type",
        "Sub_type",
        "GT",
        "Dwt",
        "Capacity",
        "Unit",
        "Built",
        "Builder",
        "Length overall (m)",
        "Beam (m)",
        "Draught (m)",
        "Main engine",
        "Main Engine Fuel Type",
        "Output (kW)",
        "Service Speed Design (kn)",
        "SFOC",
        "Distance Travelled 2021",
        "Hours Under way 2021",
        "HFO quantity 2021",
        "LFO quantity 2021",
        "Diesel/Gas Oil quantity 2021",
        "LNG 2021",
        "CO2 Emissions TtW 2021",
        "Actual CII 2021",
        "2021 CII rating",
    ]
]

data["LNG 2021"] = data["LNG 2021"].replace(np.nan, 0)

# Printing the columns of the data frame
print(data.columns)

print(data.info)

# Writing the data frame to an Excel file
data.to_excel("./data/CACIB-SAMPLE.xlsx", index=True)
