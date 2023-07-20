import os

import pandas as pd


def test_data_existence():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert os.path.exists(base_dir + "/data/CACIB-SAMPLE.xlsx")


def test_data_structure():
    df_vessels = pd.read_excel(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/data/CACIB-SAMPLE.xlsx")
    data_necessary_columns = ["Name", "Type", "Sub_type", "Dwt", "Capacity", "Unit", "Built", "Distance Travelled 2021",
                              "Hours Under way 2021", "HFO quantity 2021", "LFO quantity 2021",
                              "Diesel/Gas Oil quantity 2021", "LNG 2021", "CO2 Emissions TtW 2021", "Actual CII 2021",
                              "2021 CII rating"]
    assert set(data_necessary_columns).issubset(set(df_vessels.columns.tolist()))
