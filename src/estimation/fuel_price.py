import numpy as np
import pandas as pd

# prices of 2023-06-14
price_eu_current = {"ifo380": 457, "vlsfo": 530, "mgo": 680, "lng": 650}
price_ww_current = {"ifo380": 495, "vlsfo": 625, "mgo": 816, "lng": 650}

ifo380 = (
    pd.read_excel("./data/EU IFO380 Future Price.xlsx")
    .dropna(subset=["PRIOR"])[["MONTH", "PRIOR"]]
    .dropna()
)
ifo380["MONTH"] = ifo380["MONTH"].astype("datetime64[ns]")
ifo380["PRIOR"] = ifo380["PRIOR"].astype("float64")
ifo380_year = ifo380.groupby(ifo380.MONTH.dt.year).mean()
ifo380_prices = (
    ifo380_year["PRIOR"] * price_ww_current["ifo380"] / price_eu_current["ifo380"]
).to_numpy()

vlsfo = (
    pd.read_excel("./data/EU VLSFO Future Price.xlsx")
    .dropna(subset=["PRIOR"])[["MONTH", "PRIOR"]]
    .dropna()
)
vlsfo["MONTH"] = vlsfo["MONTH"].astype("datetime64[ns]")
vlsfo["PRIOR"] = vlsfo["PRIOR"].astype("float64")
vlsfo_year = vlsfo.groupby(vlsfo.MONTH.dt.year).mean()
vlsfo_prices = (
    vlsfo_year["PRIOR"] * price_ww_current["vlsfo"] / price_eu_current["vlsfo"]
).to_numpy()

mgo = (
    pd.read_excel("./data/EU MGO Future Price.xlsx")
    .dropna(subset=["PRIOR"])[["MONTH", "PRIOR"]]
    .dropna()
)
mgo["MONTH"] = mgo["MONTH"].astype("datetime64[ns]")
mgo["PRIOR"] = mgo["PRIOR"].astype("float64")
mgo_year = mgo.groupby(mgo.MONTH.dt.year).mean()
mgo_prices = (
    mgo_year["PRIOR"] * price_ww_current["mgo"] / price_eu_current["mgo"]
).to_numpy()

# 1 metric ton LNG = 52.0 million British thermal unists (MMBtu)
lng = (
    pd.read_excel("./data/EU LNG Future Price.xlsx")
    .dropna(subset=["PRIOR"])[["MONTH", "PRIOR"]]
    .dropna()
)
lng["MONTH"] = lng["MONTH"].astype("datetime64[ns]")
lng["PRIOR"] = lng["PRIOR"].astype("float64") * 52.0
lng_year = lng.groupby(lng.MONTH.dt.year).mean()
lng_prices = (
    lng_year["PRIOR"] * price_ww_current["lng"] / price_eu_current["lng"]
).to_numpy()

print(f"ifo380\t{ifo380_prices}")
print(f"vlsfo\t{vlsfo_prices}")
print(f"mgo\t{mgo_prices}")
print(f"lng\t{lng_prices}")
