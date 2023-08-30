import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import PolynomialFeatures

data_cacib = pd.read_excel("./data/CACIB fleet sample V2.xlsx", skiprows=[1], usecols="A:AN").dropna(
    subset=["Distance Travelled 2019", "Hours Under way 2019", "Distance Travelled 2020", "Hours Under way 2020",
            "Distance Travelled 2021", "Hours Under way 2021"])
data_cacib.rename(columns={"Distance Travelled 2019": "D2019", "Distance Travelled 2020": "D2020",
                           "Distance Travelled 2021": "D2021", "Hours Under way 2019": "H2019",
                           "Hours Under way 2020": "H2020", "Hours Under way 2021": "H2021"}, inplace=True)
voyage = data_cacib[["Name", "D2019", "H2019", "D2020", "H2020", "D2021", "H2021"]]
voyage["S2019"] = voyage["D2019"] / voyage["H2019"]
voyage["S2020"] = voyage["D2020"] / voyage["H2020"]
voyage["S2021"] = voyage["D2021"] / voyage["H2021"]

voyage_melted = voyage.melt(id_vars="Name", value_vars=["H2019", "H2020", "H2021"], var_name="year", value_name="H")[
    ["Name", "year", "H"]]
voyage_melted["year"] = voyage_melted["year"].str[1:].apply(lambda x: int(x))

voyage_melted["S"] = \
    voyage.melt(id_vars="Name", value_vars=["S2019", "S2020", "S2021"], var_name="year", value_name="S")["S"]

voyage_melted["N"] = voyage_melted["S"] * voyage_melted["H"]
polynomial_features = PolynomialFeatures(degree=1)
xp = polynomial_features.fit_transform(voyage_melted["S"].to_numpy().reshape(-1, 1))

md = sm.RLM(voyage_melted["N"], xp)
mdf = md.fit()
print(mdf.summary())
plt.plot(voyage_melted["S"], mdf.fittedvalues, "x-")
plt.scatter(voyage_melted["S"], voyage_melted["N"], color="red")
plt.xlabel("Speed")
plt.ylabel("Voyage Distance")
# plt.show()

voyage_melted["S"] = voyage_melted["S"].apply(lambda x: x)
md = smf.mixedlm("N ~ 0 + S", voyage_melted, groups=voyage_melted["Name"])
mdf = md.fit()
print(mdf.summary())

voyage_melted["predicted_N"] = mdf.fittedvalues
hours_year = 365 * 24
unique_names = voyage_melted["Name"].unique()

rows = - (-len(unique_names) // 2)
fig, axes = plt.subplots(rows, 2, figsize=(rows * 4, 8))
for idx, name in enumerate(unique_names):
    row = idx // 2
    col = idx % 2
    subset = voyage_melted[voyage_melted["Name"] == name]
    axes[row, col].plot(subset["S"], subset["N"], 'x', label=f"Actual {name}")
    axes[row, col].plot(subset["S"], subset["predicted_N"], '-', label=f"Predicted {name}")
    YT = 365 * 24
    VT = subset["H"].iloc[0]
    v = subset["S"].iloc[0]
    axes[row, col].plot(subset["S"], subset.S.apply(lambda x: (0.9 * YT + 0.1 * VT) / ((0.9 * (YT / VT - 1) / v) + 1 / x)),
                        '-', label=f"Theoretical {name}")
    axes[row, col].set_xlabel("Speed")
    axes[row, col].set_ylabel("Voyage distance")
    axes[row, col].legend()
    axes[row, col].set_title(f"{name}")

plt.tight_layout()
plt.show()
