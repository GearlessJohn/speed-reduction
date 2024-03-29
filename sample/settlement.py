import matplotlib.pyplot as plt
import numpy as np

import vessel


class Settlement:
    """Route speed optimization center.

    For a given vessel, a route, and market policy information, calculate the speed that maximizes profit.
    After the speed optimization, calculate the change compared with 2021 in speed and carbon emission, etc.

    Attributes:
        vessel (Vessel): Selected vessel.
        route (Route): Route matching for the selected vessel.
        global_env (GlobalEnv): Current global markets and policies.

    """

    def __init__(self, vessel, route, global_env):
        # Vessel type and route type must be the same.
        assert (
                vessel.vessel_type == route.route_type
        ), "Route and vessel types do not match!"
        self.vessel = vessel
        self.route = route
        self.global_env = global_env

    def fuel_cost(self, speed, saving, power, year):
        """Calculate the fuel cost of one voyage.

        Referring to the 2021 data, calculate the fuel consumption at the new speed
        and multiply by the price of different fuels to sum up.

        Args:
            speed (float): Vessel's actual navigation speed, in nautical miles per hour.
            saving (float): Fuel saving factor after modification, in percentage.
            power (float): The relationship between fuel consumption per unit distance travelled and speed,
                is generally between 2 and 3.
            year (int): The year in which the calculation was performed. Used to obtain fuel prices.

        Returns:
             float: The fuel cost in USD for one voyage of the selected route.

        """
        assert year < 2000, "The year should start from 0."
        return (
                (
                        self.vessel.hfo_2021 * self.global_env.ifo380_prices[year]
                        + self.vessel.lfo_2021 * self.global_env.vlsifo_prices[year]
                        + self.vessel.diesel_2021 * self.global_env.mgo_prices[year]
                        + self.vessel.lng_2021 * self.global_env.lng_prices[year]
                )
                / self.vessel.distance_2021
                * (speed / self.vessel.speed_2021) ** power
                * self.route.distance
                * (1 - saving)
        )

    def retrofitting(self, speed, power, year):
        """Evaluate the vessel for retrofitting.

        Retrofitting has a fixed cost and has the effect of reducing fuel consumption by a percentage.
        Modifications include engine, propeller, and bulbous.
        Determine if the modification is economically viable based on the fuel consumed for the modification.
        Return the total cost and fuel saving factor.

        Args:
            speed (float): Vessel's actual navigation speed, in nautical miles per hour.
            power (float): The relationship between fuel consumption per unit distance travelled and speed,
                is generally between 2 and 3.
            year (int): The year in which the calculation was performed.

        Returns:
            cost_retrofit (float): Total cost of retrofitting, in USD.
            saving (float): Percentage of fuel consumption that can be reduced by retrofitting, between 0% and 100%.

        """
        fuel_cost = self.fuel_cost(speed=speed, saving=0.0, power=power, year=year)
        # Consider now the possible retrofitting measures
        years_left = 25 - self.vessel.age
        engine = 10000.0 / years_left  # 2.5%
        propeller = 450000.0 / years_left  # 7.0%
        bulbous = 550000.0 / years_left  # 4.0%

        cost_retrofit = 0.0
        saving = 0.0
        if -fuel_cost * 0.025 > engine:
            cost_retrofit += engine
            saving += 0.025
        if -fuel_cost * 0.07 > propeller:
            cost_retrofit += propeller
            saving += 0.07
        if -fuel_cost * 0.04 > bulbous:
            cost_retrofit += bulbous
            saving += 0.04
        return cost_retrofit, saving

    def unit_fuel_cost(self, speed, saving, power, year):
        """Return the fuel cost per unit."""

        cost = self.fuel_cost(speed=speed, saving=saving, power=power, year=year) / (
                self.vessel.capacity * self.route.utilization_rate
        )
        return cost

    def cii_class(self, speed, power, year):
        """Return the CII rating class of the vessel based on actual speed.

        Based on 2021 data, CII ratings are determined by a combination of speed and year.

        Args:
            speed (float): Vessel's actual navigation speed, in nautical miles per hour.
            power (float): The relationship between fuel consumption per unit distance travelled and speed,
                is generally between 2 and 3.
            year (int): The year in which the calculation was performed, 0 means 2023, 1 means 2024 ...

        Returns:
            rating (str): A letter from A to E indicating the CII rating class of the vessel

        """
        assert year < 2000, "The year should start from 0."
        rating = self.global_env.cii_class(
            self.vessel.cii_score_2021 * (speed / self.vessel.speed_2021) ** power,
            self.vessel.vessel_type,
            self.vessel.sub_type,
            self.vessel.dwt,
            year=2023 + year,
        )
        return rating

    def operation_ghg(self, speed, saving, power):
        """Return the CO2 emission of one voyage."""
        return (
                self.vessel.co2_2021
                / self.vessel.distance_2021
                * self.route.distance
                * (speed / self.vessel.speed_2021) ** power
                * (1 - saving)
        )

    def carbon_tax(self, speed, saving, power, year):
        """Return the carbon tax costs based on speed and year for one voyage."""
        return (
                self.operation_ghg(speed=speed, saving=saving, power=power)
                * self.global_env.carbon_tax_rates[year]
        )

    def operation_cost(self):
        """Return the expenses associated with vessel operations for one voyage."""

        # The operation cost is calculated by the hypothesis of the proportion of fuel cost to total cost.
        return (
                self.fuel_cost(speed=self.vessel.speed_2021, saving=0.0, power=3.0, year=0)
                * (1 - self.route.fuel_ratio)
                / self.route.fuel_ratio
        )

    def freight(self, year):
        """Return the revenue generated from the transportation of goods for one voyage."""
        return (
                self.vessel.capacity
                * self.route.utilization_rate
                * self.route.freight_rates[year]
        )

    def voyage_hours(self, speed, acc):
        """Return the voyage hours in one year.

        Solution 1: Assume that sailing time is independent of speed, then it is equal to the value in 2021.
        Solution 2: Waiting and loading/unloading time are assumed to be proportional to the number of voyages.

        Args:
            speed (float): Vessel's actual navigation speed, in nautical miles per hour.
            acc (bool): If True, the answer is calculated by solution 2, else solution 1.

        Returns:
            float: The voyage hours in one year, between 0 and 365*24.

        """
        if acc:
            h0 = self.vessel.hours_2021
            p0 = (365 * 24 - h0) * 0.9
            return (h0 + p0) / (1 + p0 * speed / (h0 * self.vessel.speed_2021))
        else:
            return self.vessel.hours_2021

    def nmb_trip(self, speed, acc):
        """Return the number of possible voyages in one year."""
        return self.voyage_hours(speed=speed, acc=acc) * speed / self.route.distance

    def profit_trip(self, speed, power, retrofit, year):
        """Return the profit of one voyage for given speed and year."""
        res = 0.0
        saving = 0.0
        if retrofit:
            cost_retrofit, saving = self.retrofitting(
                speed=speed, power=power, year=year
            )
            res -= cost_retrofit
        res -= self.fuel_cost(speed=speed, saving=saving, power=power, year=year)
        res -= self.carbon_tax(speed=speed, saving=saving, power=power, year=year)
        res -= self.operation_cost()
        res += self.freight(year=year)

        return res

    def annual_emission(self, speed, saving, power, acc):
        return self.operation_ghg(
            speed=speed, saving=saving, power=power
        ) * self.nmb_trip(speed=speed, acc=acc)

    def annual_profit(self, speed, power, retrofit, year, acc):
        res = self.profit_trip(
            speed=speed, power=power, retrofit=retrofit, year=year
        ) * self.nmb_trip(speed=speed, acc=acc)

        return res

    def plot_annual_profit(self, retrofit, power, year, acc, pr=False, plot=True):
        """Return the optimal speed to maximize the annual profit.

        The optimal speed was calculated from 7 to 24 in intervals of 0.01 knots.

        Args:
            retrofit (bool): Whether consider the retrofitting, else not.
            power (float): The relationship between fuel consumption per unit distance travelled and speed,
              is generally between 2 and 3.
            year (int): The year in which the calculation was performed.
            acc (bool):Whether consider the change of voyages hours, else the voyage hours are constant.
            pr (bool): Whether print all the information and plot, else not.

        Returns:
            v_best (float): The speed that maximizes the annual profit.

        """

        # vs is the range of speeds to be explored.
        vs = np.arange(7, 24, 0.01)

        # The profit and emission of each speed in vs is calculated.
        profits = (
                np.array(
                    [
                        self.annual_profit(
                            speed=vs[j], power=power, retrofit=retrofit, acc=acc, year=year
                        )
                        for j in range(len(vs))
                    ]
                )
                / 1e6
        )
        emissions = np.array(
            [
                self.annual_emission(
                    speed=vs[j],
                    saving=self.retrofitting(speed=vs[j], year=year, power=3)[1]
                    if retrofit
                    else 0.0,
                    power=power,
                    acc=acc,
                )
                for j in range(len(vs))
            ]
        )
        # The best speed is the one that maximize the annual profit.
        v_best = vs[np.argmax(profits)]
        profit_best = np.max(profits)
        saving_best = (
            self.retrofitting(speed=v_best, power=power, year=year)[1]
            if retrofit
            else 0.0
        )

        # Present all the information.
        if pr:
            print("-" * 60)
            print("\tVessel Name:\t", self.vessel.name)
            print("\tType:\t\t", self.vessel.vessel_type)
            print("\tSub-type:\t", self.vessel.sub_type)
            print("\tCapacity:\t", self.vessel.capacity, self.vessel.unit)
            print()

            print("\tRoute:\t\t", self.route.name)
            print("\tDistance:\t", self.route.distance, "knots")
            print(
                "\tFreight Rate:\t",
                self.route.freight_rates[year],
                f"$/{self.vessel.unit}",
            )
            print("\tCarbon Tax:\t", self.global_env.carbon_tax_rates[year], "$/ton")
            print("\tUtilization:\t", self.route.utilization_rate * 100, "%")
            print("\tRetrofit:\t", retrofit)
            print()

            print("\t2021 Speed:\t", f"{self.vessel.speed_2021:.2f} knots")
            print("\tOptimal Speed:\t", f"{v_best:.2f} knots")
            fc = self.unit_fuel_cost(
                speed=v_best, saving=saving_best, power=power, year=year
            )
            print(
                "\tFuel cost:\t",
                f"{fc:.2f} $/{self.vessel.unit}",
            )
            oc = self.operation_cost() / (
                    self.vessel.capacity * self.route.utilization_rate
            )
            print(
                "\tOperation cost:\t",
                f"{oc:.2f} $/{self.vessel.unit}",
            )
            print(
                "\tProfitability:\t",
                f"{(self.route.freight_rates[year] + oc + fc) / self.route.freight_rates[year] * 100:.2f} %",
            )
            print("\tAnnual Profit:\t", f"{profit_best:.2f} M $")
            print()
            print(
                "\tSpeed Variation:\t",
                f"{(v_best - self.vessel.speed_2021) / self.vessel.speed_2021 * 100:+.2f} %",
            )
            print(
                "\tEmission Variation:\t",
                f"{(emissions[np.argmax(profits)] - self.annual_emission(speed=self.vessel.speed_2021, saving=0.0, power=power, acc=acc)) / self.annual_emission(speed=self.vessel.speed_2021, saving=0.0, power=power, acc=acc) * 100:+.2f} %",
            )
            print(
                "\t2021 CII class:\t\t",
                f"{self.vessel.cii_class_2021}",
            )
            print(
                "\tCurrent CII class:\t",
                f"{self.cii_class(speed=v_best, power=power, year=year)}",
            )
            print("-" * 60)

        if plot:
            # Plot the profits as function of speed.
            fig, ax = plt.subplots()
            ax.plot(vs, profits, label="profit", color="blue")
            ax.set_xlabel("Vessel Speed (knot)")
            ax.set_ylabel("Profit (dollar)", color="blue")
            ax.axvline(x=v_best, ymax=profit_best, c="red", linestyle="--")
            ax.axvline(
                x=self.vessel.speed_2021,
                ymax=profit_best,
                c="grey",
                linestyle="-.",
            )
            ax.annotate(
                f"Optimal Speed={v_best:0.2f} knots",
                xy=(v_best, profit_best),
                xytext=(v_best, profit_best * 0.95),
                arrowprops=dict(facecolor="red"),
            )
            ax.legend(loc="upper left")

            ax1 = ax.twinx()
            ax1.plot(vs, emissions, label="emission", color="green")
            ax1.set_ylabel("CO2 Emission (ton)", color="green")
            ax1.legend(loc="upper right")

            fig.suptitle(
                f"{2023 + year:d} annual result, Carbon Tax: {self.global_env.carbon_tax_rates[year]}, Retrofit: {retrofit} "
            )
            plt.show()

        return v_best

    @staticmethod
    def cii_profits(profits, cii_class):
        """Return the optimal speed choices for 2023-2026, and the corresponding total profit.

           This function operates by iterating over each possible combination of indices
            and applying certain rules based on the cii_class.
           If the cii_class is 'D' for 3 consecutive years, or if the cii_class is 'E' for one year,
           it zeroes out all profits for the following years.
           Finally, it sums all profits and returns the indices that yield the maximum total profit and the maximum
           total profit itself.

        Args:
            profits (numpy.ndarray): 2D array of profits, where each element at index (i,j)
                                    represents the profit at year i with speed choice j.
            cii_class (numpy.ndarray): 2D array of strings, where each element at index (i,j)
                                        represents the cii_class at year i with speed choice j.

        Returns:
            tuple: A tuple of two elements. The first element is a tuple of indices
                    where the maximum total profit is found. The second element is the
                    maximum total profit itself.

        """
        m = profits.shape[1]

        # Create an array of profit indices
        indices = np.arange(m)

        # Create a grid of all combinations of indices
        I0, I1, I2, I3 = np.meshgrid(indices, indices, indices, indices, indexing="ij")

        # Create a condition mask for cii_class values where all first three indices are "D"
        mask_D = (
                (cii_class[0, I0] == "D")
                & (cii_class[1, I1] == "D")
                & (cii_class[2, I2] == "D")
        )

        # Create masks for cii_class values where any index is "E"
        mask_E0 = cii_class[0, I0] == "E"
        mask_E1 = cii_class[1, I1] == "E"
        mask_E2 = cii_class[2, I2] == "E"

        # Set the profits to zero where the cii_class is "E"
        profits_E0 = profits[0, I0]
        profits_E1 = np.where(mask_E0, 0.0, profits[1, I1])
        profits_E2 = np.where(mask_E0 | mask_E1, 0.0, profits[2, I2])
        profits_E3 = np.where(
            mask_E0 | mask_E1 | mask_E2, 0.0, profits[3, I3]
        )

        # Calculate total profits using the indices and the condition mask
        total_profit = (
                profits_E0 + profits_E1 + profits_E2 + np.where(mask_D, 0.0, profits_E3)
        )
        res = np.unravel_index(np.argmax(total_profit, axis=None), total_profit.shape)
        return res, total_profit[res]

    @staticmethod
    def cii_profits_reverse(profits, cii_class):
        """Return the optimal speed choices for 2023-2026, and the corresponding total profit.

            This function uses a reverse-order algorithm to find the optimal solution.
            Finally, it sums all profits and returns the indices that yield the maximum total profit and the maximum
            total profit itself.

         Args:
             profits (numpy.ndarray): 2D array of profits, where each element at index (i,j)
                                     represents the profit at year i with speed choice j.
             cii_class (numpy.ndarray): 2D array of strings, where each element at index (i,j)
                                         represents the cii_class at year i with speed choice j.

         Returns:
             tuple: A tuple of two elements. The first element is a tuple of indices
                     where the maximum total profit is found. The second element is the
                     maximum total profit itself.

         """
        m = profits.shape[1]
        zeros = np.zeros(m)

        i3 = np.argmax(profits[3])
        P3 = np.where(cii_class[2] == "E", zeros, profits[3][i3])

        i2 = np.argmax(profits[2] + P3)
        P2 = np.where(cii_class[1] == "E", zeros, P3[i2] + profits[2][i2])

        i1 = np.argmax(profits[1] + P2)
        P1 = np.where(
            cii_class[0] == "E",
            zeros,
            P2[i1] + profits[1][i1]
        )

        i0 = np.argmax(profits[0] + P1)

        res = np.array([i0, i1, i2, i3])
        total_profit = profits[0][i0] + P1[i0]

        if (
                (cii_class[2][i2] == "D")
                & (cii_class[1][i1] == "D")
                & (cii_class[0][i0] == "D")
        ):
            last_c = np.zeros(3, dtype="int")
            last_c[0] = np.where(cii_class[0] == "C")[0][-1]
            last_c[1] = np.where(cii_class[1] == "C")[0][-1]
            last_c[2] = np.where(cii_class[2] == "C")[0][-1]

            total_profit_ddd = np.zeros(4)
            total_profit_ddd[0] = total_profit + (profits[0][last_c[0]] - profits[0][i0])
            total_profit_ddd[1] = total_profit + (profits[1][last_c[1]] - profits[1][i1])
            total_profit_ddd[2] = total_profit + (profits[2][last_c[2]] - profits[2][i2])
            total_profit_ddd[3] = total_profit - profits[3][i3]

            solution = np.argmax(total_profit_ddd)

            if solution <= 2:
                res[solution] = last_c[solution]

            total_profit = total_profit_ddd[solution]

        return res, total_profit

    def optimization(self, retrofit, power, years, acc, cii_limit=True, pr=False, plot=True):
        """Perform optimization on vessel speed for maximum 4-year profit, and calculates related metrics.

        The function calculates maximum 4-year profit for a range of speeds for each year and
        identifies the optimal speed combination that maximizes profit. Also calculates retrofitting savings,
        annual emissions and CII class. Can optionally plot the results.

        Args:
            retrofit (bool): Whether consider the retrofitting, else not.
            power (float): The relationship between fuel consumption per unit distance travelled and speed,
              is generally between 2 and 3.
            years (List[int]): List of years for which the optimization is performed.
            acc (bool): Whether consider the change of voyages hours.
            cii_limit (bool, optional): Whether apply the Carbon Intensity Indicator (CII) limit. Default is True.
            pr (bool): If True, prints and plots the optimization results. Default is False.

        Returns:
            tuple: A tuple containing arrays of optimal speeds, profits, emissions, and
                   CII classes for each year.

        """
        n = len(years)
        m = 601
        speed_ini = self.vessel.speed_2021
        vs = speed_ini + np.linspace(-3, 3, m)

        # Initialize arrays to hold profits and cii_class for each speed and year
        profits = np.empty((n, m))
        cii_class = np.empty((n, m), dtype=object)

        # Calculate profits and cii_class for each speed and year
        for i in years:
            for j in range(m):
                profits[i, j] = self.annual_profit(
                    speed=vs[j], power=power, retrofit=retrofit, year=i, acc=acc
                )
                cii_class[i, j] = self.cii_class(speed=vs[j], power=power, year=i)

        # Apply CII limit if specified
        if cii_limit:
            # best, profit_max = self.cii_profits(profits=profits, cii_class=cii_class)
            best, profit_max = self.cii_profits_reverse(profits=profits, cii_class=cii_class)
        else:
            best = [np.argmax(profits[i]) for i in years]
            profit_max = np.sum([profits[i, best[i]] for i in years])

        profits_best = np.array([profits[i, best[i]] for i in years])
        v_best = np.array([vs[best[i]] for i in years])

        # Calculate retrofit savings and emissions for optimal speed
        savings = [
            self.retrofitting(speed=v_best[i], power=power, year=i)[1] for i in years
        ]
        emissions_best = [
            self.annual_emission(
                v_best[i],
                saving=(savings[i] if retrofit else 0.0),
                power=power,
                acc=acc,
            )
            for i in years
        ]

        # Adjust profits if the sum of profits is equal to max profit
        if np.sum(profits_best[:-1]) == profit_max:
            profits_best[3] = 0.0
        cii_best = [cii_class[i, best[i]] for i in years]

        # Print and plot results if specified
        if pr:
            # print statements here
            print(self.vessel.name)
            print("Max profit:", profit_max / 1e6)
            print("Optimal Speed:", v_best)
            print("Optimal Profit:", profits_best)
            print("Emission:", emissions_best)
            print("CII Class:", cii_best)

        if plot:
            # plot code here
            fig, ax = plt.subplots()
            ax.plot(2023 + np.array(years), v_best, c="blue")
            ax.set_xlabel("Year")
            ax.set_xticks(2023 + np.array(years))
            ax.set_ylabel("Speed (knot)", color="blue")
            # ax.legend()

            ax1 = ax.twinx()
            ax1.plot(2023 + np.array(years), profits_best, c="green")
            ax1.set_ylabel("Profit (M$)", color="green")
            # ax1.legend()
            plt.show()
        return v_best, profits_best, emissions_best, cii_best


def settle(i, data_vessels, env, route, power, retrofit, year, pr, plot=True, acc=True):
    # Creating a list of Vessel objects
    vessels = [vessel.Vessel(row) for _, row in data_vessels.iterrows()]

    stm = Settlement(
        vessel=vessels[i],
        route=route,
        global_env=env,
    )
    v_best = .0
    if pr:
        v_best = stm.plot_annual_profit(retrofit=retrofit, power=power, year=year, pr=pr, plot=plot, acc=acc)
    return stm, v_best
