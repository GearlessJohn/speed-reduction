import random

import matplotlib.pyplot as plt
import numpy as np

import settlement
import vessel


class MeanField:
    def __init__(self, vessels, route, global_env, q, value_exit, year):
        self.vessels = vessels
        self.route = route
        self.global_env = global_env
        self.value_exit_ = value_exit
        self.theta_ = 10
        self.e_delta_ = 0
        self.q_ = q

        self.a_ = []
        self.b_ = []
        self.l_ = []
        self.gamma_ = []
        self.x_ = []
        self.lam_ = []
        self.u0_ = []
        self.p0_ = []
        self.v_ = []
        self.delta_ = []

        for i in range(len(self.vessels)):
            vessel = vessels[i]
            t = vessel.hours_2021
            d = route.distance
            u_actual = vessel.speed_2021
            stm = settlement.Settlement(vessel, route, global_env)

            p = route.freight_rates[year] * 0.95 * vessel.capacity * t / d
            a = 5 * p
            b = 4 * p / u_actual

            cf = stm.fuel_cost(
                speed=vessel.speed_2021, saving=0.0, power=3.0, year=year
            ) / (0.95 * vessel.capacity)
            l = cf * 0.95 * vessel.capacity * t / d
            gamma = u_actual ** 2 / (2 * cf * 0.95 * vessel.capacity * t * u_actual / d)
            u0 = gamma * (a - l) / (1 + gamma * b)

            cii = vessel.cii_score_2021
            cii_e = 7.65
            # delta_max = (np.sqrt(cii_e / cii) - 1) * u0
            delta_max = (cii_e / cii - 1) * u0

            x = self.theta_ - delta_max
            lam = 1

            p0 = a - b * u0
            c0 = l * u0 + 1.0 / 2 / gamma * u0 ** 2
            v = (-c0 + p0 * u0) * self.value_exit_

            self.a_.append(a)
            self.b_.append(b)
            self.l_.append(l)
            self.gamma_.append(gamma)
            self.x_.append(x)
            self.lam_.append(lam)
            self.u0_.append(u0)
            self.p0_.append(p0)
            self.v_.append(v)
            self.delta_.append(0)

        self.a_ = np.array(self.a_)
        self.b_ = np.array(self.b_)
        self.l_ = np.array(self.l_)
        self.gamma_ = np.array(self.gamma_)
        self.x_ = np.array(self.x_)
        self.lam_ = np.array(self.lam_)
        self.u0_ = np.array(self.u0_)
        self.p0_ = np.array(self.p0_)
        self.v_ = np.array(self.v_)
        self.delta_ = np.array(self.delta_)

    def x_estimator(self, theta, e_delta):
        # Equation (15)
        x_hat = theta + self.lam_ * (
                self.u0_
                - self.gamma_ * (self.p0_ - self.b_ * e_delta - self.l_)
                + np.sqrt(self.gamma_ ** 2 * (self.p0_ - self.b_ * e_delta - self.l_) ** 2
                          - 2 * self.gamma_ * self.v_)
        )
        print("x_hat\t", x_hat[:5])
        return x_hat

    def delta_hat(self, x, theta, e_delta, binary=False, lim=0.3):
        # Equation (16)
        x_hat = self.x_estimator(theta=theta, e_delta=e_delta)

        res = np.where(
            x <= theta,
            np.minimum((theta - x) / self.lam_, -self.gamma_ * self.b_ * e_delta),
            np.where(x <= x_hat, -(x - theta) / self.lam_, np.zeros(len(x))),
        )
        if binary:
            return np.where(
                res < -self.vessels[0].speed_2021 * 0.05,
                -self.vessels[0].speed_2021 * 0.05,
                self.delta_,
            )
        return np.where(
            np.abs(res) > self.vessels[0].speed_2021 * lim,
            self.vessels[0].speed_2021 * lim * np.sign(res),
            res,
        )

    def theta_hat(self, x, delta):
        # Equation (9)
        # return np.percentile(x + self.lam_ * delta, 100 - self.q_ * 100)
        return np.sort(x + self.lam_ * delta)[int(len(x) * (1 - self.q_))]

    def one_step(self, binary=False):
        # From previous distribution of x, a new theta can be estimated
        self.theta_ = self.theta_hat(self.x_, self.delta_)
        print("theta:\t", self.theta_)
        print(
            "proportion:",
            np.sum((self.x_ + self.lam_ * self.delta_) > self.theta_) / len(self.x_),
        )
        # Based on new theta, every vessel changes its speed (delta)
        self.delta_ = self.delta_hat(
            x=self.x_, theta=self.theta_, e_delta=self.e_delta_, binary=binary
        )
        print("delta:\t", self.delta_[:5])

        y = self.x_ + self.lam_ * self.delta_
        print("y:\t", y[:5])

        # Then we need to recalculate the E[delta]
        self.e_delta_ = np.mean(self.delta_)
        print("e_delta:\t", self.e_delta_)

        return

    def simulate(self, tol=1e-5, max_iter=20, binary=False):
        errs = []
        delta0 = []
        pis = []
        n = len(self.x_)
        np.set_printoptions(precision=14)

        err = np.sum((self.x_ + self.lam_ * self.delta_) > self.theta_) / n

        errs.append(err)
        delta0.append(self.delta_[0])
        pis.append(
            (
                    (self.p0_ - self.b_ * self.delta_) * (self.u0_ + self.delta_)
                    - self.l_ * (self.u0_ + self.delta_)
                    - 1 / 2 / self.gamma_ * (self.u0_ + self.delta_) ** 2
            )[0]
        )

        for i in range(1, max_iter + 1):
            delta_current = self.delta_

            print(f"iteration {i}:")
            self.one_step(binary=binary)
            print()

            err = np.sum((self.x_ + self.lam_ * self.delta_) > self.theta_) / n

            errs.append(err)
            delta0.append(self.delta_[0])
            pis.append(
                (
                        (self.p0_ - self.b_ * self.delta_) * (self.u0_ + self.delta_)
                        - self.l_ * (self.u0_ + self.delta_)
                        - 1 / 2 / self.gamma_ * (self.u0_ + self.delta_) ** 2
                )[0]
            )
            if (
                    np.abs(err - self.q_) <= tol
                    and np.mean((delta_current - self.delta_) ** 2) <= 1e-4
            ):
                print(f"Tolerance satisfied at iteration {i}!")
                break
        return errs, delta0, pis


def vessels_sampling(row, global_env, num, pcts=None):
    """Create a sample of num vessels from an origin vessel"""
    if pcts is None:
        pcts = [0.15, 0.2, 0.3, 0.2, 0.15]
    assert np.abs(np.sum(pcts) - 1.0) <= 1e-10, "Wrong distribution of CII scores"

    vessels_virtual = [vessel.Vessel(row) for _ in range(num)]

    fronts = np.insert(
        global_env.cii_fronts(
            vessels_virtual[0].vessel_type,
            vessels_virtual[0].sub_type,
            vessels_virtual[0].dwt,
            year=2021,
        ),
        0,
        1,
    )
    fronts = np.append(fronts, [1.5 * fronts[-1]])

    ciis = []
    for i in range(len(pcts)):
        ciis.extend(np.random.uniform(fronts[i], fronts[i + 1], int(num * pcts[i])))

    random.shuffle(ciis)
    for i in range(num):
        vessels_virtual[i].cii_score_2021 = ciis[i]

    return vessels_virtual, ciis, fronts


def mf(num, data_vessels, env, route, q=0.15, value_exit=0.5, binary=False, year=0):
    # Launch Model
    # Create a virtual sample of vessels with same information

    vessels_virtual, ciis, fronts = vessels_sampling(
        row=data_vessels.iloc[5], global_env=env, num=num
    )

    mf = MeanField(vessels_virtual, route, env, q=q, value_exit=value_exit, year=year)

    # Simulate
    errs, delta0, pis = mf.simulate(tol=0.01, max_iter=15, binary=binary)

    fig, axs = plt.subplots(5, figsize=(8, 8))
    plt.subplots_adjust(hspace=0.7)
    axs[0].hist(ciis, bins=50)
    axs[0].axvline(x=fronts[1], color="green")
    axs[0].axvline(x=fronts[2], color="yellow")
    axs[0].axvline(x=fronts[3], color="purple")
    axs[0].axvline(x=fronts[4], color="red")
    axs[0].set_title("Distribution of CII")

    axs[1].plot(errs)
    axs[1].set_title("Proportion of vessels with y>theta")
    axs[1].axline(xy1=(0, mf.q_), slope=0, c="red")

    axs[2].plot(delta0)
    axs[2].set_title("Speed variation of the first vessel")

    axs[3].plot(pis)
    axs[3].set_title("Profit of the first vessel")

    axs[4].scatter(range(len(mf.x_)), mf.x_ + mf.lam_ * mf.delta_, s=0.7)
    axs[4].axline(xy1=(0, mf.theta_), slope=0, c="red")
    axs[4].set_title("Distribution of final y")

    fig.suptitle(
        f"{len(mf.x_):d} vessels, q: {mf.q_:.2f}, exit value rate: {mf.value_exit_:.2f}"
    )

    plt.show()
    # fig.savefig(
    #     f"./fig/meanfield-{len(mf.x_):d} vessels-exit value rate {mf.value_exit_:.1f}.png"
    # )
