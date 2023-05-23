import numpy as np
from settlement import Settlement


class MeanField:
    def __init__(self, vessels, route, global_market, q=0.05):
        self.vessels = vessels
        self.route = route
        self.global_market = global_market

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

        for vessel in self.vessels:
            settlement = Settlement(vessel, route, global_market)
            T = vessel.hours_2021
            D = route.distance
            u_actual = vessel.speed_2021

            p = route.freight_rate * 0.95 * vessel.capacity * T * u_actual / D
            a = 5 * p
            b = 4 * p / u_actual

            cf = -settlement.fuel_cost() / (0.95 * vessel.capacity)
            l = cf * 0.95 * vessel.capacity * T / D
            gamma = u_actual**2 / (2 * cf * 0.95 * vessel.capacity * T * u_actual / D)
            u0 = gamma * (a - l) / (1 + gamma * b)

            cii = vessel.cii_score_2021
            cii_e = 7.65
            delta_max = (np.sqrt(cii_e / cii) - 1) * u0

            x = self.theta_ - delta_max
            lam = 1

            p0 = a - b * u0
            c0 = l * u0 + 1.0 / 2 / gamma * u0**2
            v = (-c0 + p0 * u0) * 0.1

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
            + np.sqrt(
                self.gamma_**2 * (self.p0_ - self.b_ * e_delta - self.l_) ** 2
                - 2 * self.gamma_ * self.v_
            )
        )
        print("x_hat\t", x_hat[:5])
        return x_hat

    def delta_hat(self, x, theta, e_delta):
        # Equation (16)
        x_hat = self.x_estimator(theta=theta, e_delta=e_delta)

        return np.where(
            x < theta,
            np.minimum((theta - x) / self.lam_, -self.gamma_ * self.b_ * e_delta),
            np.where(x <= x_hat, -(x - theta) / self.lam_, np.zeros(len(x))),
        )

    def theta_hat(self, x, delta):
        # Equation (9)
        return np.percentile(x + self.lam_ * delta, 100 - self.q_ * 100)
        # return np.sort(x + self.lam_ * delta)[int(len(x) * (1 - self.q_)) - 1]

    def one_step(self):
        # From previous distribution of x, a new theta can be estimated
        self.theta_ = self.theta_hat(self.x_, self.delta_)
        print("theta:\t", self.theta_)

        # Based on new theta, every vessel changes its speed (delta)
        self.delta_ = self.delta_hat(self.x_, self.theta_, self.e_delta_)
        print("delta:\t", self.delta_[:5])

        y = self.x_ + self.lam_ * self.delta_
        print("y:\t", y[:5])

        # Then we need to recalculate the E[delta]
        self.e_delta_ = np.mean(self.delta_)
        print("e_delta:\t", self.e_delta_)

        return

    def simulate(self, tol=1e-5, max_iter=20):
        errs = []
        delta0 = []
        n = len(self.x_)
        np.set_printoptions(precision=14)

        for i in range(1, max_iter + 1):
            delta_current = self.delta_

            print(f"iteration {i}:")
            self.one_step()
            print()

            err = (
                np.sum((self.x_ + self.lam_ * self.delta_) >= self.theta_) / n - self.q_
            )

            errs.append(err)
            delta0.append(self.delta_[0])
            if (
                np.abs(err) < tol
                and np.mean((delta_current - self.delta_) ** 2) <= 1e-4
            ):
                print(f"Tolerance satisfied at iteration {i}!")
                break
        return errs, delta0
