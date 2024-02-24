import numpy as np
from scipy import stats as st
from tqdm import tqdm

from utils import simulate, plot_trace


class BayesProbit:

    def __init__(
        self,
        n_draw: int = 10000,
        n_burn: int = 10000,
        prior_mu: float = 0.0,
        prior_var: float = 100.0,
        seed: int = 0,
    ):
        self.n_draw_ = n_draw
        self.n_burn_ = n_burn
        self.prior_mu_ = prior_mu
        self.prior_var_ = prior_var
        self.seed_ = seed

        self.n_iter_ = self.n_draw_ + self.n_burn_
        self.rng_ = np.random.default_rng(self.seed_)

    def fit(self, X: np.ndarray, y: np.ndarray):

        n_beta = X.shape[1] + 1
        n_sample = X.shape[0]
        bv = self.prior_mu_ / self.prior_var_

        # 重新调整，将y=1的放在前面，y=0的放在后面
        ind_1 = y == 1
        ind_0 = np.logical_not(ind_1)
        y = np.concatenate([y[ind_1], y[ind_0]], axis=0)
        X = np.concatenate([X[ind_1, :], X[ind_0, :]], axis=0)
        ny1 = ind_1.sum()

        dmat = np.concatenate([np.ones((n_sample, 1)), X], axis=1)
        XX = dmat.T @ dmat
        XX_ = XX + np.diag([1 / self.prior_var_] * n_beta)
        XX_inv = np.linalg.inv(XX_)

        self.hist_ = {
            "w": np.zeros((self.n_iter_, n_beta)),
            "z": np.zeros((self.n_iter_, n_sample)),
        }

        def _sample_z(z_mu, ny1):
            z_i_1 = st.truncnorm.rvs(
                a=-z_mu[:ny1], b=np.inf, loc=z_mu[:ny1], random_state=self.rng_
            )
            z_i_0 = st.truncnorm.rvs(
                a=-np.inf,
                b=-z_mu[ny1:],
                loc=z_mu[ny1:],
                random_state=self.rng_,
            )
            z_i = np.concatenate([z_i_1, z_i_0], axis=0)
            return z_i

        z_i = 0  # placeholder
        for i in tqdm(range(self.n_iter_)):
            if i == 0:
                w_i = (
                    self.rng_.normal(size=n_beta) * np.sqrt(self.prior_var_)
                    + self.prior_mu_
                )
            else:
                B = XX_inv @ (bv + dmat.T @ z_i)
                w_i = st.multivariate_normal.rvs(
                    mean=B, cov=XX_inv, random_state=self.rng_
                )
            z_mu = dmat @ w_i
            z_i = _sample_z(z_mu, ny1)

            self.hist_["w"][i, :] = w_i
            self.hist_["z"][i, :] = z_i

        self.beta_ = self.hist_["w"][self.n_burn_ :, :].mean(axis=0)
        self.beta_std_ = self.hist_["w"][self.n_burn_ :, :].std(axis=0)


def main():
    seed = 0

    dat = simulate(
        n=1000, sfunc="probit", beta=np.array([0, -3, -1, 0, 1, 3]), seed=seed
    )

    model = BayesProbit(prior_mu=0, prior_var=100.0, seed=seed)
    model.fit(dat["X"], dat["y"])

    print(model.beta_, model.beta_std_)
    plot_trace("./trace.png", dat["beta"], model.hist_)


if __name__ == "__main__":
    main()
