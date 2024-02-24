from typing import Literal
import warnings

import numpy as np
from scipy import stats as st
from scipy.special import log_expit, expit

# from tqdm import tqdm

from utils import simulate  # , plot_trace


def newton_estimate(
    designX: np.ndarray,
    y: np.ndarray,
    prior_mu: float = 0.0,
    prior_var: float = 100.0,
    max_iter: int = 100,
    w_init: float | np.ndarray = 0.0,
    lr: float | None = 1.0,
    tolerance: float = 1e-5,
) -> dict:
    assert isinstance(w_init, (float, np.ndarray))
    if isinstance(w_init, float):
        w_init = np.full(designX.shape[1], w_init)
    if lr is None:
        lr = 1 / designX.shape[0]

    prior_var_inv = np.diag(np.full(designX.shape[1], 1 / prior_var))
    w = w_init
    for _ in range(max_iter):
        p = expit(designX @ w)
        grad = designX.T @ (p - y) + (w - prior_mu) / prior_var

        if np.sqrt((grad ** 2).sum()) < tolerance:
            break

        hessian = designX.T @ np.diag(p * (1 - p)) @ designX + prior_var_inv
        w -= lr * np.linalg.inv(hessian) @ grad
    else:
        warnings.warn("newton method dose not converge.")

    p = expit(designX @ w)
    hessian = designX.T @ np.diag(p * (1 - p)) @ designX + prior_var_inv

    return {"beta": w, "hessian": hessian}


class BayesLogistic:

    def __init__(
        self,
        solver: Literal["IS", "Gibbs"] = "IS",
        n_draw: int = 10000,
        n_burn: int = 10000,
        prior_mu: float = 0.0,
        prior_var: float = 100.0,
        seed: int = 0,
        # must be multivariate
        proposed_distribution: st.rv_continuous | None = None,
    ):
        assert solver in ["IS"]

        self.solver_ = solver
        self.n_draw_ = n_draw
        self.n_burn_ = n_burn
        self.prior_mu_ = prior_mu
        self.prior_var_ = prior_var
        self.seed_ = seed
        self.prop_dist_ = proposed_distribution

        self.n_iter_ = self.n_draw_ + self.n_burn_
        self.rng_ = np.random.default_rng(self.seed_)

    def _fit_IS(self, designX: np.ndarray, y: np.ndarray):
        y_ = 2 * y - 1  # 转换成-1, 1

        if self.prop_dist_ is None:
            # 使用laplacian approximation来作为proposed distribution
            newton_res = newton_estimate(
                designX, y, self.prior_mu_, self.prior_var_,
            )
            self.prop_dist_ = st.multivariate_normal(
                newton_res["beta"], np.linalg.inv(newton_res["hessian"])
            )

        w = self.prop_dist_.rvs(size=self.n_draw_)  # L x p
        log_gw = self.prop_dist_.logpdf(w)

        log_pw = log_expit((w @ designX.T) * y_).sum(axis=1)  # L
        log_pw += (
            st.norm(loc=self.prior_mu_, scale=np.sqrt(self.prior_var_))
            .logpdf(w)
            .sum(axis=1)
        )

        self.importance_ratios_ = np.exp(log_pw - log_gw)
        self.beta_ = (w * self.importance_ratios_[:, None]).sum(
            axis=0
        ) / self.importance_ratios_.sum()

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_sample = X.shape[0]
        # n_beta = X.shape[1] + 1
        dmat = np.concatenate([np.ones((n_sample, 1)), X], axis=1)

        if self.solver_ == "IS":
            self._fit_IS(dmat, y)
        else:
            pass


def main():
    seed = 1
    # beta = np.array([0, -3, -1, 0, 1, 3])
    beta = np.random.randn(6)

    dat = simulate(
        n=100,
        sfunc="logistic",
        beta=beta,
        seed=seed,
    )

    model = BayesLogistic(
        solver="IS",
        prior_mu=0,
        prior_var=1.0,
        n_draw=1000,
        seed=seed,
        # proposed_distribution=st.multivariate_t(loc=[0] * 6, df=3),
    )
    model.fit(dat["X"], dat["y"])

    print(beta)
    print(model.prop_dist_.mean)
    print(model.beta_)
    # print(model.beta_, model.beta_std_)
    # plot_trace("./trace.png", dat["beta"], model.hist_)


if __name__ == "__main__":
    main()
