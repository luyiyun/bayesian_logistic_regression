from typing import Literal
import warnings

import numpy as np
import pandas as pd
import pymc
import arviz as az
from scipy import stats as st
from scipy.special import log_expit, expit
import seaborn as sns
import matplotlib.pyplot as plt

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

        if np.sqrt((grad**2).sum()) < tolerance:
            break

        hessian = designX.T @ np.diag(p * (1 - p)) @ designX + prior_var_inv
        w -= lr * np.linalg.inv(hessian) @ grad
    else:
        warnings.warn("newton method dose not converge.")

    p = expit(designX @ w)
    hessian = designX.T @ np.diag(p * (1 - p)) @ designX + prior_var_inv

    return {"beta": w, "hessian": hessian}


def calc_ess(is_ratios: np.ndarray) -> float:
    is_ratios_ = is_ratios / is_ratios.sum()
    return 1 / np.sum(is_ratios_**2)


class BayesLogistic:

    def __init__(
        self,
        solver: Literal["IS", "SIR", "Gibbs", "pymc"] = "IS",
        n_draw: int = 1000,
        n_burn: int = 1000,
        prior_mu: float = 0.0,
        prior_var: float = 100.0,
        seed: int = 0,
        # must be multivariate
        proposed_distribution: st.rv_continuous | None = None,
        replace: bool = True,
    ):
        assert solver in ["IS", "SIR", "pymc"]

        self.solver_ = solver
        self.n_draw_ = n_draw
        self.n_burn_ = n_burn
        self.prior_mu_ = prior_mu
        self.prior_var_ = prior_var
        self.seed_ = seed
        self.prop_dist_ = proposed_distribution
        self.replace_ = replace

        self.n_iter_ = self.n_draw_ + self.n_burn_
        self.rng_ = np.random.default_rng(self.seed_)

    def _fit_IS(self, designX: np.ndarray, y: np.ndarray):
        assert self.solver_ in ["IS", "SIR"]

        y_ = 2 * y - 1  # 转换成-1, 1

        if self.prop_dist_ is None:
            # 使用laplacian approximation来作为proposed distribution
            newton_res = newton_estimate(
                designX,
                y,
                self.prior_mu_,
                self.prior_var_,
            )
            self.prop_dist_ = st.multivariate_normal(
                newton_res["beta"], np.linalg.inv(newton_res["hessian"])
            )

        if self.solver_ == "IS":
            self.samples_IS_ = self.prop_dist_.rvs(size=self.n_draw_)  # L x p
        else:
            self.samples_IS_ = self.prop_dist_.rvs(size=self.n_iter_)  # L x p
        log_gw = self.prop_dist_.logpdf(self.samples_IS_)

        log_pw = log_expit((self.samples_IS_ @ designX.T) * y_).sum(
            axis=1
        )  # L
        log_pw += (
            st.norm(loc=self.prior_mu_, scale=np.sqrt(self.prior_var_))
            .logpdf(self.samples_IS_)
            .sum(axis=1)
        )

        self.importance_ratios_ = np.exp(log_pw - log_gw)
        if self.solver_ == "IS":
            self.beta_ = (
                self.samples_IS_ * self.importance_ratios_[:, None]
            ).sum(axis=0) / self.importance_ratios_.sum()
            self.ess_ = calc_ess(self.importance_ratios_)

    def _fit_SIR(self, designX: np.ndarray, y: np.ndarray):
        assert self.solver_ == "SIR"
        self._fit_IS(designX, y)
        iratios_norm = self.importance_ratios_ / self.importance_ratios_.sum()
        sample_indice = self.rng_.choice(
            self.n_iter_,
            self.n_draw_,
            replace=self.replace_,
            p=iratios_norm,
        )
        self.samples_SIR_ = self.samples_IS_[sample_indice, :]
        self.beta_ = self.samples_SIR_.mean(axis=0)

    def _fit_pymc(self, designX: np.ndarray, y: np.ndarray):
        with pymc.Model():
            beta = pymc.Normal(
                "beta",
                mu=np.full(designX.shape[1], self.prior_mu_),
                sigma=np.sqrt(self.prior_var_),
            )
            y = pymc.Bernoulli("y", logit_p=designX @ beta, observed=y)

            self.idata_pymc_ = pymc.sample(
                draws=self.n_draw_, tune=self.n_burn_, random_seed=self.seed_
            )
            self.samples_pymc_ = self.idata_pymc_["posterior"][
                "beta"
            ].to_numpy()
            self.beta_ = az.summary(self.idata_pymc_, hdi_prob=0.95)

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_sample = X.shape[0]
        # n_beta = X.shape[1] + 1
        dmat = np.concatenate([np.ones((n_sample, 1)), X], axis=1)

        if self.solver_ == "IS":
            self._fit_IS(dmat, y)
        elif self.solver_ == "pymc":
            self._fit_pymc(dmat, y)
        elif self.solver_ == "SIR":
            self._fit_SIR(dmat, y)
        else:
            raise NotImplementedError


def main():
    flag = "test_SIR"  # one of test_IS and test_SIR
    seed = 1
    beta = np.array([0, -3, -1, 0, 1, 3])

    dat = simulate(
        n=100,
        sfunc="logistic",
        beta=beta,
        seed=seed,
    )

    # pymc
    model_pymc = BayesLogistic(
        solver="pymc",
        prior_mu=0,
        prior_var=100.0,
        n_draw=1000,
        n_burn=10000,
        seed=seed,
        # proposed_distribution=st.multivariate_t(loc=[0] * 6, df=3),
    )
    model_pymc.fit(dat["X"], dat["y"])

    if flag == "test_IS":
        # is
        model_IS = BayesLogistic(
            solver="IS",
            prior_mu=0,
            prior_var=100.0,
            n_draw=1000,
            seed=seed,
        )
        model_IS.fit(dat["X"], dat["y"])
        print(f"ESS: {model_IS.ess_: .4f}")

        # plot
        samples = model_pymc.samples_pymc_
        samples = samples.reshape(-1, samples.shape[-1])

        fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(6, 4))
        axs = axs.flatten()
        for i in range(6):
            ax = axs[i]
            sns.kdeplot(x=samples[:, i], ax=ax)
            ax.axvline(x=beta[i], color="red")
            ax.axvline(x=model_pymc.beta_["mean"].iloc[i], color="blue")
            ax.axvline(x=model_IS.prop_dist_.mean[i], color="green")
            ax.axvline(x=model_IS.beta_[i], color="purple")
        fig.tight_layout()
        fig.savefig("./res/is_res.png")

        res_df = pd.DataFrame(
            {
                "true": beta,
                "MAP": model_IS.prop_dist_.mean,
                "MAP-IS": model_IS.beta_,
                "PYMC": model_pymc.beta_["mean"],
            }
        )
        print(res_df)
        res_df.to_csv("./res/IS_res.csv")
    elif flag == "test_SIR":
        # SIR
        model_SIR = BayesLogistic(
            solver="SIR",
            prior_mu=0,
            prior_var=100.0,
            n_draw=1000,
            n_burn=1000,
            seed=seed,
        )
        model_SIR.fit(dat["X"], dat["y"])
        # SIR without replacement
        model_SIR2 = BayesLogistic(
            solver="SIR",
            prior_mu=0,
            prior_var=100.0,
            n_draw=1000,
            n_burn=1000,
            seed=seed,
            replace=False,
        )
        model_SIR2.fit(dat["X"], dat["y"])

        # SIR bad proposed distribution
        prop_dist = st.multivariate_normal(np.full_like(beta, 0.0))
        model_SIR3 = BayesLogistic(
            solver="SIR",
            prior_mu=0,
            prior_var=100.0,
            n_draw=1000,
            n_burn=1000,
            seed=seed,
            proposed_distribution=prop_dist
        )
        model_SIR3.fit(dat["X"], dat["y"])
        # SIR without replacement, bad proposed distribution
        model_SIR4 = BayesLogistic(
            solver="SIR",
            prior_mu=0,
            prior_var=100.0,
            n_draw=1000,
            n_burn=1000,
            seed=seed,
            replace=False,
            proposed_distribution=prop_dist
        )
        model_SIR4.fit(dat["X"], dat["y"])

        samples_pymc = model_pymc.samples_pymc_.reshape(
            -1, model_pymc.samples_pymc_.shape[-1]
        )
        samples_all = [samples_pymc]
        for modeli in [model_SIR, model_SIR2, model_SIR3, model_SIR4]:
            samples_all.append(modeli.samples_SIR_)
        samples_df = pd.DataFrame(
            np.concatenate(samples_all),
            columns=[f"w{i}" for i in range(samples_pymc.shape[1])],
        )
        samples_df["method"] = np.repeat(
            ["pymc", "SIR", "SIRwoRep", "SIR_bad_prop", "SIRwoRep_bad_prop"],
            [sample.shape[0] for sample in samples_all]
        )

        samples_df = samples_df.melt(id_vars=["method"], var_name="beta")
        fg = sns.displot(
            data=samples_df,
            x="value",
            hue="method",
            col="beta",
            col_wrap=3,
            kind="ecdf",
            # kind="kde",
            # common_norm=False,
            height=3,
            facet_kws={"sharex": False, "sharey": False},
        )
        fg.savefig("./res/sir_res.png")
    else:
        raise NotImplementedError


if __name__ == "__main__":
    main()
