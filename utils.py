import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns


def simulate(
    beta: np.ndarray | None = None, n: int = 100, sfunc: str = "probit",
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    if beta is None:
        beta = rng.normal(size=(5,))
    x = rng.normal(size=(n, beta.shape[0] - 1))
    dmat = np.concatenate([np.ones((n, 1)), x], axis=1)
    logit = dmat @ beta
    if sfunc == "probit":
        p = st.norm.cdf(logit)
    elif sfunc == "logistic":
        p = 1 / (1 + np.exp(-logit))
    else:
        raise NotImplementedError
    y = rng.binomial(1, p)
    return {"X": x, "y": y, "beta": beta}


def plot_trace(fn, beta, hist, start=0):
    colors = sns.color_palette()
    w = hist["w"]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(w.shape[0] - start)
    for i in range(w.shape[1]):
        ax.plot(
            x,
            w[start:, i],
            label=f"w[{i}]",
            color=colors[i],
            marker=".",
            markersize=1,
            alpha=0.5,
        )
        ax.axhline(y=beta[i], color=colors[i])
    fig.legend()
    fig.savefig(fn)
