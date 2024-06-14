"""Implement the cross-classified model from "Stochastic loss Reserving Using Bayesian MCMC Models, 2nd Edition" by Glenn Meyers.

Model Specification
-------------------
A triangle is defined with `W` rows (or accident years) and `D` columns (or development years).

For each accident year `w` and development year `d`, the model is defined as follows:

1. `logelr` ~ Normal(-0.4, sqrt(10))
2. `alpha[w]` ~ Normal(0, sqrt(10)) for `w`>=2, `alpha[1]` = 0
3. `beta[d]` ~ Normal(0, sqrt(10)) for `d`<=`D`, `beta[10]` = 0
4. `a[d]` ~ Uniform(0, 1) for each development year `d`
5. `sigma2[d]` = `a[d]` + `a[d+1]` + ... + `a[D]` for each development year `d`
6. `mu[w, d]` = `log(Premium[w])` + `logelr` + `alpha[w]` + `beta[d]`

Then `loss[w, d]` ~ LogNormal(`mu[w, d]`, `sigma2[d]`)

Parameters
----------
logelr : float
    The log of the expected ultimate loss ratio for the triangle.
alpha : array-like, shape (W,)
    The accident year effects, or the log of the expected ultimate loss ratios for each accident year.
beta : array-like, shape (D,)
    The development year effects, or the log of the percent of ultimate loss paid in each development year.
a : array-like, shape (D,)
    The development year variance components.
sigma2 : array-like, shape (D,)
    The development year variances. The sum of `a[i]`, `i>=d` for each development year `d`. Monotonically decreasing.
mu : array-like, shape (W, D)
    The mean of the log of the cumulative loss ratio for each cell in the triangle.
loss : array-like, shape (W, D)
    The cumulative loss amounts for each cell in the triangle.
"""

import numpy as np
import pymc
import pytensor.tensor as pt
import pandas as pd
from scipy.stats import gamma

from .base import BaseModel
from .opts import MCMCOptions


class CrossClassified(BaseModel):
    """Cross-classified model for stochastic loss reserving."""

    def __init__(
        self,
        prem: np.ndarray,
        loss_obs: np.ndarray,
        vars: list[str] | None = None,
        opts: MCMCOptions = MCMCOptions(
            n_burn_in=1000,
            n_samples=10_000,
            n_chains=4,
            n_cores=4,
            nuts_sampler="numpyro",
            random_seed=42,
            progressbar=True,
        ),
    ):
        super().__init__()

        self.vars = (
            vars
            if vars is not None
            else ["logelr", "alpha", "beta", "sigma2", "mu", "loss"]
        )

        self.prem = prem
        self.loss_obs = loss_obs

        self.opts = opts

        with pymc.Model() as self.model:
            # Define the priors

            # logelr ~ Normal(-0.4, 10)
            self.logelr = pymc.Normal("logelr", mu=-0.4, tau=1 / 10)

            # the random components of alpha and beta (recall that the first
            # element of alpha and the last element of beta are fixed to 0):
            self.r_alpha = pymc.Normal(
                "r_alpha", mu=0, tau=1 / 10, shape=self.loss_obs.shape[0] - 1
            )
            self.r_beta = pymc.Normal(
                "r_beta", mu=0, tau=1 / 10, shape=self.loss_obs.shape[1] - 1
            )

            # alpha is r_alpha, with a 0 prepended
            self.alpha = pymc.Deterministic(
                "alpha", pt.concatenate([[0], self.r_alpha])
            )

            # beta is r_beta, with a 0 appended
            self.beta = pymc.Deterministic("beta", pt.concatenate([self.r_beta, [0]]))

            # a_ig ~ InverseGamma(1, 1)
            self.a_ig = pymc.InverseGamma(
                "a_ig", alpha=1, beta=1, shape=self.loss_obs.shape[1]
            )

            # Compute sigma2 using the inverse cumulative sum
            self.sigma2 = pymc.Deterministic(
                "sigma2", pt.cumsum(1 / self.a_ig[::-1])[::-1]
            )

            prem_tensor = pt.as_tensor_variable(self.prem)

            # Define mu
            self.mu = pymc.Deterministic(
                "mu",
                pt.outer(
                    pt.log(prem_tensor),
                    pt.as_tensor_variable(np.ones(self.loss_obs.shape[1])),
                )
                + self.logelr
                + self.alpha[:, None]
                + self.beta,
            )

            # Define the likelihood
            self.loss = pymc.Lognormal(
                "loss", mu=self.mu, sigma=pt.sqrt(self.sigma2), observed=self.loss_obs
            )

    def sample(self):
        with self.model:
            self.trace = pymc.sample(
                draws=self.opts.n_samples,
                tune=self.opts.n_burn_in,
                chains=self.opts.n_chains,
                cores=self.opts.n_cores,
                nuts_sampler=self.opts.nuts_sampler,
                random_seed=self.opts.random_seed,
                progressbar=self.opts.progressbar,
                var_names=self.vars,
            )
        self.is_model_fit = True

    def df(self) -> pd.DataFrame:
        if not self.is_model_fit:
            raise ValueError("Model has not been fit yet.")
        return pymc.trace_to_dataframe(self.trace, varnames=self.vars)
