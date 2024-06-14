from dataclasses import dataclass


@dataclass
class MCMCOptions:
    """Represents MCMC options."""

    n_burn_in: int = 1000
    n_samples: int = 10_000
    n_chains: int = 4
    n_cores: int = 4
    n_thin: int = 1
    nuts_sampler: str = "numpyro"

    random_seed: int = 42
    progressbar: bool = True

    def __post_init__(self):
        if self.n_burn_in < 0:
            raise ValueError("n_burn_in must be non-negative.")
        if self.n_samples < 0:
            raise ValueError("n_samples must be non-negative.")
        if self.n_chains < 1:
            raise ValueError("n_chains must be positive.")

        self.n_samples = (self.n_samples // self.n_chains) + 1

        self.n_total = self.n_burn_in + self.n_samples
