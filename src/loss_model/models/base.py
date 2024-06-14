"""Define the abstract base classes."""

from abc import ABC, abstractmethod
import pymc
import arviz
import pandas as pd

from .samplers import Samplers
from .opts import MCMCOptions


class BaseModel(ABC):
    """Abstract base class for models."""

    opts: MCMCOptions
    is_model_fit: bool = False
    sampler: Samplers = Samplers.NUTS
    model: pymc.Model | None = None
    inference: arviz.InferenceData | None = None

    @abstractmethod
    def sample(self):
        """Fit the model."""
        pass

    @abstractmethod
    def df(self) -> pd.DataFrame:
        """Return the posterior samples as a DataFrame."""
        pass
