from enum import Enum, auto


class Samplers(Enum):
    """Represents the available samplers."""

    Metropolis = auto()
    Slice = auto()
    Hamiltonian = auto()
    NUTS = auto()
    Gibbs = auto()
