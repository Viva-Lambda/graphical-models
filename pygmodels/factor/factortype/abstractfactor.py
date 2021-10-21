"""
\file abstractfactor.py Abstract Factor object
"""
from abc import abstractmethod
from pygmodels.gtype.abstractobj import (
    AbstractGraphObj,
    AbstractNode,
)
from pygmodels.pgmtype.codomaintype import NumericValue, Outcome


class AbstractFactor(AbstractGraphObj):
    """"""

    @abstractmethod
    def scope_vars(self, f: Callable[[Set[AbstractNode]], Set[AbstractNode]]):
        """"""
        raise NotImplementedError

    @abstractmethod
    def partition_value(self, domains: List[FrozenSet[Tuple[str, NumericValue]]]):
        """"""
        raise NotImplementedError

    @abstractmethod
    def phi(self, scope_product: Set[Tuple[str, float]]):
        """"""
        raise NotImplementedError
