"""!
\file abstractrandvar.py Represents an abstract random variable
"""
from abc import ABC, abstractmethod

from pygmodels.graph.graphtype.abstractobj import AbstractNode
from pygmodels.value.codomain import Codomain, CodomainValue, Outcome
from pygmodels.value.domain import Domain
from pygmodels.value.value import NumericValue

PossibleOutcomes = Domain
AssociatedValueSet = Codomain


class AbstractRandomVariable(AbstractNode):
    """!
    Abstract random variable
    """

    @property
    @abstractmethod
    def inputs(self) -> PossibleOutcomes:
        """!
        Inputs, that is outcomes of the random variable.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def image(self) -> AssociatedValueSet:
        """!
        Image/Range of the random variable. It can be either a representation
        or the full range.
        """
        raise NotImplementedError

    @abstractmethod
    def p(self, out: CodomainValue) -> NumericValue:
        """!
        Measure the probability of the given outcome
        """
        raise NotImplementedError